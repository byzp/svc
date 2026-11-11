#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cub/block/block_reduce.cuh>
#include <torch/library.h>

namespace {
constexpr int kMaxFeatures = 64, kThreads = 256;
struct Features {
  const void* real[kMaxFeatures];
  const void* fake[kMaxFeatures];
  void* gradient[kMaxFeatures];
  int64_t size[kMaxFeatures];
  int offset[kMaxFeatures+1];
  at::ScalarType real_type[kMaxFeatures], fake_type[kMaxFeatures];
  int count;
};
static_assert(sizeof(Features) < 4000, "Kernel parameters must fit Pascal's parameter space");

__device__ float read(const void* p, at::ScalarType type, int64_t i) {
  switch (type) {
    case at::kHalf: return float(static_cast<const at::Half*>(p)[i]);
    case at::kBFloat16: return float(static_cast<const at::BFloat16*>(p)[i]);
    case at::kDouble: return float(static_cast<const double*>(p)[i]);
    default: return static_cast<const float*>(p)[i];
  }
}
__device__ void write(void* p, at::ScalarType type, int64_t i, float value) {
  switch (type) {
    case at::kHalf: static_cast<at::Half*>(p)[i]=at::Half(value); break;
    case at::kBFloat16: static_cast<at::BFloat16*>(p)[i]=at::BFloat16(value); break;
    case at::kDouble: static_cast<double*>(p)[i]=double(value); break;
    default: static_cast<float*>(p)[i]=value;
  }
}
__device__ int feature_index(const Features& p) {
  int lo=0, hi=p.count;
  while (lo<hi) {
    const int mid=(lo+hi)/2;
    if (p.offset[mid+1] <= int(blockIdx.x)) lo=mid+1;
    else hi=mid;
  }
  return lo;
}

__global__ void feature_partials(Features p, float* partials) {
  const int f=feature_index(p), block=int(blockIdx.x)-p.offset[f];
  const int64_t stride=int64_t(p.offset[f+1]-p.offset[f])*blockDim.x;
  float sum=0;
  for (int64_t i=int64_t(block)*blockDim.x+threadIdx.x; i<p.size[f]; i+=stride) {
    sum += fabsf(read(p.fake[f],p.fake_type[f],i)-read(p.real[f],p.real_type[f],i));
  }
  using Reduce=cub::BlockReduce<float,kThreads>;
  __shared__ typename Reduce::TempStorage storage;
  sum=Reduce(storage).Sum(sum);
  if (threadIdx.x==0) partials[blockIdx.x]=sum;
}

__global__ void feature_finish(Features p, const float* partials, float* loss) {
  __shared__ float means[kMaxFeatures];
  const int lane=threadIdx.x%32, warp=threadIdx.x/32;
  for (int f=warp; f<p.count; f+=kThreads/32) {
    float sum=0;
    for (int i=p.offset[f]+lane; i<p.offset[f+1]; i+=32) sum+=partials[i];
    for (int shift=16; shift>0; shift/=2) sum+=__shfl_down_sync(0xffffffff,sum,shift);
    if (lane==0) means[f]=sum/float(p.size[f]);
  }
  __syncthreads();
  if (threadIdx.x==0) {
    float sum=0;
    for (int f=0; f<p.count; ++f) sum+=means[f];
    *loss=sum*2.0f;
  }
}

__global__ void feature_backward(Features p, const float* grad) {
  const int f=feature_index(p), block=int(blockIdx.x)-p.offset[f];
  const int64_t stride=int64_t(p.offset[f+1]-p.offset[f])*blockDim.x;
  const float scale=(*grad*2.0f)/float(p.size[f]);
  for (int64_t i=int64_t(block)*blockDim.x+threadIdx.x; i<p.size[f]; i+=stride) {
    const float diff=read(p.fake[f],p.fake_type[f],i)-read(p.real[f],p.real_type[f],i);
    const float sign=float(diff>0)-float(diff<0);
    write(p.gradient[f],p.fake_type[f],i,scale*sign);
  }
}

Features parameters(at::TensorList real, at::TensorList fake) {
  TORCH_CHECK(real.size()==fake.size() && !fake.empty() && fake.size()<=kMaxFeatures,
              "Expected 1..64 matching feature pairs");
  Features p{};p.count=fake.size();
  for (int f=0; f<p.count; ++f) {
    const auto& r=real[f];const auto& g=fake[f];
    TORCH_CHECK(r.is_cuda() && g.is_cuda() && r.device()==g.device() && g.device()==fake[0].device() &&
                r.sizes()==g.sizes() && r.is_contiguous() && g.is_contiguous() && g.numel()>0,
                "Feature pairs must have matching nonempty contiguous CUDA shapes/devices");
    for (auto type : {r.scalar_type(),g.scalar_type()}) {
      TORCH_CHECK(type==at::kHalf || type==at::kFloat || type==at::kBFloat16 || type==at::kDouble,
                  "Unsupported feature dtype");
    }
    p.real[f]=r.const_data_ptr();p.fake[f]=g.const_data_ptr();
    p.real_type[f]=r.scalar_type();p.fake_type[f]=g.scalar_type();
    p.size[f]=g.numel();
    p.offset[f+1]=p.offset[f]+int(std::min<int64_t>((g.numel()+1023)/1024,256));
  }
  return p;
}

at::Tensor forward(at::TensorList real, at::TensorList fake) {
  auto p=parameters(real,fake);
  const c10::cuda::CUDAGuard guard(fake[0].device());
  auto options=fake[0].options().dtype(at::kFloat);
  auto partials=at::empty({p.offset[p.count]},options),loss=at::empty({},options);
  feature_partials<<<p.offset[p.count],kThreads,0,at::cuda::getCurrentCUDAStream()>>>(p,partials.data_ptr<float>());
  feature_finish<<<1,kThreads,0,at::cuda::getCurrentCUDAStream()>>>(p,partials.data_ptr<float>(),loss.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return loss;
}

std::vector<at::Tensor> backward(const at::Tensor& grad, at::TensorList real, at::TensorList fake) {
  auto p=parameters(real,fake);
  TORCH_CHECK(grad.is_cuda() && grad.device()==fake[0].device() && grad.scalar_type()==at::kFloat &&
              grad.numel()==1 && grad.is_contiguous(),"Expected scalar FP32 upstream gradient");
  const c10::cuda::CUDAGuard guard(fake[0].device());
  std::vector<at::Tensor> result;
  result.reserve(p.count);
  for (int f=0; f<p.count; ++f) {
    result.push_back(at::empty_like(fake[f]));
    p.gradient[f]=result.back().mutable_data_ptr();
  }
  feature_backward<<<p.offset[p.count],kThreads,0,at::cuda::getCurrentCUDAStream()>>>(p,grad.const_data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return result;
}
} // namespace

TORCH_LIBRARY(svc_feature_loss,m) {
  m.def("forward(Tensor[] real, Tensor[] fake) -> Tensor");
  m.def("backward(Tensor grad, Tensor[] real, Tensor[] fake) -> Tensor[]");
}
TORCH_LIBRARY_IMPL(svc_feature_loss,CUDA,m) {
  m.impl("forward",forward);
  m.impl("backward",backward);
}
