#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cub/block/block_reduce.cuh>
#include <torch/library.h>

namespace {
constexpr int kThreads=256, kMax=64;
struct Maps {
  const void* input[kMax];void* gradient[kMax];
  int64_t size[kMax];int offset[kMax+1];
  at::ScalarType type[kMax];float target[kMax], norm[kMax];int count;
};
static_assert(sizeof(Maps)<4000,"Pascal kernel parameter limit");
__device__ float read(const void* p, at::ScalarType t, int64_t i) {
  switch(t) {
    case at::kHalf:return float(static_cast<const at::Half*>(p)[i]);
    case at::kBFloat16:return float(static_cast<const at::BFloat16*>(p)[i]);
    case at::kDouble:return float(static_cast<const double*>(p)[i]);
    default:return static_cast<const float*>(p)[i];
  }
}
__device__ void write(void* p,at::ScalarType t,int64_t i,float v) {
  switch(t) {
    case at::kHalf:static_cast<at::Half*>(p)[i]=at::Half(v);break;
    case at::kBFloat16:static_cast<at::BFloat16*>(p)[i]=at::BFloat16(v);break;
    case at::kDouble:static_cast<double*>(p)[i]=double(v);break;
    default:static_cast<float*>(p)[i]=v;
  }
}
__device__ int index(const Maps& p) {
  int lo=0,hi=p.count;
  while(lo<hi) {int m=(lo+hi)/2;if(p.offset[m+1]<=int(blockIdx.x))lo=m+1;else hi=m;}
  return lo;
}
Maps parameters(at::TensorList inputs, at::ArrayRef<double> targets) {
  TORCH_CHECK(!inputs.empty() && inputs.size()<=kMax && inputs.size()==targets.size(),"Expected 1..64 adversarial maps");
  Maps p{};p.count=inputs.size();
  for(int i=0;i<p.count;++i) {
    const auto& x=inputs[i];const auto t=x.scalar_type();
    TORCH_CHECK(x.is_cuda() && x.device()==inputs[0].device() && x.is_contiguous() && x.numel()>0,"Invalid adversarial map");
    TORCH_CHECK(t==at::kHalf || t==at::kFloat || t==at::kDouble || t==at::kBFloat16,"Invalid adversarial dtype");
    p.input[i]=x.const_data_ptr();p.type[i]=t;p.size[i]=x.numel();p.target[i]=targets[i];
    p.norm[i]=float(2.0/double(x.numel()));
    p.offset[i+1]=p.offset[i]+int(std::min<int64_t>((x.numel()+1023)/1024,256));
  }
  return p;
}
__global__ void mse_partials(Maps p,float* partial) {
  const int f=index(p), b=int(blockIdx.x)-p.offset[f];
  const int64_t stride=int64_t(p.offset[f+1]-p.offset[f])*blockDim.x;
  float sum=0;
  for(int64_t i=int64_t(b)*blockDim.x+threadIdx.x;i<p.size[f];i+=stride) {
    const float d=read(p.input[f],p.type[f],i)-p.target[f];sum+=d*d;
  }
  using Reduce=cub::BlockReduce<float,kThreads>;
  __shared__ typename Reduce::TempStorage shared;
  sum=Reduce(shared).Sum(sum);if(threadIdx.x==0)partial[blockIdx.x]=sum;
}
__global__ void mse_finish(Maps p,const float* partial,float* total,float* terms,bool paired) {
  __shared__ float means[kMax];
  const int lane=threadIdx.x%32,warp=threadIdx.x/32;
  for(int f=warp;f<p.count;f+=kThreads/32) {
    float sum=0;for(int i=p.offset[f]+lane;i<p.offset[f+1];i+=32)sum+=partial[i];
    for(int s=16;s;s/=2)sum+=__shfl_down_sync(0xffffffff,sum,s);
    if(lane==0)means[f]=terms[f]=sum/float(p.size[f]);
  }
  __syncthreads();
  if(threadIdx.x==0) {
    float sum=0;
    if(paired)for(int f=0;f<p.count;f+=2)sum+=means[f]+means[f+1];
    else for(int f=0;f<p.count;++f)sum+=means[f];
    *total=sum;
  }
}
__global__ void mse_grad(Maps p,const float* gt,const float* gterms) {
  const int f=index(p),b=int(blockIdx.x)-p.offset[f];
  const int64_t stride=int64_t(p.offset[f+1]-p.offset[f])*blockDim.x;
  const float g=(gt ? *gt : 0.0f)+(gterms ? gterms[f] : 0.0f);
  for(int64_t i=int64_t(b)*blockDim.x+threadIdx.x;i<p.size[f];i+=stride) {
    const float d=read(p.input[f],p.type[f],i)-p.target[f];
    write(p.gradient[f],p.type[f],i,(p.norm[f]*d)*g);
  }
}
std::tuple<at::Tensor,at::Tensor> mse_forward(at::TensorList inputs,at::ArrayRef<double> targets,bool paired) {
  auto p=parameters(inputs,targets);TORCH_CHECK(!paired || p.count%2==0,"Paired maps must have even count");
  const c10::cuda::CUDAGuard guard(inputs[0].device());
  auto opt=inputs[0].options().dtype(at::kFloat);
  auto partial=at::empty({p.offset[p.count]},opt),total=at::empty({},opt),terms=at::empty({p.count},opt);
  mse_partials<<<p.offset[p.count],kThreads,0,at::cuda::getCurrentCUDAStream()>>>(p,partial.data_ptr<float>());
  mse_finish<<<1,kThreads,0,at::cuda::getCurrentCUDAStream()>>>(p,partial.data_ptr<float>(),total.data_ptr<float>(),terms.data_ptr<float>(),paired);
  C10_CUDA_KERNEL_LAUNCH_CHECK();return {total,terms};
}
std::vector<at::Tensor> mse_backward(at::TensorList inputs,at::ArrayRef<double> targets,
                                   const std::optional<at::Tensor>& gt,const std::optional<at::Tensor>& gterms) {
  auto p=parameters(inputs,targets);const c10::cuda::CUDAGuard guard(inputs[0].device());
  for(const auto& g:{gt,gterms})if(g)TORCH_CHECK(g->is_cuda() && g->device()==inputs[0].device() && g->scalar_type()==at::kFloat && g->is_contiguous(),"Invalid upstream MSE gradient");
  TORCH_CHECK(!gt || gt->numel()==1,"Total gradient must be scalar");
  TORCH_CHECK(!gterms || gterms->numel()==p.count,"Term gradient count mismatch");
  std::vector<at::Tensor> result;result.reserve(p.count);
  for(int i=0;i<p.count;++i){result.push_back(at::empty_like(inputs[i]));p.gradient[i]=result.back().mutable_data_ptr();}
  mse_grad<<<p.offset[p.count],kThreads,0,at::cuda::getCurrentCUDAStream()>>>(p,gt ? gt->const_data_ptr<float>():nullptr,gterms ? gterms->const_data_ptr<float>():nullptr);
  C10_CUDA_KERNEL_LAUNCH_CHECK();return result;
}

void check_kl(const at::Tensor& z,const at::Tensor& lq,const at::Tensor& m,const at::Tensor& lp,const at::Tensor& mask) {
  TORCH_CHECK(z.dim()==3 && z.numel()>0 && mask.dim()==3 && mask.size(0)==z.size(0) && mask.size(1)==1 && mask.size(2)==z.size(2),"Invalid KL shapes");
  for(const auto& x:{z,lq,m,lp,mask})TORCH_CHECK(x.is_cuda() && x.device()==z.device() && x.scalar_type()==at::kFloat && x.is_contiguous(),"Expected contiguous CUDA FP32 KL inputs");
  TORCH_CHECK(z.sizes()==lq.sizes() && z.sizes()==m.sizes() && z.sizes()==lp.sizes(),"KL shape mismatch");
}
__device__ float kl_term(float z,float lq,float m,float lp) {
  const float d=z-m;
  return ((lp-lq)-0.5f)+((0.5f*(d*d))*expf(-2.0f*lp));
}
__global__ void kl_partials(const float* z,const float* lq,const float* m,const float* lp,const float* mask,
                            float* num,float* den,int64_t n,int channels,int time) {
  float ns=0,ds=0;
  for(int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;i<n;i+=int64_t(gridDim.x)*blockDim.x) {
    const int64_t row=i/time;const int mi=(row/channels)*time+i%time;
    ns+=kl_term(z[i],lq[i],m[i],lp[i])*mask[mi];
    if(row%channels==0)ds+=mask[mi];
  }
  using Reduce=cub::BlockReduce<float,kThreads>;
  __shared__ typename Reduce::TempStorage shared;
  ns=Reduce(shared).Sum(ns);if(threadIdx.x==0)num[blockIdx.x]=ns;
  __syncthreads();ds=Reduce(shared).Sum(ds);if(threadIdx.x==0)den[blockIdx.x]=ds;
}
__global__ void kl_finish(const float* num,const float* den,int count,float* loss,float* denominator) {
  float ns=0,ds=0;for(int i=threadIdx.x;i<count;i+=blockDim.x){ns+=num[i];ds+=den[i];}
  using Reduce=cub::BlockReduce<float,kThreads>;
  __shared__ typename Reduce::TempStorage shared;
  ns=Reduce(shared).Sum(ns);__syncthreads();ds=Reduce(shared).Sum(ds);
  if(threadIdx.x==0){*loss=ns/ds;*denominator=ds;}
}
__global__ void kl_grad(const float* grad,const float* z,const float* m,const float* lp,const float* mask,
                        const float* denominator,float* dz,float* dlq,float* dm,float* dlp,
                        int64_t n,int channels,int time) {
  const float base=*grad / *denominator;
  for(int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;i<n;i+=int64_t(gridDim.x)*blockDim.x) {
    const int64_t row=i/time;const int mi=(row/channels)*time+i%time;
    const float g=base*mask[mi],d=z[i]-m[i],e=expf(-2.0f*lp[i]);
    const float dleft=g*e,dpow=dleft*0.5f;
    const float value=dpow*(2.0f*d);
    dz[i]=value;dm[i]=-value;dlq[i]=-g;
    dlp[i]=g+(-2.0f*((g*(0.5f*(d*d)))*e));
  }
}
std::tuple<at::Tensor,at::Tensor> kl_forward(const at::Tensor& z,const at::Tensor& lq,const at::Tensor& m,
                                          const at::Tensor& lp,const at::Tensor& mask) {
  check_kl(z,lq,m,lp,mask);const c10::cuda::CUDAGuard guard(z.device());
  const int count=int(std::min<int64_t>((z.numel()+1023)/1024,512));
  auto num=at::empty({count},z.options()),den=at::empty_like(num),loss=at::empty({},z.options()),denominator=at::empty({},z.options());
  kl_partials<<<count,kThreads,0,at::cuda::getCurrentCUDAStream()>>>(z.data_ptr<float>(),lq.data_ptr<float>(),m.data_ptr<float>(),lp.data_ptr<float>(),mask.data_ptr<float>(),num.data_ptr<float>(),den.data_ptr<float>(),z.numel(),z.size(1),z.size(2));
  kl_finish<<<1,kThreads,0,at::cuda::getCurrentCUDAStream()>>>(num.data_ptr<float>(),den.data_ptr<float>(),count,loss.data_ptr<float>(),denominator.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();return {loss,denominator};
}
std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor> kl_backward(
    const at::Tensor& grad,const at::Tensor& z,const at::Tensor& m,const at::Tensor& lp,
    const at::Tensor& mask,const at::Tensor& denominator) {
  check_kl(z,lp,m,lp,mask);const c10::cuda::CUDAGuard guard(z.device());
  TORCH_CHECK(grad.device()==z.device() && grad.scalar_type()==at::kFloat && grad.numel()==1 && grad.is_contiguous() && denominator.device()==z.device() && denominator.scalar_type()==at::kFloat && denominator.numel()==1,"Invalid KL scalar gradient");
  auto dz=at::empty_like(z),dm=at::empty_like(z),dlq=at::empty_like(z),dlp=at::empty_like(z);
  const int count=int(std::min<int64_t>((z.numel()+255)/256,4096));
  kl_grad<<<count,kThreads,0,at::cuda::getCurrentCUDAStream()>>>(grad.data_ptr<float>(),z.data_ptr<float>(),m.data_ptr<float>(),lp.data_ptr<float>(),mask.data_ptr<float>(),denominator.data_ptr<float>(),dz.data_ptr<float>(),dlq.data_ptr<float>(),dm.data_ptr<float>(),dlp.data_ptr<float>(),z.numel(),z.size(1),z.size(2));
  C10_CUDA_KERNEL_LAUNCH_CHECK();return {dz,dlq,dm,dlp};
}
} // namespace
TORCH_LIBRARY(svc_reductions,m) {
  m.def("mse_forward(Tensor[] inputs, float[] targets, bool paired) -> (Tensor, Tensor)");
  m.def("mse_backward(Tensor[] inputs, float[] targets, Tensor? total_grad, Tensor? term_grads) -> Tensor[]");
  m.def("kl_forward(Tensor z, Tensor lq, Tensor mean, Tensor lp, Tensor mask) -> (Tensor, Tensor)");
  m.def("kl_backward(Tensor grad, Tensor z, Tensor mean, Tensor lp, Tensor mask, Tensor denominator) -> (Tensor, Tensor, Tensor, Tensor)");
}
TORCH_LIBRARY_IMPL(svc_reductions,CUDA,m) {
  m.impl("mse_forward",mse_forward);m.impl("mse_backward",mse_backward);
  m.impl("kl_forward",kl_forward);m.impl("kl_backward",kl_backward);
}
