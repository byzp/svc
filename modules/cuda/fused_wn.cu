#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/OpMathType.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <torch/library.h>

namespace {
constexpr int kThreads = 256;

template <typename T>
__global__ void gate_forward_kernel(const T* a, const T* b, T* out, T* ts, T* ss,
                                   int64_t n, int64_t channels, int64_t time,
                                   int64_t bs0, int64_t bs1, int64_t bs2, bool broadcast) {
  using F = at::opmath_type<T>;
  for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
       i < n; i += int64_t(blockDim.x) * gridDim.x) {
    const int64_t row = i / time, t = i % time;
    const int64_t batch = row / channels, channel = row % channels;
    const int64_t ai = (batch * 2 * channels + channel) * time + t;
    const int64_t bi = batch * bs0 + channel * bs1 + (broadcast ? 0 : t * bs2);
    // Match PyTorch's intermediate dtype rounding, including FP16 add/activations.
    const T u = T(F(a[ai]) + F(b[bi]));
    const T v = T(F(a[ai + channels * time]) + F(b[bi + channels * bs1]));
    const T th = T(tanh(F(u)));
    const T sg = T(F(1) / (F(1) + exp(-F(v))));
    ts[i] = th;
    ss[i] = sg;
    out[i] = T(F(th) * F(sg));
  }
}

template <typename T>
__global__ void gate_backward_kernel(const T* grad, const T* ts, const T* ss, T* dx,
                                    int64_t n, int64_t channels, int64_t time) {
  using F = at::opmath_type<T>;
  for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
       i < n; i += int64_t(blockDim.x) * gridDim.x) {
    const int64_t row = i / time, t = i % time;
    const int64_t ai = ((row / channels) * 2 * channels + row % channels) * time + t;
    const F th = F(ts[i]), sg = F(ss[i]);
    const T dt = T(F(grad[i]) * sg), ds = T(F(grad[i]) * th);
    // ATen's real tanh/sigmoid backward uses scalar_t arithmetic, not
    // opmath arithmetic: FP16 rounds each intermediate subtraction/product.
    dx[ai] = dt * (T(1.) - ts[i] * ts[i]);
    dx[ai + channels * time] = ds * (T(1.) - ss[i]) * ss[i];
  }
}

template <typename X, typename R, bool Last>
__global__ void residual_forward_kernel(const X* x, const X* output, const R* residual,
                                       const X* mask, X* nx, X* no,
                                       int64_t n, int64_t channels, int64_t time) {
  using F = at::opmath_type<X>;
  for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
       i < n; i += int64_t(blockDim.x) * gridDim.x) {
    const int64_t row = i / time, t = i % time, batch = row / channels;
    const F m = F(mask[batch * time + t]);
    if constexpr (Last) {
      no[i] = X(F(X(F(output[i]) + F(residual[i]))) * m);
    } else {
      const int64_t ri = (batch * 2 * channels + row % channels) * time + t;
      nx[i] = X(F(X(F(x[i]) + F(residual[ri]))) * m);
      no[i] = X(F(output[i]) + F(residual[ri + channels * time]));
    }
  }
}

template <typename X, typename R, bool Last>
__global__ void residual_backward_kernel(const X* gx, const X* go, const X* mask,
                                        X* dx, R* dr,
                                        int64_t n, int64_t channels, int64_t time) {
  using F = at::opmath_type<X>;
  for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
       i < n; i += int64_t(blockDim.x) * gridDim.x) {
    const int64_t row = i / time, t = i % time, batch = row / channels;
    const X value = X(F(Last ? go[i] : gx[i]) * F(mask[batch * time + t]));
    dx[i] = value;
    if constexpr (Last) {
      dr[i] = R(value);
    } else {
      const int64_t ri = (batch * 2 * channels + row % channels) * time + t;
      dr[ri] = R(value);
      dr[ri + channels * time] = R(go[i]);
    }
  }
}

int blocks(int64_t n) { return int(std::min<int64_t>((n + kThreads - 1) / kThreads, 4096)); }

void check_tensor(const at::Tensor& t) {
  TORCH_CHECK(t.is_cuda() && t.dim() == 3 && t.numel() > 0,
              "Expected nonempty 3D CUDA tensor");
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> gate_forward(const at::Tensor& a, const at::Tensor& b) {
  check_tensor(a); check_tensor(b);
  TORCH_CHECK(a.is_contiguous() && a.size(1) % 2 == 0, "Invalid gate input layout");
  TORCH_CHECK(a.device() == b.device() && a.scalar_type() == b.scalar_type(), "Gate dtype/device mismatch");
  TORCH_CHECK(b.size(0) == a.size(0) && b.size(1) == a.size(1) &&
              (b.size(2) == 1 || b.size(2) == a.size(2)), "Invalid condition shape");
  const c10::cuda::CUDAGuard guard(a.device());
  const int64_t c = a.size(1) / 2, time = a.size(2), n = a.size(0) * c * time;
  auto out = at::empty({a.size(0), c, time}, a.options());
  auto ts = at::empty_like(out), ss = at::empty_like(out);
  AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, a.scalar_type(), "svc_gate_forward", [&] {
    gate_forward_kernel<scalar_t><<<blocks(n), kThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
        a.data_ptr<scalar_t>(), b.data_ptr<scalar_t>(), out.data_ptr<scalar_t>(),
        ts.data_ptr<scalar_t>(), ss.data_ptr<scalar_t>(), n, c, time,
        b.stride(0), b.stride(1), b.stride(2), b.size(2) == 1);
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {out, ts, ss};
}

at::Tensor gate_backward(const at::Tensor& grad, const at::Tensor& ts, const at::Tensor& ss) {
  check_tensor(grad);
  const c10::cuda::CUDAGuard guard(grad.device());
  const int64_t c = grad.size(1), time = grad.size(2), n = grad.numel();
  auto dx = at::empty({grad.size(0), 2*c, time}, grad.options());
  AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, grad.scalar_type(), "svc_gate_backward", [&] {
    gate_backward_kernel<scalar_t><<<blocks(n), kThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
        grad.data_ptr<scalar_t>(), ts.data_ptr<scalar_t>(), ss.data_ptr<scalar_t>(),
        dx.data_ptr<scalar_t>(), n, c, time);
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return dx;
}

std::tuple<at::Tensor, at::Tensor> residual_forward(
    const at::Tensor& x, const at::Tensor& output, const at::Tensor& residual,
    const at::Tensor& mask, bool last) {
  check_tensor(x); check_tensor(output); check_tensor(residual); check_tensor(mask);
  TORCH_CHECK(x.is_contiguous() && output.is_contiguous() && residual.is_contiguous() && mask.is_contiguous(),
              "Residual inputs must be contiguous");
  TORCH_CHECK(x.sizes() == output.sizes() && x.scalar_type() == output.scalar_type() &&
              x.scalar_type() == mask.scalar_type(), "Residual input dtype/shape mismatch");
  TORCH_CHECK(c10::promoteTypes(x.scalar_type(), residual.scalar_type()) == x.scalar_type(),
              "Residual accumulator must use the promoted dtype");
  TORCH_CHECK(x.device() == residual.device() && x.device() == output.device() && x.device() == mask.device(),
              "Residual device mismatch");
  TORCH_CHECK(mask.size(0) == x.size(0) && mask.size(1) == 1 && mask.size(2) == x.size(2) &&
              residual.size(0) == x.size(0) && residual.size(1) == x.size(1)*(last ? 1 : 2) &&
              residual.size(2) == x.size(2), "Residual shape mismatch");
  const c10::cuda::CUDAGuard guard(x.device());
  auto nx = last ? at::empty({0}, x.options()) : at::empty_like(x);
  auto no = at::empty_like(output);
  const int64_t n=x.numel(), c=x.size(1), time=x.size(2);
  AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, x.scalar_type(), "svc_residual_forward_x", [&] {
    using X = scalar_t;
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, residual.scalar_type(), "svc_residual_forward_r", [&] {
      using R = scalar_t;
      if (last) {
        residual_forward_kernel<X,R,true><<<blocks(n),kThreads,0,at::cuda::getCurrentCUDAStream()>>>(
            x.data_ptr<X>(),output.data_ptr<X>(),residual.data_ptr<R>(),mask.data_ptr<X>(),
            nx.data_ptr<X>(),no.data_ptr<X>(),n,c,time);
      } else {
        residual_forward_kernel<X,R,false><<<blocks(n),kThreads,0,at::cuda::getCurrentCUDAStream()>>>(
            x.data_ptr<X>(),output.data_ptr<X>(),residual.data_ptr<R>(),mask.data_ptr<X>(),
            nx.data_ptr<X>(),no.data_ptr<X>(),n,c,time);
      }
    });
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {nx,no};
}

std::tuple<at::Tensor, at::Tensor> residual_backward(
    const at::Tensor& gx, const at::Tensor& go, const at::Tensor& mask,
    at::ScalarType residual_type, bool last) {
  check_tensor(go);
  const c10::cuda::CUDAGuard guard(go.device());
  const int64_t n=go.numel(), c=go.size(1), time=go.size(2);
  auto dx=at::empty_like(go);
  auto dr=at::empty({go.size(0),c*(last ? 1 : 2),time},go.options().dtype(residual_type));
  AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, go.scalar_type(), "svc_residual_backward_x", [&] {
    using X=scalar_t;
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, residual_type, "svc_residual_backward_r", [&] {
      using R=scalar_t;
      if (last) {
        residual_backward_kernel<X,R,true><<<blocks(n),kThreads,0,at::cuda::getCurrentCUDAStream()>>>(
            gx.data_ptr<X>(),go.data_ptr<X>(),mask.data_ptr<X>(),dx.data_ptr<X>(),dr.data_ptr<R>(),n,c,time);
      } else {
        residual_backward_kernel<X,R,false><<<blocks(n),kThreads,0,at::cuda::getCurrentCUDAStream()>>>(
            gx.data_ptr<X>(),go.data_ptr<X>(),mask.data_ptr<X>(),dx.data_ptr<X>(),dr.data_ptr<R>(),n,c,time);
      }
    });
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {dx,dr};
}
} // namespace

TORCH_LIBRARY(svc_wn, m) {
  m.def("gate_forward(Tensor a, Tensor b) -> (Tensor, Tensor, Tensor)");
  m.def("gate_backward(Tensor grad, Tensor tanh, Tensor sigmoid) -> Tensor");
  m.def("residual_forward(Tensor x, Tensor output, Tensor residual, Tensor mask, bool last) -> (Tensor, Tensor)");
  m.def("residual_backward(Tensor gx, Tensor go, Tensor mask, ScalarType residual_type, bool last) -> (Tensor, Tensor)");
}
TORCH_LIBRARY_IMPL(svc_wn, CUDA, m) {
  m.impl("gate_forward", gate_forward);
  m.impl("gate_backward", gate_backward);
  m.impl("residual_forward", residual_forward);
  m.impl("residual_backward", residual_backward);
}
