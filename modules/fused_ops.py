"""Fused WN operations; CUDA kernels live in cuda/fused_wn.cu.

When nvcc is unavailable, ordinary PyTorch operations run on the input device.
Other build failures remain explicit. No checkpoint parameters change.
"""
from pathlib import Path
import threading

import torch
from torch.autograd.function import once_differentiable
from modules.cuda_extension import cuda_compiler_available

_extension_lock = threading.Lock()
_loaded = False


def load_cuda_ops():
    global _loaded
    if _loaded:
        return True
    if not cuda_compiler_available():
        return False
    with _extension_lock:
        if _loaded:
            return True
        from torch.utils.cpp_extension import load
        load(
            name="svc_fused_wn",
            sources=[str(Path(__file__).with_name("cuda") / "fused_wn.cu")],
            extra_cuda_cflags=[
                "-O3", "--fmad=false",
                "-gencode=arch=compute_61,code=sm_61",
                "-gencode=arch=compute_70,code=sm_70",
                "-gencode=arch=compute_70,code=compute_70",
            ],
            is_python_module=False,
        )
        _loaded = True
        return True


class _Gate(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b):
        out, tanh, sigmoid = torch.ops.svc_wn.gate_forward(a.contiguous(), b)
        ctx.save_for_backward(tanh, sigmoid)
        ctx.condition_shape = b.shape
        return out

    @staticmethod
    @once_differentiable
    def backward(ctx, grad):
        tanh, sigmoid = ctx.saved_tensors
        grad_input = torch.ops.svc_wn.gate_backward(grad.contiguous(), tanh, sigmoid)
        return grad_input, grad_input.sum_to_size(ctx.condition_shape)


def gated_activation(a, b, channels):
    if a.shape[1] != 2 * channels:
        raise ValueError("Gate input must contain two channel groups")
    if a.is_cuda and load_cuda_ops():
        return _Gate.apply(a, b)
    summed = a + b
    return summed[:, :channels].tanh() * summed[:, channels:].sigmoid()


class _Residual(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, output, residual, mask, last):
        contiguous_mask = mask.contiguous()
        new_x, new_output = torch.ops.svc_wn.residual_forward(
            x.contiguous(), output.contiguous(), residual.contiguous(), contiguous_mask, last)
        ctx.last = last
        ctx.residual_dtype = residual.dtype
        ctx.mask_gradient = mask.requires_grad
        ctx.save_for_backward(contiguous_mask, *([x, output, residual] if ctx.mask_gradient else []))
        return new_x, new_output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_x, grad_output):
        mask = ctx.saved_tensors[0]
        dx, dr = torch.ops.svc_wn.residual_backward(
            grad_x.contiguous(), grad_output.contiguous(), mask,
            ctx.residual_dtype, ctx.last)
        dm = None
        if ctx.mask_gradient:
            x, output, residual = ctx.saved_tensors[1:]
            if ctx.last:
                dm = ((output + residual) * grad_output).sum(1, keepdim=True)
            else:
                dm = ((x + residual[:, :x.shape[1]]) * grad_x).sum(1, keepdim=True)
        return (None if ctx.last else dx), (dx if ctx.last else grad_output), dr, dm, None


def residual_step(x, output, residual, mask):
    if x.is_cuda and load_cuda_ops():
        return _Residual.apply(x, output, residual, mask, False)
    channels = x.shape[1]
    return (x + residual[:, :channels]) * mask, output + residual[:, channels:]


def final_skip(output, residual, mask):
    if output.is_cuda and load_cuda_ops():
        return _Residual.apply(output, output, residual, mask, True)[1]
    return (output + residual) * mask
