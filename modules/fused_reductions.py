"""Fused FP32 adversarial and KL reductions for CUDA training."""
from pathlib import Path
import threading

import torch
from torch.autograd.function import once_differentiable
from modules.cuda_extension import cuda_compiler_available

_lock = threading.Lock()
_loaded = False


def _load():
    global _loaded
    if _loaded:
        return True
    if not cuda_compiler_available():
        return False
    with _lock:
        if _loaded:
            return True
        from torch.utils.cpp_extension import load
        load(name="svc_fused_reductions",
             sources=[str(Path(__file__).with_name("cuda") / "loss_reductions.cu")],
             extra_cuda_cflags=["-O3", "--fmad=false",
                               "-gencode=arch=compute_61,code=sm_61",
                               "-gencode=arch=compute_70,code=sm_70",
                               "-gencode=arch=compute_70,code=compute_70"],
             is_python_module=False)
        _loaded = True
        return True


class _Adversarial(torch.autograd.Function):
    @staticmethod
    def forward(ctx, targets, paired, *outputs):
        ctx.targets = targets
        ctx.save_for_backward(*outputs)
        ctx.set_materialize_grads(False)
        return torch.ops.svc_reductions.mse_forward(list(outputs), targets, paired)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_total, grad_terms):
        grads = torch.ops.svc_reductions.mse_backward(
            list(ctx.saved_tensors), ctx.targets,
            grad_total.contiguous() if grad_total is not None else None,
            grad_terms.contiguous() if grad_terms is not None else None)
        return (None, None, *grads)


def adversarial_losses(outputs, targets, paired=False):
    outputs, targets = list(outputs), list(targets)
    if not 1 <= len(outputs) <= 64 or len(outputs) != len(targets) or (paired and len(outputs) % 2):
        raise ValueError('Expected 1..64 adversarial maps with matching targets and complete pairs')
    if outputs[0].is_cuda and _load():
        return _Adversarial.apply(targets, paired, *[x.contiguous() for x in outputs])
    terms = [torch.nn.functional.mse_loss(x.float(), torch.full_like(x, target, dtype=torch.float32))
             for x, target in zip(outputs, targets)]
    total = (sum(terms[i] + terms[i+1] for i in range(0, len(terms), 2)) if paired else sum(terms))
    return total, torch.stack(terms)


class _KL(torch.autograd.Function):
    @staticmethod
    def forward(ctx, z, logs_q, m, logs_p, mask):
        loss, denominator = torch.ops.svc_reductions.kl_forward(z, logs_q, m, logs_p, mask)
        ctx.mask_gradient = ctx.needs_input_grad[4]
        ctx.save_for_backward(z, m, logs_p, mask, denominator,
                              *([logs_q] if ctx.mask_gradient else []))
        return loss

    @staticmethod
    @once_differentiable
    def backward(ctx, grad):
        z, m, logs_p, mask, denominator = ctx.saved_tensors[:5]
        dz, dlq, dm, dlp = torch.ops.svc_reductions.kl_backward(
            grad.contiguous(), z, m, logs_p, mask, denominator)
        dmask = None
        if ctx.mask_gradient:
            logs_q = ctx.saved_tensors[5]
            term = logs_p - logs_q - 0.5 + 0.5 * (z-m).square() * (-2*logs_p).exp()
            # Learned masks are not used by training. Preserve ATen's operation
            # order for their optional derivative, including FP16 cast rounding.
            with torch.enable_grad():
                differentiable_mask = mask.detach().requires_grad_()
                native_loss = (term*differentiable_mask).sum() / differentiable_mask.sum()
                dmask, = torch.autograd.grad(native_loss, differentiable_mask, grad)
        return dz, dlq, dm, dlp, dmask


def kl_divergence(z, logs_q, m, logs_p, mask):
    if z.is_cuda and _load():
        return _KL.apply(*[x.float().contiguous() for x in (z, logs_q, m, logs_p, mask)])
    z, logs_q, m, logs_p, mask = (x.float() for x in (z, logs_q, m, logs_p, mask))
    term = logs_p - logs_q - 0.5
    term = term + 0.5 * (z - m).square() * (-2 * logs_p).exp()
    return (term * mask).sum() / mask.sum()
