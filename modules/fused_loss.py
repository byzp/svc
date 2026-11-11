"""Batch feature-matching reductions and gradients without FP32 feature copies."""
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
        load(name="svc_fused_feature_loss",
             sources=[str(Path(__file__).with_name("cuda") / "feature_loss.cu")],
             extra_cuda_cflags=["-O3", "--fmad=false",
                               "-gencode=arch=compute_61,code=sm_61",
                               "-gencode=arch=compute_70,code=sm_70",
                               "-gencode=arch=compute_70,code=compute_70"],
             is_python_module=False)
        _loaded = True
        return True


class _FeatureLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, count, *features):
        ctx.count = count
        ctx.save_for_backward(*features)
        return torch.ops.svc_feature_loss.forward(list(features[:count]), list(features[count:]))

    @staticmethod
    @once_differentiable
    def backward(ctx, grad):
        features = ctx.saved_tensors
        gradients = torch.ops.svc_feature_loss.backward(
            grad.contiguous(), list(features[:ctx.count]), list(features[ctx.count:]))
        return (None,) + (None,) * ctx.count + tuple(gradients)


def feature_matching_loss(real, fake):
    if not fake:
        return 0
    if fake[0].is_cuda and _load():
        real = [x.detach().contiguous() for x in real]
        fake = [x.contiguous() for x in fake]
        return _FeatureLoss.apply(len(fake), *real, *fake)
    loss = 0
    for r, f in zip(real, fake):
        loss = loss + torch.nn.functional.l1_loss(f.float(), r.float().detach())
    return loss * 2
