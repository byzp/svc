"""Current PyTorch weight normalization with the existing checkpoint format."""

from torch.nn.utils import parametrize
from torch.nn.utils import remove_weight_norm as remove_legacy_weight_norm
from torch.nn.utils.parametrizations import weight_norm as parametrized_weight_norm


def weight_norm(module, name="weight", dim=0):
    parametrized_weight_norm(module, name=name, dim=dim)

    def save_legacy_names(module, state_dict, prefix, local_metadata):
        # PyTorch already accepts weight_g/weight_v on load. Also retain these
        # names on save for existing inference/export tools and checkpoints.
        for original, suffix in (("original0", "g"), ("original1", "v")):
            key = f"{prefix}parametrizations.{name}.{original}"
            if key in state_dict:
                state_dict[f"{prefix}{name}_{suffix}"] = state_dict.pop(key)

    module.register_state_dict_post_hook(save_legacy_names)
    return module


def remove_weight_norm(module, name="weight"):
    if parametrize.is_parametrized(module, name):
        return parametrize.remove_parametrizations(module, name, leave_parametrized=True)
    return remove_legacy_weight_norm(module, name)
