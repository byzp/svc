
import torch
from modules.fused_ops import gated_activation


def slice_pitch_segments(x, ids_str, segment_size=4):
  offsets = torch.arange(min(segment_size, x.size(1)), device=x.device)
  return x.gather(1, ids_str.unsqueeze(1) + offsets)

def rand_slice_segments_with_pitch(x, pitch, x_lengths=None, segment_size=4):
  b, d, t = x.size()
  if x_lengths is None:
    x_lengths = t
  ids_str_max = x_lengths - segment_size + 1
  ids_str = (torch.rand([b], device=x.device) * ids_str_max).to(dtype=torch.long)
  ret = slice_segments(x, ids_str, segment_size)
  ret_pitch = slice_pitch_segments(pitch, ids_str, segment_size)
  return ret, ret_pitch, ids_str

def init_weights(m, mean=0.0, std=0.01):
  classname = m.__class__.__name__
  if classname.find("Conv") != -1:
    m.weight.data.normal_(mean, std)


def get_padding(kernel_size, dilation=1):
  return int((kernel_size*dilation - dilation)/2)


def convert_pad_shape(pad_shape):
  l = pad_shape[::-1]
  pad_shape = [item for sublist in l for item in sublist]
  return pad_shape










def slice_segments(x, ids_str, segment_size=4):
  # Tensor slice bounds call .item() on CUDA once per sample and bound.
  # Gather the complete batch without waiting for indices on the CPU.
  offsets = torch.arange(min(segment_size, x.size(2)), device=x.device)
  indices = ids_str[:, None, None] + offsets
  return x.gather(2, indices.expand(-1, x.size(1), -1))












def subsequent_mask(length):
  mask = torch.tril(torch.ones(length, length)).unsqueeze(0).unsqueeze(0)
  return mask


def fused_add_tanh_sigmoid_multiply(input_a, input_b, n_channels):
  return gated_activation(input_a, input_b, n_channels)




def sequence_mask(length, max_length=None):
  if max_length is None:
    max_length = length.max()
  x = torch.arange(max_length, dtype=length.dtype, device=length.device)
  return x.unsqueeze(0) < length.unsqueeze(1)
