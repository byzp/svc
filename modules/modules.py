import torch
from torch import nn
from torch.nn import functional as F

import modules.commons as commons
from modules.cuda_graph import CudaGraphModule
from modules.fused_ops import residual_step, final_skip
from modules.weight_norm import remove_weight_norm as remove_weight_norm_modules
from modules.weight_norm import weight_norm as weight_norm_modules

LRELU_SLOPE = 0.1

Conv1dModel = nn.Conv1d



class LayerNorm(nn.Module):
  def __init__(self, channels, eps=1e-5):
    super().__init__()
    self.channels = channels
    self.eps = eps

    self.gamma = nn.Parameter(torch.ones(channels))
    self.beta = nn.Parameter(torch.zeros(channels))

  def forward(self, x):
    x = x.transpose(1, -1)
    x = F.layer_norm(x, (self.channels,), self.gamma, self.beta, self.eps)
    return x.transpose(1, -1)




class WN(CudaGraphModule):
  def __init__(self, hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=0, p_dropout=0):
    super(WN, self).__init__()
    assert(kernel_size % 2 == 1)
    self.hidden_channels =hidden_channels
    self.kernel_size = kernel_size,
    self.dilation_rate = dilation_rate
    self.n_layers = n_layers
    self.gin_channels = gin_channels
    self.p_dropout = p_dropout

    self.in_layers = torch.nn.ModuleList()
    self.res_skip_layers = torch.nn.ModuleList()
    self.drop = nn.Dropout(p_dropout)

    if gin_channels != 0:
      cond_layer = torch.nn.Conv1d(gin_channels, 2*hidden_channels*n_layers, 1)
      self.cond_layer = weight_norm_modules(cond_layer, name='weight')

    for i in range(n_layers):
      dilation = dilation_rate ** i
      padding = int((kernel_size * dilation - dilation) / 2)
      in_layer = Conv1dModel(hidden_channels, 2*hidden_channels, kernel_size,
                                 dilation=dilation, padding=padding)
      in_layer = weight_norm_modules(in_layer, name='weight')
      self.in_layers.append(in_layer)

      # last one is not necessary
      if i < n_layers - 1:
        res_skip_channels = 2 * hidden_channels
      else:
        res_skip_channels = hidden_channels

      res_skip_layer = torch.nn.Conv1d(hidden_channels, res_skip_channels, 1)
      res_skip_layer = weight_norm_modules(res_skip_layer, name='weight')
      self.res_skip_layers.append(res_skip_layer)

  def forward(self, x, x_mask, g=None, **kwargs):
    if self.training and torch.is_grad_enabled() and x.is_cuda:
      args = (x, x_mask) if g is None else (x, x_mask, g)
      return self._call_graphed(self._forward, args)
    return self._forward(x, x_mask, g, **kwargs)

  def _forward(self, x, x_mask, g=None, **kwargs):
    output = torch.zeros_like(x)

    if g is not None:
      g = self.cond_layer(g)

    for i in range(self.n_layers):
      x_in = self.in_layers[i](x)
      if g is not None:
        cond_offset = i * 2 * self.hidden_channels
        g_l = g[:,cond_offset:cond_offset+2*self.hidden_channels,:]
      else:
        g_l = torch.zeros_like(x_in)

      acts = commons.fused_add_tanh_sigmoid_multiply(
          x_in,
          g_l,
          self.hidden_channels)
      acts = self.drop(acts)

      res_skip_acts = self.res_skip_layers[i](acts)
      if i < self.n_layers - 1:
        x, output = residual_step(x, output, res_skip_acts, x_mask)
      else:
        output = final_skip(output, res_skip_acts, x_mask)
    return output

  def remove_weight_norm(self):
    if self.gin_channels != 0:
      remove_weight_norm_modules(self.cond_layer)
    for l in self.in_layers:
      remove_weight_norm_modules(l)
    for l in self.res_skip_layers:
      remove_weight_norm_modules(l)








class Flip(nn.Module):
  def forward(self, x, *args, reverse=False, **kwargs):
    x = torch.flip(x, [1])
    if not reverse:
      logdet = torch.zeros(x.size(0)).to(dtype=x.dtype, device=x.device)
      return x, logdet
    else:
      return x




class ResidualCouplingLayer(nn.Module):
  def __init__(self,
      channels,
      hidden_channels,
      kernel_size,
      dilation_rate,
      n_layers,
      p_dropout=0,
      gin_channels=0,
      mean_only=False,
      wn_sharing_parameter=None
      ):
    assert channels % 2 == 0, "channels should be divisible by 2"
    super().__init__()
    self.channels = channels
    self.hidden_channels = hidden_channels
    self.kernel_size = kernel_size
    self.dilation_rate = dilation_rate
    self.n_layers = n_layers
    self.half_channels = channels // 2
    self.mean_only = mean_only

    self.pre = nn.Conv1d(self.half_channels, hidden_channels, 1)
    self.enc = WN(hidden_channels, kernel_size, dilation_rate, n_layers, p_dropout=p_dropout, gin_channels=gin_channels) if wn_sharing_parameter is None else wn_sharing_parameter
    self.post = nn.Conv1d(hidden_channels, self.half_channels * (2 - mean_only), 1)
    self.post.weight.data.zero_()
    self.post.bias.data.zero_()

  def forward(self, x, x_mask, g=None, reverse=False):
    x0, x1 = torch.split(x, [self.half_channels]*2, 1)
    h = self.pre(x0) * x_mask
    h = self.enc(h, x_mask, g=g)
    stats = self.post(h) * x_mask
    if not self.mean_only:
      m, logs = torch.split(stats, [self.half_channels]*2, 1)
    else:
      m = stats
      logs = torch.zeros_like(m)

    if not reverse:
      x1 = m + x1 * torch.exp(logs) * x_mask
      x = torch.cat([x0, x1], 1)
      logdet = torch.sum(logs, [1,2])
      return x, logdet
    else:
      x1 = (x1 - m) * torch.exp(-logs) * x_mask
      x = torch.cat([x0, x1], 1)
      return x
