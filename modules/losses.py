import torch
from torch.nn import functional as F
from modules.fused_loss import feature_matching_loss
from modules.fused_reductions import adversarial_losses, kl_divergence


def feature_loss(fmap_r, fmap_g):
  pairs = [(r, g) for dr, dg in zip(fmap_r, fmap_g) for r, g in zip(dr, dg)]
  return feature_matching_loss([r for r, _ in pairs], [g for _, g in pairs])


def discriminator_loss(disc_real_outputs, disc_generated_outputs, return_details=True):
  pairs = list(zip(disc_real_outputs, disc_generated_outputs))
  if pairs and pairs[0][0].is_cuda:
    outputs = [x for pair in pairs for x in pair]
    loss, terms = adversarial_losses(outputs, [1.0, 0.0]*len(pairs), paired=True)
    if return_details:
      details = terms.detach().cpu().tolist()
      return loss, details[::2], details[1::2]
    return loss, [], []
  loss = 0
  r_losses = []
  g_losses = []
  for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
    dr = dr.float()
    dg = dg.float()
    r_loss = F.mse_loss(dr, torch.ones_like(dr))
    g_loss = F.mse_loss(dg, torch.zeros_like(dg))
    loss += (r_loss + g_loss)
    if return_details:
      r_losses.append(r_loss.item())
      g_losses.append(g_loss.item())

  return loss, r_losses, g_losses


def generator_loss(disc_outputs, return_details=True):
  if disc_outputs and disc_outputs[0].is_cuda:
    loss, terms = adversarial_losses(disc_outputs, [1.0]*len(disc_outputs))
    return loss, list(terms.unbind()) if return_details else []
  loss = 0
  gen_losses = []
  for dg in disc_outputs:
    dg = dg.float()
    l = F.mse_loss(dg, torch.ones_like(dg))
    if return_details:
      gen_losses.append(l)
    loss += l

  return loss, gen_losses


def kl_loss(z_p, logs_q, m_p, logs_p, z_mask):
  """
  z_p, logs_q: [b, h, t_t]
  m_p, logs_p: [b, h, t_t]
  """
  if z_p.is_cuda:
    return kl_divergence(z_p, logs_q, m_p, logs_p, z_mask)
  z_p = z_p.float()
  logs_q = logs_q.float()
  m_p = m_p.float()
  logs_p = logs_p.float()
  z_mask = z_mask.float()
  #print(logs_p)
  kl = logs_p - logs_q - 0.5
  kl += 0.5 * ((z_p - m_p)**2) * torch.exp(-2. * logs_p)
  kl = torch.sum(kl * z_mask)
  l = kl / torch.sum(z_mask)
  return l
