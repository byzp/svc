import logging
import multiprocessing
import os
import time
import signal
import sys
import atexit
from contextlib import nullcontext
from typing import Optional, List

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.amp import GradScaler, autocast
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
import modules.commons as commons
import utils
from data_utils import TextAudioCollate, TextAudioSpeakerLoader
from models import (
    MultiPeriodDiscriminator,
    SynthesizerTrn,
)
from modules.losses import discriminator_loss, feature_loss, generator_loss, kl_loss
from modules.mel_processing import mel_spectrogram_torch, spec_to_mel_torch


logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('numba').setLevel(logging.WARNING)

# SDPA 优化
torch.set_float32_matmul_precision('high')
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)

global_step = 0
start_time = time.time()
_terminate_requested = False


def main():
    """Single Node Multi GPUs Training"""
    assert torch.cuda.is_available(), "CPU training is not allowed."
    hps = utils.get_hparams()

    n_gpus =1# torch.cuda.device_count()
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(hps.train.port)
    
    print(f"[INFO] Starting training with {n_gpus} GPUs")

    if n_gpus == 1:
        run(0, 1, hps)
    else:
        # 多 GPU 使用 spawn
        mp.spawn(run, nprocs=n_gpus, args=(n_gpus, hps,))


def run(rank, n_gpus, hps):
    global global_step, start_time
    
    # 初始化进程组
    backend = 'gloo' if os.name == 'nt' else 'nccl'
    dist.init_process_group(backend=backend, init_method='env://', world_size=n_gpus, rank=rank)
    torch.manual_seed(hps.train.seed + rank)  # 每个进程使用不同的随机种子
    torch.cuda.set_device(rank)
    
    # Logger & TensorBoard
    logger = None
    writer = None
    writer_eval = None
    if rank == 0:
        logger = utils.get_logger(hps.model_dir)
        logger.info(hps)
        utils.check_git_hash(hps.model_dir)
        writer = SummaryWriter(log_dir=hps.model_dir)
        writer_eval = SummaryWriter(log_dir=os.path.join(hps.model_dir, "eval"))
    
    # 数据集和 DataLoader
    collate_fn = TextAudioCollate()
    all_in_mem = hps.train.all_in_mem
    train_dataset = TextAudioSpeakerLoader(hps.data.training_files, hps, all_in_mem=all_in_mem)
    
    # 使用 DistributedSampler 确保每个 GPU 获得不同的数据子集
    train_sampler = DistributedSampler(
        train_dataset, 
        num_replicas=n_gpus, 
        rank=rank, 
        shuffle=True,
        drop_last=True
    )
    
    num_workers = 0 if all_in_mem else min(4, multiprocessing.cpu_count())
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=hps.train.batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        prefetch_factor=4 if num_workers > 0 else None,
        collate_fn=collate_fn,
        drop_last=True
    )
    
    eval_loader = None
    if rank == 0:
        eval_dataset = TextAudioSpeakerLoader(
            hps.data.validation_files, hps, all_in_mem=all_in_mem, vol_aug=False
        )
        eval_loader = DataLoader(
            eval_dataset,
            num_workers=1,
            shuffle=False,
            batch_size=1,
            pin_memory=False,
            drop_last=False,
            collate_fn=collate_fn
        )

    # 创建模型
    net_g = SynthesizerTrn(
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        **hps.model
    ).cuda(rank)
    
    net_d = MultiPeriodDiscriminator(hps.model.use_spectral_norm).cuda(rank)
    

    # 优化器
    optim_g = torch.optim.AdamW(
        net_g.parameters(),
        hps.train.learning_rate,
        betas=hps.train.betas,
        eps=hps.train.eps
    )
    optim_d = torch.optim.AdamW(
        net_d.parameters(),
        hps.train.learning_rate,
        betas=hps.train.betas,
        eps=hps.train.eps
    )
    
    # DDP 包装
    net_g = DDP(net_g, device_ids=[rank], find_unused_parameters=False)
    net_d = DDP(net_d, device_ids=[rank], find_unused_parameters=False)

    # 加载模型
    epoch_str = 1
    skip_optimizer = getattr(hps.train, 'skip_optimizer', False)
    
    try:
        g_path = utils.latest_checkpoint_path(hps.model_dir, "G_*.pth")
        d_path = utils.latest_checkpoint_path(hps.model_dir, "D_*.pth")
        if g_path and d_path:
            _, _, _, epoch_str = utils.load_checkpoint(g_path, net_g, optim_g, skip_optimizer)
            _, _, _, epoch_str = utils.load_checkpoint(d_path, net_d, optim_d, skip_optimizer)
            epoch_str = max(epoch_str, 1)
            global_step = int(d_path[d_path.rfind("_")+1:d_path.rfind(".")]) + 1
            if rank == 0:
                logger.info(f"Loaded checkpoint from step {global_step - 1}, epoch {epoch_str}")
    except Exception as e:
        if rank == 0:
            print(f"[WARNING] Load checkpoint failed: {e}")
        epoch_str = 1
        global_step = 0
    
    if skip_optimizer:
        epoch_str = 1
        global_step = 0

    # 学习率调度器
    warmup_epochs = hps.train.warmup_epochs
    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(
        optim_g, gamma=hps.train.lr_decay, last_epoch=epoch_str - 2
    )
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(
        optim_d, gamma=hps.train.lr_decay, last_epoch=epoch_str - 2
    )

    # 混合精度
    scaler = GradScaler(enabled=hps.train.fp16_run)
    half_type = torch.bfloat16 if getattr(hps.train, 'half_type', 'fp16') == "bf16" else torch.float16

    # 训练循环
    for epoch in range(epoch_str, hps.train.epochs + 1):
        train_sampler.set_epoch(epoch)  # 确保每个 epoch shuffle 不同
        
        # Warmup learning rate
        if epoch <= warmup_epochs:
            lr = hps.train.learning_rate * epoch / warmup_epochs
            for param_group in optim_g.param_groups:
                param_group['lr'] = lr
            for param_group in optim_d.param_groups:
                param_group['lr'] = lr
        
        train_one_epoch(
            rank=rank,
            epoch=epoch,
            hps=hps,
            nets=(net_g, net_d),
            optims=(optim_g, optim_d),
            scaler=scaler,
            train_loader=train_loader,
            eval_loader=eval_loader,
            logger=logger,
            writers=(writer, writer_eval),
            half_type=half_type
        )
        
        # 更新学习率
        if epoch > warmup_epochs:
            scheduler_g.step()
            scheduler_d.step()
        
        if rank == 0:
            now = time.time()
            duration = now - start_time
            logger.info(f'====> Epoch: {epoch}, cost {duration:.2f} s')
            start_time = now
    
    # 清理
    dist.destroy_process_group()


def train_one_epoch(rank, epoch, hps, nets, optims, scaler, train_loader, eval_loader, logger, writers, half_type):
    global global_step
    
    net_g, net_d = nets
    optim_g, optim_d = optims
    writer, writer_eval = writers if writers else (None, None)
    
    accum_steps = getattr(hps.train, "grad_accumulate", 1)
    
    net_g.train()
    net_d.train()
    
    # 初始化梯度
    optim_g.zero_grad(set_to_none=True)
    optim_d.zero_grad(set_to_none=True)
    
    for batch_idx, items in enumerate(train_loader):
        # 加载数据
        c, f0, spec, y, spk, lengths, uv, volume = items
        
        g = spk.cuda(rank, non_blocking=True)
        spec = spec.cuda(rank, non_blocking=True)
        y = y.cuda(rank, non_blocking=True)
        c = c.cuda(rank, non_blocking=True)
        f0 = f0.cuda(rank, non_blocking=True)
        uv = uv.cuda(rank, non_blocking=True)
        lengths = lengths.cuda(rank, non_blocking=True)
        if volume is not None:
            volume = volume.cuda(rank, non_blocking=True)
        
        mel = spec_to_mel_torch(
            spec,
            hps.data.filter_length,
            hps.data.n_mel_channels,
            hps.data.sampling_rate,
            hps.data.mel_fmin,
            hps.data.mel_fmax
        )
        
        # 判断是否是累积步骤
        is_accumulating = (batch_idx + 1) % accum_steps != 0
        is_last_batch = (batch_idx + 1) == len(train_loader)
        should_step = not is_accumulating or is_last_batch
        
        # Generator Forward
        with autocast(enabled=hps.train.fp16_run, dtype=half_type, device_type='cuda'):
            (
                y_hat, ids_slice, z_mask,
                (z, z_p, m_p, logs_p, m_q, logs_q),
                pred_lf0, norm_lf0, lf0
            ) = net_g(c, f0, uv, spec, g=g, c_lengths=lengths, spec_lengths=lengths, vol=volume)

            y_mel = commons.slice_segments(
                mel, ids_slice, hps.train.segment_size // hps.data.hop_length
            )
            y_hat_mel = mel_spectrogram_torch(
                y_hat.squeeze(1),
                hps.data.filter_length,
                hps.data.n_mel_channels,
                hps.data.sampling_rate,
                hps.data.hop_length,
                hps.data.win_length,
                hps.data.mel_fmin,
                hps.data.mel_fmax
            )
            y_sliced = commons.slice_segments(
                y, ids_slice * hps.data.hop_length, hps.train.segment_size
            )

            # Discriminator Forward & Loss
            y_d_hat_r, y_d_hat_g, _, _ = net_d(y_sliced, y_hat.detach())

            with autocast(enabled=False, device_type='cuda'):
                loss_disc, losses_disc_r, losses_disc_g = discriminator_loss(y_d_hat_r, y_d_hat_g)
                loss_disc_scaled = loss_disc / accum_steps  # 梯度累积需要除以步数
        
        # Discriminator Backward
        d_sync_ctx = nullcontext() if should_step else net_d.no_sync()
        with d_sync_ctx:
            scaler.scale(loss_disc_scaled).backward()

        # Generator Loss
        with autocast(enabled=hps.train.fp16_run, dtype=half_type, device_type='cuda'):
            y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(y_sliced, y_hat)
            
            with autocast(enabled=False, device_type='cuda'):
                loss_mel = F.l1_loss(y_mel, y_hat_mel) * hps.train.c_mel
                loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * hps.train.c_kl
                loss_fm = feature_loss(fmap_r, fmap_g)
                loss_gen, losses_gen = generator_loss(y_d_hat_g)
                loss_lf0 = F.mse_loss(pred_lf0, lf0) if net_g.module.use_automatic_f0_prediction else 0
                
                loss_gen_all = loss_gen + loss_fm + loss_mel + loss_kl + loss_lf0
                loss_gen_scaled = loss_gen_all / accum_steps  # 梯度累积需要除以步数
        
        # Generator Backward
        g_sync_ctx = nullcontext() if should_step else net_g.no_sync()
        with g_sync_ctx:
            scaler.scale(loss_gen_scaled).backward()

        # Optimizer Step
        if should_step:
            # Discriminator
            scaler.unscale_(optim_d)
            grad_norm_d = commons.clip_grad_value_(net_d.parameters(), None)
            scaler.step(optim_d)
            optim_d.zero_grad(set_to_none=True)
            
            # Generator
            scaler.unscale_(optim_g)
            grad_norm_g = commons.clip_grad_value_(net_g.parameters(), None)
            scaler.step(optim_g)
            scaler.update()
            optim_g.zero_grad(set_to_none=True)
        else:
            grad_norm_d = 0.0
            grad_norm_g = 0.0

        # Logging
        if rank == 0 and global_step % hps.train.log_interval == 0:
            lr = optim_g.param_groups[0]['lr']
            losses = [loss_disc, loss_gen, loss_fm, loss_mel, loss_kl]
            reference_loss = sum(l.item() if hasattr(l, 'item') else l for l in losses)
            
            logger.info(
                f'Train Epoch: {epoch} [{100. * batch_idx / len(train_loader):.0f}%] '
                f'Step: {global_step}, LR: {lr:.6f}'
            )
            logger.info(
                f"Losses - D: {loss_disc:.4f}, G: {loss_gen:.4f}, FM: {loss_fm:.4f}, "
                f"Mel: {loss_mel:.4f}, KL: {loss_kl:.4f}, Total: {reference_loss:.4f}"
            )
            
            if writer:
                scalar_dict = {
                    "loss/g/total": loss_gen_all.item(),
                    "loss/d/total": loss_disc.item(),
                    "learning_rate": lr,
                    "grad_norm_d": grad_norm_d,
                    "grad_norm_g": grad_norm_g,
                    "loss/g/fm": loss_fm.item(),
                    "loss/g/mel": loss_mel.item(),
                    "loss/g/kl": loss_kl.item(),
                    "loss/g/lf0": loss_lf0.item() if hasattr(loss_lf0, 'item') else loss_lf0,
                }
                
                image_dict = {
                    "slice/mel_org": utils.plot_spectrogram_to_numpy(y_mel[0].data.cpu().numpy()),
                    "slice/mel_gen": utils.plot_spectrogram_to_numpy(y_hat_mel[0].data.cpu().numpy()),
                    "all/mel": utils.plot_spectrogram_to_numpy(mel[0].data.cpu().numpy())
                }

                if net_g.module.use_automatic_f0_prediction:
                    image_dict.update({
                        "all/lf0": utils.plot_data_to_numpy(
                            lf0[0, 0, :].cpu().numpy(),
                            pred_lf0[0, 0, :].detach().cpu().numpy()
                        ),
                        "all/norm_lf0": utils.plot_data_to_numpy(
                            lf0[0, 0, :].cpu().numpy(),
                            norm_lf0[0, 0, :].detach().cpu().numpy()
                        )
                    })

                utils.summarize(
                    writer=writer,
                    global_step=global_step,
                    images=image_dict,
                    scalars=scalar_dict
                )
        
        # Evaluation & Checkpoint
        if rank == 0 and global_step % hps.train.eval_interval == 0 and global_step != 0:
            evaluate(hps, net_g, eval_loader, writer_eval, global_step)
            
            utils.save_checkpoint(
                net_g, optim_g, hps.train.learning_rate, epoch,
                os.path.join(hps.model_dir, f"G_{global_step}.pth"),
                Async=True
            )
            utils.save_checkpoint(
                net_d, optim_d, hps.train.learning_rate, epoch,
                os.path.join(hps.model_dir, f"D_{global_step}.pth"),
                Async=True
            )
            
            keep_ckpts = getattr(hps.train, 'keep_ckpts', 0)
            if keep_ckpts > 0:
                utils.clean_checkpoints(
                    path_to_models=hps.model_dir,
                    n_ckpts_to_keep=keep_ckpts,
                    sort_by_time=True
                )

        global_step += 1


@torch.inference_mode()
def evaluate(hps, generator, eval_loader, writer_eval, step):
    """评估函数"""
    generator.eval()
    image_dict = {}
    audio_dict = {}
    
    for batch_idx, items in enumerate(eval_loader):
        c, f0, spec, y, spk, _, uv, volume = items
        
        g = spk[:1].cuda(0)
        spec = spec[:1].cuda(0)
        y = y[:1].cuda(0)
        c = c[:1].cuda(0)
        f0 = f0[:1].cuda(0)
        uv = uv[:1].cuda(0)
        if volume is not None:
            volume = volume[:1].cuda(0)
        
        mel = spec_to_mel_torch(
            spec,
            hps.data.filter_length,
            hps.data.n_mel_channels,
            hps.data.sampling_rate,
            hps.data.mel_fmin,
            hps.data.mel_fmax
        )
        
        y_hat, _ = generator.module.infer(c, f0, uv, g=g, vol=volume)

        y_hat_mel = mel_spectrogram_torch(
            y_hat.squeeze(1).float(),
            hps.data.filter_length,
            hps.data.n_mel_channels,
            hps.data.sampling_rate,
            hps.data.hop_length,
            hps.data.win_length,
            hps.data.mel_fmin,
            hps.data.mel_fmax
        )

        audio_dict[f"gen/audio_{batch_idx}"] = y_hat[0]
        audio_dict[f"gt/audio_{batch_idx}"] = y[0]
    
    # 使用最后一个 batch 的 mel
    image_dict["gen/mel"] = utils.plot_spectrogram_to_numpy(y_hat_mel[0].cpu().numpy())
    image_dict["gt/mel"] = utils.plot_spectrogram_to_numpy(mel[0].cpu().numpy())
    
    utils.summarize(
        writer=writer_eval,
        global_step=step,
        images=image_dict,
        audios=audio_dict,
        audio_sampling_rate=hps.data.sampling_rate
    )
    
    generator.train()


if __name__ == "__main__":
    main()
