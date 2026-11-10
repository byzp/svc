import logging
import multiprocessing
import os
import signal
import time
from contextlib import nullcontext
from itertools import islice
from pathlib import Path

# Configure the allocator before torch (or a spawned worker) initializes CUDA.
_alloc_env = "PYTORCH_ALLOC_CONF" if "PYTORCH_ALLOC_CONF" in os.environ else "PYTORCH_CUDA_ALLOC_CONF"
_alloc_conf = os.environ.get(_alloc_env, "")
if "expandable_segments" not in _alloc_conf:
    os.environ[_alloc_env] = ",".join(filter(None, (_alloc_conf, "expandable_segments:True")))

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.amp import GradScaler, autocast
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tensorboardX import SummaryWriter
import modules.commons as commons
import utils
from data_utils import TextAudioCollate, TextAudioSpeakerLoader
from models import (
    MultiPeriodDiscriminator,
    SynthesizerTrn,
)
from modules.losses import discriminator_loss, feature_loss, generator_loss, kl_loss
from modules.cuda_graph import CudaGraphModule
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


class TrainingInterrupt:
    """Defer the first Ctrl+C to an optimizer boundary; a second aborts immediately."""

    def __init__(self, stop_flag=None, announce=True):
        self.stop_flag = stop_flag if stop_flag is not None else multiprocessing.get_context("spawn").Value(
            "b", False, lock=False,
        )
        self.announce = announce
        self.received = False

    def __enter__(self):
        self.previous_handler = signal.signal(signal.SIGINT, self.handle_signal)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        signal.signal(signal.SIGINT, self.previous_handler)

    def handle_signal(self, signum, frame):
        if self.received:
            raise KeyboardInterrupt
        self.received = True
        # A lock-free flag also lets the spawn parent request a stop in every rank.
        self.stop_flag.value = True
        if self.announce:
            os.write(2, b"\n[INFO] Ctrl+C received; finishing the current optimizer step, then saving and exiting. "
                     b"Press Ctrl+C again to force exit.\n")

    def should_stop(self):
        requested = bool(self.stop_flag.value)
        if dist.is_initialized():
            # Every rank must make the same decision, even if SIGINT reaches only one.
            device = torch.cuda.current_device() if dist.get_backend() == "nccl" else "cpu"
            stop = torch.tensor(int(requested), dtype=torch.int32, device=device)
            dist.all_reduce(stop, op=dist.ReduceOp.MAX)
            requested = bool(stop.item())
        return requested


def ignore_worker_sigint(worker_id):
    """The training process handles Ctrl+C; keep prefetched CPU batches available."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def unwrap(model):
    return model.module if isinstance(model, DDP) else model


def gradient_norm(parameters):
    """Keep norm reductions on-device; only logging needs a host scalar."""
    gradients = [p.grad for p in parameters if p.grad is not None]
    return torch.nn.utils.get_total_norm(gradients, foreach=True) if gradients else 0.0


def amp_settings(train):
    enabled = getattr(train, "fp16_run", True)
    half_type = getattr(train, "half_type", "fp16")
    if half_type not in {"fp16", "bf16"}:
        raise ValueError("train.half_type must be fp16 or bf16")
    if enabled and half_type == "bf16" and torch.cuda.get_device_capability()[0] < 8:
        raise ValueError("BF16 AMP requires an Ampere or newer GPU; use half_type=fp16")
    return enabled, torch.bfloat16 if half_type == "bf16" else torch.float16


def gradient_accumulation_steps(train):
    steps = getattr(train, "grad_accumulate", 1)
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
        raise ValueError("train.grad_accumulate must be a positive integer")
    return steps


def set_learning_rate(hps, optims, epoch):
    """Derive the LR from the epoch, including when optimizer state is absent."""
    warmup_epochs = hps.train.warmup_epochs
    if warmup_epochs < 0:
        raise ValueError("train.warmup_epochs must be nonnegative")
    if epoch <= warmup_epochs:
        learning_rate = hps.train.learning_rate * epoch / warmup_epochs
    else:
        decay_epochs = max(0, epoch - warmup_epochs - 1)
        learning_rate = hps.train.learning_rate * hps.train.lr_decay ** decay_epochs
    for optim in optims:
        for group in optim.param_groups:
            group["lr"] = learning_rate


def load_training_state(hps, nets, optims, scaler, batches_per_epoch, logger):
    """Resume a matching G/D pair, or use a lone generator as pretrained weights."""
    global global_step
    global_step = 0
    paths = {}
    for prefix in ("G", "D"):
        paths[prefix] = {
            int(path.stem[2:]): path
            for path in Path(hps.model_dir).glob(f"{prefix}_*.pth")
            if path.stem[2:].isdigit()
        }
    if not paths["G"]:
        return 1, 0
    paired = paths["G"].keys() & paths["D"].keys()
    skip_optimizer = getattr(hps.train, "skip_optimizer", False)
    has_pair = bool(paired)
    step = max(paired) if has_pair else max(paths["G"])
    if has_pair and step != max(paths["G"]):
        logger.warning("Ignoring an unpaired generator checkpoint; resuming step %s", step)
    checkpoints = []
    for index, prefix in enumerate(("G", "D") if has_pair else ("G",)):
        path = paths[prefix][step]
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        # Keep FP32 master weights even when pretrained weights were saved in FP16.
        unwrap(nets[index]).load_state_dict(checkpoint["model"], strict=True)
        if has_pair and not skip_optimizer and checkpoint.get("optimizer") is not None:
            optimizer_state = checkpoint["optimizer"]
            # Preserve the selected implementation when resuming an older checkpoint.
            # Set these before loading so fused AdamW's step tensors move to CUDA too.
            for group in optimizer_state["param_groups"]:
                for option in ("fused", "foreach"):
                    group[option] = optims[index].defaults[option]
            optims[index].load_state_dict(optimizer_state)
        checkpoints.append(checkpoint)
        logger.info("Loaded %s", path)
    if not has_pair or skip_optimizer:
        logger.info("Using checkpoint weights for initialization; starting optimizer/step state fresh")
        return 1, 0
    checkpoint, discriminator = checkpoints
    if any(checkpoint.get(key) != discriminator.get(key)
           for key in ("iteration", "global_step", "batch_idx")):
        raise ValueError("Generator and discriminator checkpoint training states do not match")
    if checkpoint.get("iteration", 0) == 0 and "global_step" not in checkpoint:
        return 1, 0
    if checkpoint.get("scaler") and scaler.is_enabled():
        scaler.load_state_dict(checkpoint["scaler"])
    epoch = max(int(checkpoint["iteration"]), 1)
    global_step = int(checkpoint.get("global_step", step + 1))
    batch_idx = int(checkpoint.get("batch_idx", global_step % batches_per_epoch))
    if not 0 <= batch_idx <= batches_per_epoch:
        raise ValueError("Checkpoint batch offset is incompatible with the current DataLoader")
    if batch_idx == batches_per_epoch:
        epoch += 1
        batch_idx = 0
    logger.info("Resuming epoch %s, batch %s, completed steps %s", epoch, batch_idx, global_step)
    return epoch, batch_idx


def save_training_state(hps, nets, optims, scaler, epoch, batch_idx):
    """Write complete checkpoints synchronously before retention can delete old ones."""
    for prefix, net, optim in zip(("G", "D"), nets, optims):
        path = Path(hps.model_dir) / f"{prefix}_{global_step}.pth"
        temporary = path.with_suffix(".pth.tmp")
        try:
            torch.save({
                "model": unwrap(net).state_dict(),
                "optimizer": optim.state_dict(),
                "learning_rate": optim.param_groups[0]["lr"],
                "iteration": epoch,
                "global_step": global_step,
                "batch_idx": batch_idx,
                "scaler": scaler.state_dict(),
            }, temporary)
            os.replace(temporary, path)
        finally:
            if temporary.exists():
                temporary.unlink()
    keep_ckpts = getattr(hps.train, "keep_ckpts", 0)
    if keep_ckpts > 0:
        utils.clean_checkpoints(hps.model_dir, keep_ckpts, sort_by_time=True)


def main():
    """Single-node CUDA training; keep single-GPU operation as the default."""
    if not torch.cuda.is_available():
        raise RuntimeError("Training requires a CUDA GPU")
    hps = utils.get_hparams()

    n_gpus = getattr(hps.train, "n_gpus", 1)
    if not isinstance(n_gpus, int) or not 1 <= n_gpus <= torch.cuda.device_count():
        raise ValueError("train.n_gpus must be between 1 and the available CUDA device count")
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(hps.train.port)

    print(f"[INFO] Starting training with {n_gpus} GPUs")

    if n_gpus == 1:
        run(0, 1, hps)
    else:
        # 多 GPU 使用 spawn
        stop_flag = multiprocessing.get_context("spawn").Value("b", False, lock=False)
        with TrainingInterrupt(stop_flag):
            mp.spawn(run, nprocs=n_gpus, args=(n_gpus, hps, stop_flag))


def run(rank, n_gpus, hps, stop_flag=None):
    with TrainingInterrupt(stop_flag, announce=n_gpus == 1) as interrupt:
        torch.cuda.set_device(rank)
        try:
            if n_gpus > 1:
                backend = 'gloo' if os.name == 'nt' else 'nccl'
                dist.init_process_group(backend=backend, init_method='env://', world_size=n_gpus, rank=rank)
            run_training(rank, n_gpus, hps, interrupt=interrupt)
        finally:
            if dist.is_initialized():
                dist.destroy_process_group()


def run_training(rank, n_gpus, hps, interrupt=None):
    torch.manual_seed(hps.train.seed + rank)
    torch.backends.cudnn.benchmark = getattr(hps.train, "cudnn_benchmark", True)
    amp_enabled, half_type = amp_settings(hps.train)
    gradient_accumulation_steps(hps.train)
    if hps.train.log_interval < 1 or hps.train.eval_interval < 1:
        raise ValueError("train.log_interval and train.eval_interval must be positive")
    image_interval = getattr(hps.train, "image_interval", hps.train.eval_interval)
    if not isinstance(image_interval, int) or image_interval < 0:
        raise ValueError("train.image_interval must be a nonnegative integer (0 disables images)")

    # Logger & TensorBoard
    logger = None
    writer = None
    writer_eval = None
    if rank == 0:
        logger = utils.get_logger(hps.model_dir)
        logger.info(hps)
        utils.check_git_hash(hps.model_dir)
        logger.info("AMP: %s, dtype: %s", amp_enabled, half_type)

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

    num_workers = 0 if all_in_mem else getattr(hps.train, "num_workers", min(4, multiprocessing.cpu_count()))
    prefetch_factor = getattr(hps.train, "prefetch_factor", 4)
    if not isinstance(num_workers, int) or num_workers < 0:
        raise ValueError("train.num_workers must be a nonnegative integer")
    if not isinstance(prefetch_factor, int) or prefetch_factor < 1:
        raise ValueError("train.prefetch_factor must be a positive integer")

    train_loader = DataLoader(
        train_dataset,
        batch_size=hps.train.batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        collate_fn=collate_fn,
        worker_init_fn=ignore_worker_sigint,
        drop_last=True
    )
    if len(train_loader) == 0:
        raise ValueError(
            "No complete training batch: add training samples or reduce train.batch_size "
            f"(samples={len(train_dataset)}, batch_size={hps.train.batch_size}, GPUs={n_gpus})"
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
            collate_fn=collate_fn,
            worker_init_fn=ignore_worker_sigint,
        )

    # 创建模型
    net_g = SynthesizerTrn(
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        **hps.model
    ).cuda(rank)

    net_d = MultiPeriodDiscriminator(hps.model.use_spectral_norm).cuda(rank)


    # 优化器
    fused_optimizer = getattr(hps.train, "fused_optimizer", True)
    optim_g = torch.optim.AdamW(
        net_g.parameters(),
        hps.train.learning_rate,
        betas=hps.train.betas,
        eps=hps.train.eps,
        fused=fused_optimizer,
    )
    optim_d = torch.optim.AdamW(
        net_d.parameters(),
        hps.train.learning_rate,
        betas=hps.train.betas,
        eps=hps.train.eps,
        fused=fused_optimizer,
    )

    # Load before DDP so master weights remain FP32 and bad checkpoints fail visibly.
    scaler = GradScaler("cuda", enabled=amp_enabled and half_type == torch.float16)
    epoch_str, start_batch = load_training_state(
        hps, (net_g, net_d), (optim_g, optim_d), scaler, len(train_loader),
        logger or logging.getLogger(__name__),
    )
    if n_gpus > 1:
        net_g = DDP(net_g, device_ids=[rank], find_unused_parameters=False)
        net_d = DDP(net_d, device_ids=[rank], find_unused_parameters=False)

    # 训练循环
    try:
        if rank == 0:
            writer = SummaryWriter(log_dir=hps.model_dir)
            writer_eval = SummaryWriter(log_dir=os.path.join(hps.model_dir, "eval"))
        for epoch in range(epoch_str, hps.train.epochs + 1):
            start_time = time.time()
            train_sampler.set_epoch(epoch)
            set_learning_rate(hps, (optim_g, optim_d), epoch)
            interrupted = train_one_epoch(
                rank=rank, epoch=epoch, hps=hps,
                nets=(net_g, net_d), optims=(optim_g, optim_d), scaler=scaler,
                train_loader=train_loader, eval_loader=eval_loader, logger=logger,
                writers=(writer, writer_eval), half_type=half_type,
                start_batch=start_batch if epoch == epoch_str else 0,
                interrupt=interrupt,
            )
            if interrupted:
                return
            if rank == 0:
                logger.info('====> Epoch: %s, cost %.2f s', epoch, time.time() - start_time)
                if epoch == hps.train.epochs:
                    save_training_state(hps, (net_g, net_d), (optim_g, optim_d), scaler, epoch, len(train_loader))
            if n_gpus > 1:
                dist.barrier()
    finally:
        for summary_writer in (writer, writer_eval):
            if summary_writer is not None:
                summary_writer.close()


def cuda_batches(loader, device):
    """Overlap the next pinned CPU batch transfer with the current GPU step."""
    batches = iter(loader)
    copy_stream = torch.cuda.Stream(device=device)
    main_stream = torch.cuda.current_stream(device)

    def preload():
        try:
            items = next(batches)
        except StopIteration:
            return None
        with torch.cuda.stream(copy_stream):
            return tuple(item.cuda(device, non_blocking=True) if item is not None else None
                         for item in items)

    pending = preload()
    while pending is not None:
        # Wait for this batch before enqueuing the following transfer. Waiting
        # afterwards would also serialize that transfer with the current step.
        main_stream.wait_stream(copy_stream)
        current = pending
        for item in current:
            if item is not None:
                item.record_stream(main_stream)
        pending = preload()
        yield current


def train_one_epoch(
    rank, epoch, hps, nets, optims, scaler, train_loader, eval_loader,
    logger, writers, half_type, start_batch=0, interrupt=None,
):
    global global_step

    net_g, net_d = nets
    optim_g, optim_d = optims
    writer, writer_eval = writers if writers else (None, None)

    accum_steps = gradient_accumulation_steps(hps.train)
    for module in net_g.modules():
        if isinstance(module, CudaGraphModule):
            module.set_graph_gradient_accumulation(accum_steps > 1)
    amp_enabled = getattr(hps.train, "fp16_run", True)
    image_interval = getattr(hps.train, "image_interval", hps.train.eval_interval)
    next_eval_step = (global_step // hps.train.eval_interval + 1) * hps.train.eval_interval

    net_g.train()
    net_d.train()

    # 初始化梯度
    optim_g.zero_grad(set_to_none=True)
    optim_d.zero_grad(set_to_none=True)

    # Resume skips CPU batches before copying, so skipped data never reaches CUDA.
    epoch_batches = islice(train_loader, start_batch, None)
    for batch_idx, items in enumerate(cuda_batches(epoch_batches, rank), start=start_batch):
        # 加载数据
        c, f0, spec, y, spk, lengths, uv, volume = items

        g = spk

        # The final accumulation window may contain fewer than accum_steps batches.
        window_start = start_batch + ((batch_idx - start_batch) // accum_steps) * accum_steps
        window_size = min(accum_steps, len(train_loader) - window_start)
        should_step = batch_idx + 1 == window_start + window_size
        should_log = rank == 0 and (global_step + 1) % hps.train.log_interval == 0

        # DDP no_sync must enclose both the forward and the backward pass.
        g_sync_ctx = net_g.no_sync() if isinstance(net_g, DDP) and not should_step else nullcontext()
        with g_sync_ctx:
            with autocast('cuda', enabled=amp_enabled, dtype=half_type):
                (
                    y_hat, ids_slice, z_mask,
                    (_, z_p, m_p, logs_p, _, logs_q),
                    pred_lf0, norm_lf0, lf0,
                ) = net_g(c, f0, uv, spec, g=g, c_lengths=lengths, spec_lengths=lengths, vol=volume)

            # FFT, mel projection and losses stay in FP32 even while AMP is enabled.
            with autocast('cuda', enabled=False):
                # Only the sampled segment participates in the mel loss.
                # Project the full spectrogram lazily when plotting images.
                y_mel = spec_to_mel_torch(
                    commons.slice_segments(
                        spec, ids_slice, hps.train.segment_size // hps.data.hop_length,
                    ),
                    hps.data.filter_length, hps.data.n_mel_channels,
                    hps.data.sampling_rate, hps.data.mel_fmin, hps.data.mel_fmax,
                )
                y_hat_mel = mel_spectrogram_torch(
                    y_hat.squeeze(1).float(), hps.data.filter_length,
                    hps.data.n_mel_channels, hps.data.sampling_rate,
                    hps.data.hop_length, hps.data.win_length,
                    hps.data.mel_fmin, hps.data.mel_fmax,
                    check_range=False,  # The decoder already bounds audio with tanh.
                )
                y_sliced = commons.slice_segments(
                    y, ids_slice * hps.data.hop_length, hps.train.segment_size,
                )

            d_sync_ctx = net_d.no_sync() if isinstance(net_d, DDP) and not should_step else nullcontext()
            with d_sync_ctx:
                with autocast('cuda', enabled=amp_enabled, dtype=half_type):
                    y_d_hat_r, y_d_hat_g, fmap_r, _ = net_d(y_sliced, y_hat.detach())
                discriminator = unwrap(net_d)
                reuse_real_features = not getattr(discriminator, "use_spectral_norm", True)
                if reuse_real_features:
                    # D has not stepped yet. These are exactly the real features
                    # needed by G; detach before D's graph is freed by backward.
                    fmap_r = [[feature.detach() for feature in layers] for layers in fmap_r]
                with autocast('cuda', enabled=False):
                    loss_disc, _, _ = discriminator_loss(y_d_hat_r, y_d_hat_g, return_details=False)
                scaler.scale(loss_disc / window_size).backward()

            # G gradients pass through D, but must never accumulate into D's parameters.
            discriminator.requires_grad_(False)
            try:
                with autocast('cuda', enabled=amp_enabled, dtype=half_type):
                    if reuse_real_features:
                        _, y_d_hat_g, _, fmap_g = discriminator(None, y_hat)
                    else:
                        # Spectral normalization updates its buffers on every
                        # forward, so preserve its original sequence of calls.
                        _, y_d_hat_g, fmap_r, fmap_g = discriminator(y_sliced, y_hat)
                with autocast('cuda', enabled=False):
                    loss_mel = F.l1_loss(y_mel.float(), y_hat_mel.float()) * hps.train.c_mel
                    loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * hps.train.c_kl
                    loss_fm = feature_loss(fmap_r, fmap_g)
                    loss_gen, _ = generator_loss(y_d_hat_g)
                    loss_lf0 = (
                        F.mse_loss(pred_lf0.float(), lf0.float())
                        if unwrap(net_g).use_automatic_f0_prediction else loss_mel.new_zeros(())
                    )
                    loss_gen_all = loss_gen + loss_fm + loss_mel + loss_kl + loss_lf0
                scaler.scale(loss_gen_all / window_size).backward()
            finally:
                discriminator.requires_grad_(True)

        # Optimizer Step
        if should_step:
            # Discriminator
            scaler.unscale_(optim_d)
            grad_norm_d = gradient_norm(net_d.parameters()) if should_log else 0.0
            scaler.step(optim_d)
            optim_d.zero_grad(set_to_none=True)

            # Generator
            scaler.unscale_(optim_g)
            grad_norm_g = gradient_norm(net_g.parameters()) if should_log else 0.0
            scaler.step(optim_g)
            scaler.update()
            optim_g.zero_grad(set_to_none=True)
        else:
            grad_norm_d = 0.0
            grad_norm_g = 0.0

        global_step += 1

        # Save only after both optimizers and the scaler have finished the window.
        # In-flight gradients are not checkpointed, so never stop halfway through it.
        if should_step and interrupt is not None and interrupt.should_stop():
            if rank == 0:
                logger.info("Interrupt requested; saving epoch %s, next batch %s, step %s",
                            epoch, batch_idx + 1, global_step)
                save_training_state(hps, nets, optims, scaler, epoch, batch_idx + 1)
                logger.info("Training state saved to %s; restart with the same model directory to resume",
                            hps.model_dir)
            if dist.is_initialized():
                dist.barrier()
            return True

        # Logging
        if should_log:
            lr = optim_g.param_groups[0]['lr']
            # Transfer loss scalars together instead of synchronizing for every .item().
            disc_value, gen_value, fm_value, mel_value, kl_value, lf0_value, total_value = (
                torch.stack((loss_disc, loss_gen, loss_fm, loss_mel, loss_kl, loss_lf0, loss_gen_all))
                .detach().float().cpu().tolist()
            )
            reference_loss = disc_value + gen_value + fm_value + mel_value + kl_value + lf0_value

            logger.info(
                f'Train Epoch: {epoch} [{100. * batch_idx / len(train_loader):.0f}%] '
                f'Step: {global_step}, LR: {lr:.6f}'
            )
            logger.info(
                f"Losses - D: {disc_value:.4f}, G: {gen_value:.4f}, FM: {fm_value:.4f}, "
                f"Mel: {mel_value:.4f}, KL: {kl_value:.4f}, Total: {reference_loss:.4f}"
            )

            if writer:
                scalar_dict = {
                    "loss/g/total": total_value,
                    "loss/d/total": disc_value,
                    "learning_rate": lr,
                    "grad_norm_d": grad_norm_d,
                    "grad_norm_g": grad_norm_g,
                    "loss/g/fm": fm_value,
                    "loss/g/mel": mel_value,
                    "loss/g/kl": kl_value,
                    "loss/g/lf0": lf0_value,
                }
                utils.summarize(
                    writer=writer,
                    global_step=global_step,
                    scalars=scalar_dict
                )

        # Plotting is substantially slower than scalar logging on the training thread.
        if rank == 0 and writer and image_interval and global_step % image_interval == 0:
            mel = spec_to_mel_torch(
                spec, hps.data.filter_length, hps.data.n_mel_channels,
                hps.data.sampling_rate, hps.data.mel_fmin, hps.data.mel_fmax,
            )
            image_dict = {
                "slice/mel_org": utils.plot_spectrogram_to_numpy(y_mel[0].detach().cpu().numpy()),
                "slice/mel_gen": utils.plot_spectrogram_to_numpy(y_hat_mel[0].detach().cpu().numpy()),
                "all/mel": utils.plot_spectrogram_to_numpy(mel[0].detach().cpu().numpy())
            }
            if unwrap(net_g).use_automatic_f0_prediction:
                image_dict.update({
                    "all/lf0": utils.plot_data_to_numpy(
                        lf0[0, 0, :].cpu().numpy(),
                        pred_lf0[0, 0, :].detach().float().cpu().numpy()
                    ),
                    "all/norm_lf0": utils.plot_data_to_numpy(
                        lf0[0, 0, :].cpu().numpy(),
                        norm_lf0[0, 0, :].detach().float().cpu().numpy()
                    )
                })
            utils.summarize(writer=writer, global_step=global_step, images=image_dict)

        # Evaluation & Checkpoint
        if should_step and global_step >= next_eval_step:
            if rank == 0:
                evaluate(hps, net_g, eval_loader, writer_eval, global_step)
                save_training_state(hps, nets, optims, scaler, epoch, batch_idx + 1)
            # Rank 0 evaluates the unwrapped generator; keep other ranks in sync.
            if dist.is_initialized():
                dist.barrier()
            next_eval_step = (global_step // hps.train.eval_interval + 1) * hps.train.eval_interval

    return False


@torch.inference_mode()
def evaluate(hps, generator, eval_loader, writer_eval, step):
    """评估函数"""
    if eval_loader is None or len(eval_loader) == 0 or writer_eval is None:
        return
    was_training = generator.training
    generator.eval()
    try:
        # infer() seeds torch RNGs internally; validation must not reset training RNGs.
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            evaluate_samples(hps, generator, eval_loader, writer_eval, step)
    finally:
        generator.train(was_training)


def evaluate_samples(hps, generator, eval_loader, writer_eval, step):
    image_dict = {}
    audio_dict = {}
    amp_enabled, half_type = amp_settings(hps.train)

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

        with autocast('cuda', enabled=amp_enabled, dtype=half_type):
            y_hat, _ = unwrap(generator).infer(c, f0, uv, g=g, vol=volume)

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

        # tensorboardX accepts mono [samples] or channel-last [samples, channels].
        audio_dict[f"gen/audio_{batch_idx}"] = y_hat[0, 0].float().cpu()
        audio_dict[f"gt/audio_{batch_idx}"] = y[0, 0].cpu()

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


if __name__ == "__main__":
    main()
