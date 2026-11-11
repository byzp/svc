"""Bounded, ordered CPU batch loading without multiprocessing transport."""

from collections import deque
from concurrent.futures import ThreadPoolExecutor, wait
import random
import threading

import torch
from torch.utils.data import BatchSampler, SequentialSampler

from modules.cuda_graph import cuda_graph_lock


class ThreadDataLoader:
    """Load datasets exposing get_item(index, rng) in persistent worker threads.

    Each task owns its augmentation RNG, so completion order cannot change a
    batch. Only CPU tensors are produced; pinning uses the training device.
    Close the loader when training exits to cancel outstanding prefetches.
    """

    def __init__(self, dataset, *, batch_size, collate_fn, sampler=None,
                 num_workers=4, prefetch_factor=4, pin_memory=False,
                 drop_last=False, seed=0):
        for name, value, minimum in (("num_workers", num_workers, 0),
                                     ("prefetch_factor", prefetch_factor, 1)):
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        self.dataset = dataset
        self.collate_fn = collate_fn
        self.sampler = sampler if sampler is not None else SequentialSampler(dataset)
        self.batch_sampler = BatchSampler(self.sampler, batch_size, drop_last)
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.pin_memory = pin_memory
        self.seed = seed
        self._iteration = 0
        self._iterator = None
        self._closed = False
        self._executor = None
        if num_workers:
            device = torch.cuda.current_device() if pin_memory else None
            self._executor = ThreadPoolExecutor(
                max_workers=num_workers, thread_name_prefix="audio-loader",
                initializer=self._init_worker, initargs=(device,),
            )

    @staticmethod
    def _init_worker(device):
        torch.set_grad_enabled(False)
        if device is not None:
            with cuda_graph_lock:
                torch.cuda.set_device(device)

    def _load_batch(self, indices, seed, stopped):
        rng = random.Random(seed)
        samples = []
        for index in indices:
            if stopped.is_set():
                return None
            samples.append(self.dataset.get_item(index, rng))
        if stopped.is_set():
            return None
        batch = self.collate_fn(samples)
        if self.pin_memory and not stopped.is_set():
            with cuda_graph_lock:
                batch = tuple(item.pin_memory() if item is not None else None for item in batch)
        return batch

    def __len__(self):
        return len(self.batch_sampler)

    def __iter__(self):
        if self._closed:
            raise RuntimeError("ThreadDataLoader is closed")
        if self._iterator is not None:
            self._iterator.close()
            # A restarted epoch must not read a cache file an old task is writing.
            wait(self._iterator.running)
        epoch = getattr(self.sampler, "epoch", self._iteration)
        seed = self.seed + epoch * (len(self) + 1)
        self._iteration += 1
        self._iterator = _BatchIterator(self, seed)
        return self._iterator

    def close(self):
        self._closed = True
        if self._iterator is not None:
            self._iterator.close()
            self._iterator = None
        if self._executor is not None:
            self._executor.shutdown(wait=True, cancel_futures=True)
            self._executor = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class _BatchIterator:
    def __init__(self, loader, seed):
        self.loader = loader
        self.seed = seed
        self.batches = enumerate(loader.batch_sampler)
        self.stopped = threading.Event()
        self.pending = deque()
        self.running = []
        if loader.num_workers:
            try:
                for _ in range(loader.num_workers * loader.prefetch_factor):
                    if not self._submit():
                        break
            except BaseException:
                self.close()
                raise

    def _submit(self):
        task = next(self.batches, None)
        if task is None:
            return False
        index, indices = task
        self.pending.append(self.loader._executor.submit(
            self.loader._load_batch, indices, self.seed + index, self.stopped,
        ))
        return True

    def __iter__(self):
        return self

    def __next__(self):
        if self.stopped.is_set():
            raise StopIteration
        try:
            if self.loader.num_workers:
                if not self.pending:
                    self.close()
                    raise StopIteration
                batch = self.pending.popleft().result()
                self._submit()
                return batch
            index, indices = next(self.batches)
            return self.loader._load_batch(indices, self.seed + index, self.stopped)
        except BaseException:
            self.close()
            raise

    def close(self):
        self.stopped.set()
        while self.pending:
            future = self.pending.popleft()
            if not future.cancel() and not future.done():
                self.running.append(future)
