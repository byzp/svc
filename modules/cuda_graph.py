"""Bounded native CUDA graphs for repeatedly used training submodules."""

from collections import OrderedDict
import gc
import threading
import weakref

import torch


# Global stream capture also rejects CUDA host allocations in other threads.
# Share this lock with the loader's pinned allocator, not its CPU data work.
cuda_graph_lock = threading.Lock()


class CudaGraphModule(torch.nn.Module):
    # Variable sequence lengths must not retain an unbounded number of pools.
    _max_cuda_graphs = 2

    def set_graph_gradient_accumulation(self, enabled):
        enabled = bool(enabled)
        if enabled != getattr(self, "_graph_accumulation", True):
            self.clear_cuda_graphs()
            self._graph_accumulation = enabled

    def clear_cuda_graphs(self):
        for handle in getattr(self, "_graph_grad_handles", ()):
            handle.remove()
        self._graph_grad_handles = []
        cache = getattr(self, "_graph_cache", None)
        self._graph_cache = OrderedDict()
        if cache:
            with cuda_graph_lock:
                cache.clear()
                # make_graphed_callables creates a Function <-> Backward class
                # cycle that owns static CUDA tensors even after eviction.
                # Free-threaded Python may defer collection for many batches.
                gc.collect()

    def _apply(self, fn, recurse=True):
        # Captures contain device addresses, so moving/casting a model invalidates
        # them. Neither captures nor hooks are part of the checkpoint format.
        self.clear_cuda_graphs()
        return super()._apply(fn, recurse=recurse)

    def _own_first_parameter_grad(self, parameters):
        # train_one_epoch clears grads after every optimizer step. When there
        # is just one microbatch, no later replay can overwrite an accumulated
        # gradient, so those buffers can be consumed directly by AdamW.
        if not getattr(self, "_graph_accumulation", True):
            return
        handles = []
        for parameter in parameters:
            reference = weakref.ref(parameter)

            def own_first_grad(grad, reference=reference):
                parameter = reference()
                # make_graphed_callables returns aliases of replay buffers. The
                # first accumulated grad must own its storage before replay can
                # overwrite it, including when GradScaler unscales it in place.
                return grad.clone() if parameter.grad is None else grad

            handles.append(parameter.register_hook(own_first_grad))
        self._graph_grad_handles = handles

    def _call_graphed(self, implementation, args):
        parameters = tuple(self.parameters())
        trainable = tuple(parameter for parameter in parameters if parameter.requires_grad)
        amp_enabled = torch.is_autocast_enabled("cuda")
        amp_dtype = torch.get_autocast_dtype("cuda")
        key = (
            tuple((tuple(x.shape), tuple(x.stride()), x.dtype, x.device, x.requires_grad) for x in args),
            tuple(parameter.requires_grad for parameter in parameters),
            amp_enabled, amp_dtype,
        )
        cache = getattr(self, "_graph_cache", None)
        if cache is None:
            cache = self._graph_cache = OrderedDict()
        if key not in cache:
            # Evict before capture: otherwise three pools coexist at the
            # capacity boundary. Collection must be outside stream capture;
            # destroying an unrelated graph during capture is illegal.
            with cuda_graph_lock:
                if len(cache) >= self._max_cuda_graphs:
                    while len(cache) >= self._max_cuda_graphs:
                        cache.popitem(last=False)
                    gc.collect()
            # Hooks for the outer accumulation must not be recorded inside a
            # trainable capture too. Frozen captures never visit these hooks.
            if trainable:
                for handle in getattr(self, "_graph_grad_handles", ()):
                    handle.remove()
                self._graph_grad_handles = []
            had_forward = "forward" in self.__dict__
            original_forward = self.__dict__.get("forward")
            sample_args = tuple(x.detach().clone().requires_grad_(x.requires_grad) for x in args)
            # Share parameter storage through fresh leaves, so capture never
            # visits DDP's retained AccumulateGrad nodes on the default stream.
            # Replays receive the original parameters as autograd inputs; their
            # addresses match these aliases, so no weight copies are needed.
            names = tuple(name for name, _ in self.named_parameters())
            sample_parameters = tuple(p.detach().requires_grad_(p.requires_grad) for p in parameters)

            def functional(*inputs):
                state = dict(zip(names, inputs[len(args):]))
                return torch.func.functional_call(self, state, inputs[:len(args)])

            try:
                self.forward = implementation
                # Warmup/capture must not consume the training RNG sequence.
                # AMP's weight cache must stay off while recording a graph.
                with (
                    cuda_graph_lock,
                    torch.random.fork_rng(devices=[args[0].device]),
                    torch.amp.autocast("cuda", enabled=amp_enabled, dtype=amp_dtype, cache_enabled=False),
                ):
                    cache[key] = torch.cuda.make_graphed_callables(
                        functional, sample_args + sample_parameters,
                        allow_unused_input=getattr(self, "_allow_unused_graph_input", False),
                    )
            finally:
                if had_forward:
                    self.forward = original_forward
                else:
                    del self.forward
                if trainable:
                    self._own_first_parameter_grad(trainable)
        cache.move_to_end(key)
        return cache[key](*args, *parameters)
