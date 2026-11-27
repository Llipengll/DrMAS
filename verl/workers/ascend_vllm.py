"""Single-node, TP=1 vLLM-Ascend rollout for independently trained agents."""
import os
from contextlib import contextmanager
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from packaging.version import Version
from safetensors import safe_open
from safetensors.torch import save_file
from torch.distributed.tensor import DTensor


@contextmanager
def engine_environment(device_index):
    # The spawned engine sees only its parent's assigned physical NPU.
    visible = os.environ.get("ASCEND_RT_VISIBLE_DEVICES")
    physical = visible.split(",")[device_index].strip() if visible else str(device_index)
    updates = {
        "ASCEND_RT_VISIBLE_DEVICES": physical,
        "VLLM_ENABLE_V1_MULTIPROCESSING": "1",
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
        "RANK": "0", "LOCAL_RANK": "0", "WORLD_SIZE": "1", "LOCAL_WORLD_SIZE": "1",
    }
    previous = {key: os.environ.get(key) for key in updates}
    os.environ.update(updates)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def build_engine(llm_class, device_index, **kwargs):
    if Version(version("vllm")).release != (0, 23, 0):
        raise ValueError("The Ascend math rollout requires vLLM 0.23.0")
    if kwargs["tensor_parallel_size"] != 1 or kwargs.get("enable_lora"):
        raise ValueError("The Ascend math rollout supports TP=1 and full model weights")
    kwargs.pop("disable_mm_preprocessor_cache", None)
    if kwargs.get("limit_mm_per_prompt") is None:
        kwargs.pop("limit_mm_per_prompt", None)
    # Ascend rejects weight wake-up in FRACTAL_NZ mode during RL updates.
    kwargs["additional_config"] = {**(kwargs.get("additional_config") or {}), "weight_nz_mode": 0}
    kwargs.update(distributed_executor_backend="uni", mm_processor_cache_gb=0,
                  worker_extension_cls="verl.workers.ascend_vllm.MathWeightLoader")
    with engine_environment(device_index):
        return llm_class(**kwargs)


class MathWeightLoader:
    def load_drmas_weight_bucket(self, path):
        with safe_open(path, framework="pt", device="cpu") as weights:
            self.get_model().load_weights((name, weights.get_tensor(name)) for name in weights.keys())


def sync_weights(engine, state, bucket_bytes=64 * 1024**2):
    # Local engine and Ray actor share a filesystem, not an NPU allocator.
    with TemporaryDirectory(prefix="drmas-weights-") as directory:
        path = str(Path(directory) / "bucket.safetensors")
        bucket = {}
        size = 0

        def flush():
            save_file(bucket, path)
            engine.collective_rpc("load_drmas_weight_bucket", args=(path,))
            bucket.clear()

        for name in sorted(state):
            param = state[name].detach()
            nbytes = param.numel() * param.element_size()
            if bucket and size + nbytes > bucket_bytes:
                flush()
                size = 0
            if isinstance(param, DTensor):
                param = param.full_tensor()
            # Private storage also handles tied parameters and strided views.
            bucket[name] = param.to("cpu", copy=True, memory_format=torch.contiguous_format)
            size += nbytes
        if bucket:
            flush()
    if not engine.reset_prefix_cache():
        raise RuntimeError("vLLM could not invalidate the previous policy's prefix cache")
