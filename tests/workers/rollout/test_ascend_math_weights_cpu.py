import os
from pathlib import Path

import pytest
import torch
from torch.distributed import destroy_process_group, init_process_group
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import Shard, distribute_tensor

from verl.workers.ascend_vllm import MathWeightLoader, engine_environment, sync_weights


class Model(torch.nn.Linear):
    @torch.no_grad()
    def load_weights(self, weights):
        for name, value in weights:
            self.get_parameter(name).copy_(value)


class Engine(MathWeightLoader):
    def __init__(self):
        self.model = Model(4, 4)
        self.calls = []

    def get_model(self):
        return self.model

    def collective_rpc(self, method, args):
        self.calls.append(Path(args[0]))
        getattr(self, method)(*args)

    def reset_prefix_cache(self):
        self.calls.append("reset")
        return True


def check_updates(rank, rendezvous):
    init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2)
    try:
        mesh = init_device_mesh("cpu", (2,))
        solver, verifier = Engine(), Engine()
        prompt = torch.arange(4.0) / 10
        for offset, (active, other) in enumerate([(solver, verifier), (verifier, solver)], start=1):
            before = other.model(prompt).log_softmax(0).detach().clone()
            target = Model(4, 4)
            with torch.no_grad():
                target.weight.copy_(torch.arange(16.0).reshape(4, 4) * offset)
                target.bias.copy_(torch.arange(4.0))
            state = {k: distribute_tensor(v, mesh, [Shard(0)]) for k, v in target.state_dict().items()}
            sync_weights(active, state, bucket_bytes=16)
            assert torch.equal(active.model(prompt), target(prompt))
            assert torch.equal(active.model(prompt).log_softmax(0), target(prompt).log_softmax(0))
            assert torch.equal(other.model(prompt).log_softmax(0), before)
            assert active.calls[-1] == "reset"
            assert len(active.calls) == 3
            assert all(not path.exists() for path in active.calls[:-1])
    finally:
        destroy_process_group()


def test_independent_sharded_updates(tmp_path):
    torch.multiprocessing.spawn(check_updates, args=((tmp_path / "store").as_uri(),), nprocs=2)


def test_engine_rank_environment(monkeypatch):
    monkeypatch.setenv("ASCEND_RT_VISIBLE_DEVICES", "3,5")
    monkeypatch.setenv("WORLD_SIZE", "4")
    with pytest.raises(RuntimeError), engine_environment(1):
        assert os.environ["ASCEND_RT_VISIBLE_DEVICES"] == "5"
        assert os.environ["WORLD_SIZE"] == "1"
        raise RuntimeError("initialization failed")
    assert os.environ["WORLD_SIZE"] == "4"
    assert os.environ["ASCEND_RT_VISIBLE_DEVICES"] == "3,5"
