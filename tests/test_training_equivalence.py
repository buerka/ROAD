"""Fast semantic regression tests for training-loop memory optimisations.

The legacy implementation is loaded from the commit immediately preceding the
training-loop optimisation.  Tiny recording models make it possible to compare
the complete observable training trace without touching the ROAD HDF5 file or
creating a normalised cache.
"""

import ast
import hashlib
import subprocess
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch
from torch import nn


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
LEGACY_COMMIT = "224678017bec1506cfa058ab64e276071612d4f4"


def _combine(value, dim_begin, dim_end):
    shape = list(value.shape[:dim_begin]) + [-1] + list(value.shape[dim_end:])
    return value.view(shape)


def _tensor_digest(value):
    value = value.detach().cpu().to(torch.float32).contiguous()
    digest = hashlib.sha256()
    digest.update(str(tuple(value.shape)).encode("ascii"))
    digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _canonical(value):
    """Take an immutable, exact snapshot of a plotting/progress argument."""

    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().contiguous()
        return (
            "torch",
            str(tensor.dtype),
            tuple(tensor.shape),
            tensor.to(torch.float32).numpy().tobytes(),
        )
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        return ("numpy", str(array.dtype), array.shape, array.tobytes())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return tuple((key, _canonical(item)) for key, item in value.items())
    if isinstance(value, (list, tuple)):
        return tuple(_canonical(item) for item in value)
    return value


class _Trace:
    def __init__(self):
        self.forward = []
        self.mode = []
        self.optimiser = []
        self.rng_draws = []
        self.context = []
        self.checkpoints = []
        self.loads = []
        self.loss_curves = []
        self.postfix = []
        self.patch_calls = 0
        self.data_accesses = 0


class _RecordingModule(nn.Module):
    def __init__(self, role, trace):
        super().__init__()
        self.role = role
        self.trace = trace
        self._saved_state = None

    def to(self, *args, **kwargs):
        # Keep parameters in float32 so this CPU-only test is independent of
        # which bfloat16 optimiser kernels happen to be installed.
        return self

    def train(self, mode=True):
        self.trace.mode.append((self.role, bool(mode)))
        return super().train(mode)

    def _record_forward(self, value):
        self.trace.forward.append(
            (
                self.role,
                bool(self.training),
                bool(torch.is_grad_enabled()),
                _tensor_digest(value),
            )
        )

    def save(self, *args):
        state = {
            key: value.detach().clone()
            for key, value in self.state_dict().items()
        }
        self._saved_state = state
        self.trace.checkpoints.append(
            (self.role, args[1:], _state_digest(state))
        )

    def load(self, *args):
        self.trace.loads.append((self.role, args[1:]))
        if self._saved_state is not None:
            self.load_state_dict(self._saved_state)


class _Backbone(_RecordingModule):
    def __init__(self, trace):
        super().__init__("backbone", trace)
        self.scale = nn.Parameter(torch.tensor(0.35))

    def forward(self, value):
        self._record_forward(value)
        base = value.float().reshape(len(value), -1).mean(dim=1, keepdim=True)
        offsets = torch.tensor([0.05, 0.15], dtype=torch.float32)
        return base * self.scale + offsets


class _PositionClassifier(_RecordingModule):
    def __init__(self, trace):
        super().__init__("position", trace)
        self.weight = nn.Parameter(torch.linspace(-0.2, 0.5, 8))

    def forward(self, data, neighbour):
        self._record_forward(torch.cat([data, neighbour], dim=1))
        base = (data + neighbour).mean(dim=1, keepdim=True)
        return base * self.weight.unsqueeze(0)

    @staticmethod
    def loss_function(prediction, target):
        return nn.functional.cross_entropy(prediction, target)


class _Decoder(_RecordingModule):
    def __init__(self, trace):
        super().__init__("decoder", trace)
        self.scale = nn.Parameter(torch.tensor(0.6))

    def forward(self, value):
        self._record_forward(value)
        return value.mean(dim=1).reshape(-1, 1, 1, 1) * self.scale

    @staticmethod
    def loss_function(target, prediction):
        return torch.square(target.float() - prediction).mean()


def _state_digest(state):
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        digest.update(name.encode("utf-8"))
        digest.update(value.detach().cpu().to(torch.float32).numpy().tobytes())
    return digest.hexdigest()


class _RecordingAdam:
    def __init__(self, parameters, learning_rate, trace, role):
        self.trace = trace
        self.role = role
        self.inner = torch.optim.Adam(list(parameters), lr=learning_rate)

    def zero_grad(self):
        self.trace.optimiser.append((self.role, "zero_grad"))
        self.inner.zero_grad()

    def step(self):
        self.trace.optimiser.append((self.role, "step"))
        self.inner.step()


class _TorchFacade:
    """Delegate to torch while injecting recording Adam instances."""

    def __init__(self, trace):
        roles = iter(("backbone", "position", "decoder"))

        def adam(parameters, lr):
            return _RecordingAdam(parameters, lr, trace, next(roles))

        self.optim = SimpleNamespace(Adam=adam)

    def __getattr__(self, name):
        return getattr(torch, name)


class _TinyLoader:
    """Two batches whose construction consumes both global RNG streams."""

    def __init__(self, trace):
        self.trace = trace

    def __len__(self):
        return 2

    def __iter__(self):
        for batch_index in range(2):
            numpy_draw = int(np.random.randint(0, 8))
            torch_draw = float(torch.rand(()).item())
            self.trace.rng_draws.append(
                ("train", batch_index, numpy_draw, torch_draw)
            )
            value = np.float32(0.25 + numpy_draw / 100.0 + torch_draw / 10.0)
            data = torch.full((2, 1, 1, 1, 1), float(value))
            neighbour = torch.full(
                (2, 1, 1, 1, 1), float(value + np.float32(0.1))
            )
            target = torch.zeros((2, 1), dtype=torch.long)
            context_label = torch.full(
                (2, 1), numpy_draw, dtype=torch.long
            )
            yield data, target, context_label, neighbour


class _TinyValidationDataset:
    def __init__(self, trace):
        self.trace = trace
        self._data = torch.tensor(
            [
                [[[0.4]]],
                [[[0.8]]],
            ],
            dtype=torch.float64,
        )

    @property
    def data(self):
        self.trace.data_accesses += 1
        return self._data

    def patch(self, value):
        self.trace.patch_calls += 1
        return value.clone()

    def context_prediction(self, patches):
        draws = np.random.randint(0, 8, size=len(patches)).astype(np.int64)
        labels = torch.from_numpy(draws.copy())
        # Guarantee at least one correct class so checkpoint behaviour is
        # exercised, while retaining all RNG calls and random variation.
        labels[0] = 7
        neighbour = patches.float() + labels.reshape(-1, 1, 1, 1) / 100.0
        self.trace.context.append(
            (tuple(int(item) for item in labels), _tensor_digest(neighbour))
        )
        return labels, neighbour


class _Progress:
    def __init__(self, iterable, trace, **kwargs):
        self.iterable = iterable
        self.trace = trace

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def __iter__(self):
        return iter(self.iterable)

    def set_description(self, value):
        pass

    def set_postfix(self, **kwargs):
        self.trace.postfix.append(_canonical(kwargs))


def _extract_train_ssl(source, filename, namespace):
    module = ast.parse(source, filename=filename)
    function = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == "train_ssl"
    )
    extracted = ast.Module(body=[function], type_ignores=[])
    ast.fix_missing_locations(extracted)
    exec(compile(extracted, filename, "exec"), namespace)
    return namespace["train_ssl"]


def _legacy_train_ssl(namespace):
    result = subprocess.run(
        ["git", "show", f"{LEGACY_COMMIT}:train.py"],
        cwd=REPOSITORY_ROOT,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
    )
    return _extract_train_ssl(result.stdout, "legacy/train.py", namespace)


def _current_train_ssl(namespace):
    source = (REPOSITORY_ROOT / "train.py").read_text(encoding="utf-8")
    return _extract_train_ssl(source, "train.py", namespace)


def _numpy_rng_snapshot():
    algorithm, keys, position, has_gauss, cached = np.random.get_state()
    return algorithm, keys.tobytes(), position, has_gauss, cached


def _run(legacy):
    np.random.seed(1729)
    torch.manual_seed(2718)

    trace = _Trace()
    torch_facade = _TorchFacade(trace)
    fake_os = SimpleNamespace(
        path=SimpleNamespace(exists=lambda path: True),
        makedirs=lambda path: None,
    )

    def progress(iterable, **kwargs):
        return _Progress(iterable, trace, **kwargs)

    def loss_curve(*args, **kwargs):
        trace.loss_curves.append(_canonical((args, kwargs)))

    namespace = {
        "torch": torch_facade,
        "os": fake_os,
        "tqdm": progress,
        "loss_curve": loss_curve,
        "combine": _combine,
        # Names used only while evaluating the legacy function annotations.
        "DataLoader": object,
        "Dataset": object,
        "BackBone": object,
        "PositionClassifier": object,
        "Decoder": object,
        "args": object,
        "open": mock.mock_open(),
    }
    function = (
        _legacy_train_ssl(namespace)
        if legacy
        else _current_train_ssl(namespace)
    )

    loader = _TinyLoader(trace)
    validation = _TinyValidationDataset(trace)
    backbone = _Backbone(trace)
    position = _PositionClassifier(trace)
    decoder = _Decoder(trace)
    arguments = SimpleNamespace(
        device=torch.device("cpu"),
        learning_rate=0.01,
        epochs=2,
        model_path="unused",
        model_name="tiny",
        patch_size=256,
    )

    function(
        loader,
        validation,
        backbone,
        position,
        decoder,
        arguments,
    )

    return {
        "trace": trace,
        "numpy_state": _numpy_rng_snapshot(),
        "torch_state": torch.get_rng_state().numpy().tobytes(),
        "parameters": (
            _state_digest(backbone.state_dict()),
            _state_digest(position.state_dict()),
            _state_digest(decoder.state_dict()),
        ),
    }


class TrainingEquivalenceTests(unittest.TestCase):
    def test_train_ssl_low_memory_validation_is_semantically_equivalent(self):
        legacy = _run(legacy=True)
        current = _run(legacy=False)
        old_trace = legacy["trace"]
        new_trace = current["trace"]

        self.assertEqual(legacy["numpy_state"], current["numpy_state"])
        self.assertEqual(legacy["torch_state"], current["torch_state"])
        self.assertEqual(old_trace.rng_draws, new_trace.rng_draws)
        self.assertEqual(old_trace.context, new_trace.context)

        # Forward order, mode, and values must stay exact.  Validation is the
        # sole intended grad-mode difference.
        old_without_grad = [event[:2] + event[3:] for event in old_trace.forward]
        new_without_grad = [event[:2] + event[3:] for event in new_trace.forward]
        self.assertEqual(old_without_grad, new_without_grad)
        old_validation = [event for event in old_trace.forward if not event[1]]
        new_validation = [event for event in new_trace.forward if not event[1]]
        self.assertTrue(old_validation)
        self.assertTrue(all(event[2] for event in old_validation))
        self.assertTrue(all(not event[2] for event in new_validation))
        self.assertTrue(
            all(event[2] for event in new_trace.forward if event[1])
        )

        self.assertEqual(old_trace.mode, new_trace.mode)
        self.assertEqual(old_trace.optimiser, new_trace.optimiser)
        self.assertEqual(legacy["parameters"], current["parameters"])
        self.assertEqual(old_trace.postfix, new_trace.postfix)
        self.assertEqual(old_trace.loss_curves, new_trace.loss_curves)
        self.assertEqual(old_trace.checkpoints, new_trace.checkpoints)
        self.assertEqual(old_trace.loads, new_trace.loads)
        self.assertTrue(new_trace.checkpoints)

        expected_validation_calls = 2 * 2  # epochs * batches
        self.assertEqual(old_trace.patch_calls, expected_validation_calls)
        self.assertEqual(old_trace.data_accesses, expected_validation_calls)
        self.assertEqual(new_trace.patch_calls, 1)
        self.assertEqual(new_trace.data_accesses, 1)
        self.assertEqual(len(old_trace.context), expected_validation_calls)
        self.assertEqual(len(new_trace.context), expected_validation_calls)


if __name__ == "__main__":
    unittest.main()
