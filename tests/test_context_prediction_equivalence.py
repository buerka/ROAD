"""Bitwise oracle tests for ``LOFARDataset.context_prediction``.

The tests extract the method from the pre-optimisation commit and from the
working tree without importing ``data.py``.  This avoids argument-parsing and
model-package side effects and keeps the test independent of the ROAD HDF5
file.
"""

import ast
import hashlib
import os
import subprocess
import unittest
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader, Dataset, get_worker_info


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
LEGACY_COMMIT = "224678017bec1506cfa058ab64e276071612d4f4"
_CLASS_CACHE = {}


def _extract_context_class(source, filename):
    tree = ast.parse(source, filename=filename)
    dataset_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "LOFARDataset"
    )
    method_names = {
        "_get_context_executor",
        "_shutdown_context_executor",
        "__getstate__",
        "__del__",
        "context_prediction",
    }
    dataset_class.body = [
        node
        for node in dataset_class.body
        if isinstance(node, ast.FunctionDef) and node.name in method_names
    ]
    dataset_class.name = "ExtractedContextDataset"
    dataset_class.bases = []
    dataset_class.keywords = []
    helpers = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_fixed_resized_crop"
    ]
    module = ast.Module(body=helpers + [dataset_class], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {
        "__name__": "_context_prediction_oracle",
        "np": np,
        "torch": torch,
        "T": T,
        "TF": TF,
        "os": os,
        "deque": deque,
        "ThreadPoolExecutor": ThreadPoolExecutor,
    }
    exec(compile(module, filename, "exec"), namespace)
    return namespace["ExtractedContextDataset"]


def _context_class(source):
    key = hashlib.sha256(source.encode("utf-8")).hexdigest()
    if key not in _CLASS_CACHE:
        _CLASS_CACHE[key] = _extract_context_class(
            source, f"context_prediction-{key[:12]}.py"
        )
    return _CLASS_CACHE[key]


def _context_host(source,
                  n_patches=4,
                  patch_size=3,
                  context_workers=1,
                  random_resize=False):
    dataset_class = _context_class(source)
    host = dataset_class.__new__(dataset_class)
    host.n_patches = n_patches
    host.args = SimpleNamespace(patch_size=patch_size)
    host.context_workers = context_workers
    host._context_executor = None
    host._context_executor_pid = None
    scale = (0.55, 1.0) if random_resize else (1.0, 1.0)
    ratio = (0.75, 4.0 / 3.0) if random_resize else (1.0, 1.0)
    host.resizer = T.RandomResizedCrop(
        scale=scale,
        ratio=ratio,
        size=(patch_size, patch_size),
        antialias=False,
    )
    return host


def _close_host(host):
    shutdown = getattr(host, "_shutdown_context_executor", None)
    if shutdown is not None:
        shutdown(wait=True, cancel_futures=True)


def _grid_data(item=0, n_patches=4, patch_size=3, images=2):
    patches = []
    for image in range(images):
        for patch in range(n_patches**2):
            base = item * 1000 + image * 100 + patch
            values = torch.arange(
                2 * patch_size * patch_size, dtype=torch.float64
            ).reshape(2, patch_size, patch_size)
            patches.append(values / 100.0 + base)
    return torch.stack(patches)


def _tensor_snapshot(value):
    array = value.detach().cpu().contiguous().numpy()
    return (
        str(value.dtype),
        tuple(value.shape),
        tuple(value.stride()),
        array.tobytes(),
    )


def _result_snapshot(result):
    labels, neighbours = result
    return _tensor_snapshot(labels), _tensor_snapshot(neighbours)


def _numpy_rng_snapshot():
    algorithm, keys, position, has_gauss, cached = np.random.get_state()
    return algorithm, keys.tobytes(), position, has_gauss, cached


def _rng_snapshot():
    return _numpy_rng_snapshot(), torch.get_rng_state().numpy().tobytes()


def _run_stream(source,
                context_workers=1,
                numpy_seed=4103,
                torch_seed=9209,
                calls=4):
    np.random.seed(numpy_seed)
    torch.manual_seed(torch_seed)
    host = _context_host(
        source, context_workers=context_workers, random_resize=True
    )
    outputs = []
    try:
        for item in range(calls):
            outputs.append(
                _result_snapshot(
                    host.context_prediction(_grid_data(item=item))
                )
            )
        return tuple(outputs), _rng_snapshot()
    finally:
        _close_host(host)


def _worker_init(worker_id):
    np.random.seed(30011 + worker_id)
    torch.manual_seed(50021 + worker_id)


def _identity(value):
    return value


class _WorkerContextDataset(Dataset):
    def __init__(self, source, context_workers, length=8):
        self.source = source
        self.context_workers = context_workers
        self.length = length
        self.host = None

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        if self.host is None:
            self.host = _context_host(
                self.source,
                context_workers=self.context_workers,
                random_resize=True,
            )
        result = self.host.context_prediction(_grid_data(item=index))
        worker = get_worker_info()
        return (
            int(index),
            -1 if worker is None else int(worker.id),
            _result_snapshot(result),
            _rng_snapshot(),
        )

    def __del__(self):
        if self.host is not None:
            _close_host(self.host)


def _run_workers(source, workers, context_workers):
    generator = torch.Generator().manual_seed(71237)
    loader = DataLoader(
        _WorkerContextDataset(source, context_workers),
        batch_size=None,
        shuffle=False,
        num_workers=workers,
        collate_fn=_identity,
        worker_init_fn=_worker_init,
        generator=generator,
        prefetch_factor=1,
        persistent_workers=False,
    )
    return tuple(loader)


def _run_n_equals_one(source, seed):
    np.random.seed(seed)
    torch.manual_seed(8081)
    host = _context_host(source, n_patches=1, patch_size=2)
    data = _grid_data(n_patches=1, patch_size=2, images=1)
    try:
        result = ("result", _result_snapshot(host.context_prediction(data)))
    except Exception as error:  # The legacy edge case intentionally may fail.
        result = ("error", type(error).__name__)
    finally:
        rng = _rng_snapshot()
        _close_host(host)
    return result, rng


class ContextPredictionEquivalenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        legacy = subprocess.run(
            ["git", "show", f"{LEGACY_COMMIT}:data.py"],
            cwd=REPOSITORY_ROOT,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
        )
        cls.legacy_source = legacy.stdout
        cls.current_source = (REPOSITORY_ROOT / "data.py").read_text(
            encoding="utf-8"
        )

    def test_all_corner_edge_and_interior_choices_match(self):
        data = _grid_data(n_patches=4, patch_size=3, images=1)

        np.random.seed(1181)
        torch.manual_seed(2203)
        old_host = _context_host(self.legacy_source, random_resize=False)
        old_result = old_host.context_prediction(data)
        old_rng = _rng_snapshot()
        _close_host(old_host)

        np.random.seed(1181)
        torch.manual_seed(2203)
        new_host = _context_host(self.current_source, random_resize=False)
        new_result = new_host.context_prediction(data)
        new_rng = _rng_snapshot()
        _close_host(new_host)

        self.assertEqual(_result_snapshot(old_result), _result_snapshot(new_result))
        self.assertEqual(old_rng, new_rng)

        labels, neighbours = new_result
        offsets = (-5, -4, -3, -1, 1, 3, 4, 5)
        allowed = {
            0: (4, 6, 7),
            1: (3, 4, 5, 6, 7),
            2: (3, 4, 5, 6, 7),
            3: (3, 5, 6),
            4: (1, 2, 4, 6, 7),
            5: tuple(range(8)),
            6: tuple(range(8)),
            7: (0, 1, 3, 5, 6),
            8: (1, 2, 4, 6, 7),
            9: tuple(range(8)),
            10: tuple(range(8)),
            11: (0, 1, 3, 5, 6),
            12: (1, 2, 4),
            13: (0, 1, 2, 3, 4),
            14: (0, 1, 2, 3, 4),
            15: (0, 1, 3),
        }
        for patch_index, label in enumerate(labels.tolist()):
            self.assertIn(label, allowed[patch_index])
            neighbour_index = patch_index + offsets[label]
            self.assertTrue(
                torch.equal(neighbours[patch_index], data[neighbour_index].float())
            )

    def test_repeated_calls_preserve_both_rng_streams_and_outputs(self):
        serial_legacy = _run_stream(self.legacy_source, context_workers=1)
        serial_current = _run_stream(self.current_source, context_workers=1)
        parallel_legacy = _run_stream(self.legacy_source, context_workers=4)
        parallel_current = _run_stream(self.current_source, context_workers=4)
        self.assertEqual(serial_legacy, serial_current)
        self.assertEqual(parallel_legacy, parallel_current)
        self.assertEqual(serial_current, parallel_current)

    def test_one_and_multiple_worker_streams_match(self):
        for workers in (1, 2):
            with self.subTest(workers=workers):
                legacy = _run_workers(
                    self.legacy_source, workers, context_workers=4
                )
                current = _run_workers(
                    self.current_source, workers, context_workers=4
                )
                self.assertEqual(legacy, current)
                self.assertEqual(
                    {result[1] for result in current}, set(range(workers))
                )

    def test_legacy_exception_boundary_and_rng_state_match(self):
        failing_seed = None
        for seed in range(64):
            result, _ = _run_n_equals_one(self.legacy_source, seed)
            if result[0] == "error":
                failing_seed = seed
                break
        self.assertIsNotNone(failing_seed)
        legacy = _run_n_equals_one(self.legacy_source, failing_seed)
        current = _run_n_equals_one(self.current_source, failing_seed)
        self.assertEqual(legacy, current)


if __name__ == "__main__":
    unittest.main()
