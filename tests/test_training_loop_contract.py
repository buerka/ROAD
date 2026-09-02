import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch
import torch.nn.functional as F


# ``utils.args`` parses argv while ``train`` is imported.  Keep test-runner
# flags away from the paper code, as in test_low_memory_data.py.
_test_runner_argv = sys.argv
sys.argv = [sys.argv[0]]
try:
    import fine_tune as fine_tune_module
    import train
finally:
    sys.argv = _test_runner_argv


class _SilentProgress:
    """Small tqdm stand-in that keeps the production loop shape intact."""

    def __init__(self, iterable, unit=None):
        self._iterable = iterable

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def __iter__(self):
        return iter(self._iterable)

    def set_description(self, *_args, **_kwargs):
        pass

    def set_postfix(self, *_args, **_kwargs):
        pass


class _NoOpAdam:
    """Avoid optimizer work while retaining a real autograd/backward pass."""

    def __init__(self, parameters, lr):
        self._parameters = list(parameters)

    def zero_grad(self):
        for parameter in self._parameters:
            parameter.grad = None

    def step(self):
        pass


class _TracedModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(1.0))
        self.forward_trace = []

    def _record_forward(self):
        self.forward_trace.append((self.training, torch.is_grad_enabled()))

    def save(self, *_args, **_kwargs):
        pass

    def load(self, *_args, **_kwargs):
        pass


class _TracedBackbone(_TracedModule):
    def forward(self, inputs):
        self._record_forward()
        return inputs.mean(dim=(-2, -1)) * self.scale


class _TracedSupervisedBackbone(_TracedBackbone):
    @staticmethod
    def loss_function(prediction, target):
        return F.cross_entropy(prediction.float(), target)


class _TracedPositionClassifier(_TracedModule):
    def __init__(self):
        super().__init__()
        self.bias = torch.nn.Parameter(torch.arange(8, dtype=torch.float32))

    def forward(self, data_latent, neighbour_latent):
        self._record_forward()
        signal = (data_latent + neighbour_latent).mean(dim=1, keepdim=True)
        return signal * self.scale + self.bias.unsqueeze(0)

    @staticmethod
    def loss_function(prediction, target):
        return F.cross_entropy(prediction.float(), target)


class _TracedDecoder(_TracedModule):
    def forward(self, latent):
        self._record_forward()
        return (latent * self.scale).unsqueeze(-1).unsqueeze(-1).expand(
            -1, -1, 2, 2
        )

    @staticmethod
    def loss_function(target, reconstruction):
        return torch.mean((target.float() - reconstruction.float()) ** 2)


class _TracedBinaryClassifier(_TracedModule):
    def forward(self, latent):
        self._record_forward()
        return torch.sigmoid(latent.mean(dim=1, keepdim=True) * self.scale)


class _FloatBCELoss:
    """CPU bfloat16-safe differentiable stand-in for the loop contract test."""

    def __call__(self, prediction, target):
        return torch.mean((prediction.float() - target.float()) ** 2)


class _ToggleDataset:
    def __init__(self):
        self.supervision_values = []

    def set_supervision(self, value):
        self.supervision_values.append(value)


class _Loader:
    def __init__(self, batches):
        self._batches = batches
        self.dataset = _ToggleDataset()

    def __len__(self):
        return len(self._batches)

    def __iter__(self):
        return iter(self._batches)


class _ValidationDataset:
    def __init__(self):
        self.data = torch.arange(64, dtype=torch.float32).reshape(1, 4, 4, 4)
        self.patch_calls = 0
        self.context_calls = 0
        self.cached_patches = None

    def patch(self, data):
        self.patch_calls += 1
        if data is not self.data:
            raise AssertionError("train_ssl must patch val_dataset.data")
        self.cached_patches = data.unfold(2, 2, 2).unfold(3, 2, 2)
        self.cached_patches = self.cached_patches.permute(
            0, 2, 3, 1, 4, 5
        ).reshape(-1, 4, 2, 2)
        return self.cached_patches

    def context_prediction(self, patches):
        self.context_calls += 1
        if patches is not self.cached_patches:
            raise AssertionError("context prediction must reuse cached patches")
        labels = torch.zeros(len(patches), dtype=torch.long)
        return labels, torch.flip(patches, dims=(-1,))


def _training_batches(count):
    batches = []
    for batch_index in range(count):
        data = torch.arange(32, dtype=torch.float32).reshape(1, 2, 4, 2, 2)
        data = data + batch_index
        labels = torch.zeros((1, 2), dtype=torch.long)
        batches.append((data, torch.zeros(1), labels, torch.flip(data, (-1,))))
    return batches


def _assert_train_and_validation_trace(test_case, module, step_count,
                                       forwards_per_step=1):
    train_events = [event for event in module.forward_trace if event[0]]
    validation_events = [event for event in module.forward_trace if not event[0]]
    expected = forwards_per_step * step_count
    test_case.assertEqual(len(train_events), expected)
    test_case.assertEqual(len(validation_events), expected)
    test_case.assertTrue(all(grad_enabled for _, grad_enabled in train_events))
    test_case.assertTrue(all(
        not grad_enabled for _, grad_enabled in validation_events
    ))


class TrainingLoopContractTests(unittest.TestCase):
    def test_supervised_validation_disables_grad_without_skipping_forwards(self):
        epochs = 2
        batch_count = 3
        batches = []
        for batch_index in range(batch_count):
            data = torch.arange(32, dtype=torch.float32).reshape(2, 4, 2, 2)
            batches.append((
                data + batch_index,
                torch.zeros(2, dtype=torch.long),
                ["synthetic"] * 2,
            ))
        loader = _Loader(batches)
        validation = SimpleNamespace(
            data=torch.arange(32, dtype=torch.float32).reshape(2, 4, 2, 2),
            labels=torch.zeros(2, dtype=torch.long),
        )
        backbone = _TracedSupervisedBackbone()

        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(
                model_path=str(Path(directory)),
                model_name="supervised-loop-contract",
                device=torch.device("cpu"),
                learning_rate=1e-3,
                epochs=epochs,
            )
            with mock.patch.object(train, "tqdm", _SilentProgress), \
                    mock.patch.object(train.torch.optim, "Adam", _NoOpAdam), \
                    mock.patch.object(train, "loss_curve"):
                train.train_supervised(loader, validation, backbone, args)

        self.assertEqual(loader.dataset.supervision_values, [True])
        _assert_train_and_validation_trace(
            self, backbone, epochs * batch_count
        )

    def test_ssl_caches_val_patches_without_removing_any_forwards(self):
        """Freeze the performance change without changing SSL sampling cadence."""

        epochs = 2
        batch_count = 3
        step_count = epochs * batch_count
        validation = _ValidationDataset()
        backbone = _TracedBackbone()
        position_classifier = _TracedPositionClassifier()
        decoder = _TracedDecoder()

        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(
                model_path=str(Path(directory)),
                model_name="loop-contract",
                device=torch.device("cpu"),
                learning_rate=1e-3,
                epochs=epochs,
            )
            with mock.patch.object(train, "tqdm", _SilentProgress), \
                    mock.patch.object(train.torch.optim, "Adam", _NoOpAdam), \
                    mock.patch.object(train, "loss_curve"):
                train.train_ssl(
                    _training_batches(batch_count),
                    validation,
                    backbone,
                    position_classifier,
                    decoder,
                    args,
                )

        # Only the deterministic patch construction is hoisted.  The random
        # context sampler still runs once per train step, as in the paper code.
        self.assertEqual(validation.patch_calls, 1)
        self.assertEqual(validation.context_calls, step_count)

        backbone_train = [event for event in backbone.forward_trace if event[0]]
        backbone_val = [event for event in backbone.forward_trace if not event[0]]
        classifier_train = [
            event for event in position_classifier.forward_trace if event[0]
        ]
        classifier_val = [
            event for event in position_classifier.forward_trace if not event[0]
        ]

        # Original SSL cadence: two backbone train forwards, two backbone val
        # forwards, one classifier train/val forward, and two decoder train
        # forwards per optimization step.
        self.assertEqual(len(backbone_train), 2 * step_count)
        self.assertEqual(len(backbone_val), 2 * step_count)
        self.assertEqual(len(classifier_train), step_count)
        self.assertEqual(len(classifier_val), step_count)
        self.assertEqual(len(decoder.forward_trace), 2 * step_count)

        self.assertTrue(all(grad_enabled for _, grad_enabled in backbone_train))
        self.assertTrue(all(grad_enabled for _, grad_enabled in classifier_train))
        self.assertTrue(all(grad_enabled for _, grad_enabled in decoder.forward_trace))
        self.assertTrue(all(not grad_enabled for _, grad_enabled in backbone_val))
        self.assertTrue(all(not grad_enabled for _, grad_enabled in classifier_val))

    def test_fine_tune_validation_disables_grad_without_skipping_forwards(self):
        # fine_tune intentionally fixes its schedule at 20 epochs.
        epochs = 20
        batch_count = 2
        batches = []
        for batch_index in range(batch_count):
            data = torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 2, 2)
            batches.append((
                data + batch_index,
                torch.full((1, 1), 9, dtype=torch.long),
                torch.zeros((1, 1), dtype=torch.long),
                torch.flip(data, (-1,)),
            ))
        loader = _Loader(batches)

        class FineTuneValidation:
            def __init__(self):
                self.data = torch.arange(16, dtype=torch.float32).reshape(
                    1, 4, 2, 2
                )
                self.labels = torch.ones(1, dtype=torch.long)
                self.patch_calls = 0

            def patch(self, data):
                self.patch_calls += 1
                return data

        validation = FineTuneValidation()
        backbone = _TracedBackbone()
        classifier = _TracedBinaryClassifier()
        args = SimpleNamespace(
            model_name="fine-tune-loop-contract",
            device=torch.device("cpu"),
            patch_size=2,
            latent_dim=4,
        )
        zero_metrics = (
            torch.tensor(0.0), torch.tensor([0.0]), torch.tensor(0.0)
        )

        with mock.patch.object(fine_tune_module, "tqdm", _SilentProgress), \
                mock.patch.object(
                    fine_tune_module.torch.optim, "Adam", _NoOpAdam
                ), \
                mock.patch.object(fine_tune_module.nn, "BCELoss", _FloatBCELoss), \
                mock.patch.object(fine_tune_module, "compute_metrics",
                                  return_value=zero_metrics), \
                mock.patch.object(fine_tune_module, "loss_curve"), \
                mock.patch.object(fine_tune_module.os.path, "exists",
                                  return_value=True), \
                mock.patch.object(fine_tune_module.defaults, "SIZE", (2, 2)):
            fine_tune_module.fine_tune(
                loader, validation, None, backbone, classifier, args
            )

        self.assertEqual(loader.dataset.supervision_values, [False])
        self.assertEqual(validation.patch_calls, 1)
        step_count = epochs * batch_count
        _assert_train_and_validation_trace(self, backbone, step_count)
        _assert_train_and_validation_trace(self, classifier, step_count)


if __name__ == "__main__":
    unittest.main()
