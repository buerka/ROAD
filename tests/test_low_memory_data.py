import hashlib
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import h5py
import numpy as np
import torch
from sklearn.model_selection import train_test_split

# Importing the original ``utils`` package parses process arguments as a side
# effect.  Hide unittest/pytest flags while importing the paper code.
_test_runner_argv = sys.argv
sys.argv = [sys.argv[0]]
try:
    from data import LOFARDataset, get_data
    from utils.data import defaults
    from utils.data.lazy_h5 import (NormalizedMemmapStore,
                                    build_catalog,
                                    normalise_sample)
finally:
    sys.argv = _test_runner_argv


def legacy_normalise(data, amount):
    """The implementation from the published data.py, kept as an oracle."""

    output = np.zeros(data.shape)
    for index, spectrogram in enumerate(data):
        for polarization in range(data.shape[-1]):
            minimum, maximum = np.percentile(
                spectrogram[..., polarization],
                [amount, 100 - amount],
            )
            temporary = np.clip(
                spectrogram[..., polarization], minimum, maximum
            )
            temporary = np.log(temporary)
            temporary = (
                (temporary - np.min(temporary)) /
                (np.max(temporary) - np.min(temporary))
            )
            output[index, ..., polarization] = temporary
    return np.nan_to_num(output, 0)


def _sample(value, shape=(8, 8, 4)):
    result = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    result = result / np.float32(1000.0) + np.float32(value + 1.0)
    result[..., -1] = np.float32(value + 2.0)
    return result


def _write_group(h5_file, path, rows):
    group = h5_file.require_group(path)
    data = np.stack([row[3] for row in rows]) if rows else np.empty(
        (0, 8, 8, 4), dtype=np.float32
    )
    frequency = np.stack([
        np.full((8, 8, 1), row[4], dtype=np.float32)
        for row in rows
    ]) if rows else np.empty((0, 8, 8, 1), dtype=np.float32)
    group.create_dataset("data", data=data)
    group.create_dataset("frequency_band", data=frequency)
    group.create_dataset("ids", data=np.asarray(
        [row[0] for row in rows], dtype="S128"
    ))
    group.create_dataset("labels", data=np.asarray(
        [row[1] for row in rows], dtype="S256"
    ))
    group.create_dataset("source", data=np.asarray(
        [row[2] for row in rows], dtype="S128"
    ))


def _create_synthetic_h5(path):
    compound_label = f"{defaults.anomalies[0]}-{defaults.anomalies[1]}"
    compound_data = _sample(100)
    with h5py.File(path, "w") as h5_file:
        _write_group(h5_file, "train_data", [
            ("train-normal", "", "train-source", _sample(0), 10),
        ])
        _write_group(h5_file, "test_data", [
            ("test-normal", "", "test-source", _sample(1), 11),
        ])
        for index, anomaly in enumerate(defaults.anomalies):
            rows = [(
                f"exact-{index}", anomaly, f"source-{index}",
                _sample(index + 2), index + 20,
            )]
            if index in (0, 1):
                rows.append((
                    "compound-01", compound_label, "compound-source",
                    compound_data, 99,
                ))
            _write_group(h5_file, f"anomaly_data/{anomaly}", rows)


class LowMemoryDataTests(unittest.TestCase):
    def test_catalog_deduplicates_metadata_but_keeps_legacy_view(self):
        with tempfile.TemporaryDirectory() as directory:
            data_path = Path(directory) / "synthetic.h5"
            _create_synthetic_h5(data_path)
            catalog = build_catalog(str(data_path), defaults.anomalies)

            self.assertEqual(catalog.physical_sample_count, 13)
            self.assertEqual(catalog.unique_sample_count, 12)
            self.assertEqual(catalog.normal_sample_count, 2)
            self.assertEqual(catalog.anomaly_sample_count, 10)
            self.assertEqual(catalog.multi_label_sample_count, 1)
            self.assertEqual(catalog.duplicate_physical_row_count, 1)
            self.assertEqual(len(catalog.canonical_records), 12)

            # The model-facing view deliberately excludes the compound row,
            # exactly as _join(..., compound=False) did in the paper code.
            self.assertEqual(catalog.experiment_sample_count, 11)
            experiment_ids = [r.sample_id for r in catalog.experiment_records]
            self.assertNotIn("compound-01", experiment_ids)
            self.assertEqual(len(experiment_ids), len(set(experiment_ids)))

    def test_cache_matches_legacy_float64_values_and_layout(self):
        with tempfile.TemporaryDirectory() as directory:
            data_path = Path(directory) / "synthetic.h5"
            cache_path = Path(directory) / "cache"
            _create_synthetic_h5(data_path)
            catalog = build_catalog(str(data_path), defaults.anomalies)
            store = NormalizedMemmapStore(
                str(data_path), catalog.experiment_records, 0.1,
                str(cache_path),
            )

            self.assertEqual(store._open_array().dtype, np.dtype("float64"))
            self.assertEqual(store._open_array().shape, (11, 8, 8, 4))
            with h5py.File(data_path, "r") as h5_file:
                for index, record in enumerate(catalog.experiment_records):
                    raw = h5_file[f"{record.group}/data"][record.row:record.row + 1]
                    expected = legacy_normalise(raw, 0.1)
                    actual = store.get_data(index)
                    np.testing.assert_array_equal(
                        actual, expected[0].transpose(2, 0, 1)
                    )
                    self.assertEqual(actual.strides, (8, 256, 32))
                del actual

            materialized = torch.from_numpy(
                store.materialize_data(np.arange(3, dtype=np.int64))
            )
            self.assertEqual(materialized.stride(), (256, 1, 32, 4))

            modified_time = Path(store.cache_path).stat().st_mtime_ns
            store.close()
            reused = NormalizedMemmapStore(
                str(data_path), catalog.experiment_records, 0.1,
                str(cache_path),
            )
            self.assertEqual(Path(reused.cache_path).stat().st_mtime_ns,
                             modified_time)
            reused.close()

    def test_dataset_views_use_indexes_and_preserve_subsample_order(self):
        class MetadataOnlyStore:
            def __init__(self, labels):
                self.raw_labels = np.asarray(labels, dtype=str)
                self.sources = np.asarray([
                    f"source-{i}" for i in range(len(labels))
                ], dtype=str)
                self.sample_ids = np.asarray([
                    f"id-{i}" for i in range(len(labels))
                ], dtype=str)

            def get_data(self, index):
                raise AssertionError("metadata operations must not read data")

        labels = []
        for anomaly in defaults.anomalies:
            labels.extend([anomaly] * 100)
        labels.extend([""] * 500)
        store = MetadataOnlyStore(labels)
        args = SimpleNamespace(
            seed=42,
            patch_size=32,
            resize_amount=0.05,
            ood=-1,
            amount=0.1,
        )
        dataset = LOFARDataset(
            store, np.arange(len(labels)), args,
            test=True, supervised=True,
        )

        self.assertEqual(len(dataset), 670)
        np.testing.assert_array_equal(
            np.bincount(dataset.labels.numpy(), minlength=10),
            np.asarray([5, 5, 10, 20, 20, 30, 40, 30, 10, 500]),
        )
        active_before = dataset.active_store_indexes.copy()
        dataset.set_anomaly_mask(4)
        self.assertEqual(len(dataset), 520)
        self.assertTrue(np.all(np.isin(dataset.labels.numpy(), [4, 9])))
        dataset.set_anomaly_mask(-1)
        np.testing.assert_array_equal(dataset.active_store_indexes,
                                      active_before)

    def test_normalise_sample_is_the_old_operation_bit_for_bit(self):
        data = np.stack([_sample(2), _sample(7)])
        expected = legacy_normalise(data, 0.1)
        actual = np.stack([normalise_sample(sample, 0.1) for sample in data])
        self.assertEqual(actual.dtype, np.dtype("float64"))
        np.testing.assert_array_equal(actual, expected)


class PublishedDatasetMetadataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data_path = (
            Path(__file__).resolve().parents[3] / "ROAD_dataset.h5"
        )
        if not cls.data_path.exists():
            raise unittest.SkipTest("published ROAD_dataset.h5 is not available")
        cls.catalog = build_catalog(str(cls.data_path), defaults.anomalies)

    def test_published_counts(self):
        catalog = self.catalog
        self.assertEqual(catalog.physical_sample_count, 7053)
        self.assertEqual(catalog.unique_sample_count, 6708)
        self.assertEqual(len(catalog.canonical_records), 6708)
        self.assertEqual(catalog.normal_sample_count, 4687)
        self.assertEqual(catalog.anomaly_sample_count, 2021)
        self.assertEqual(catalog.multi_label_sample_count, 407)
        self.assertEqual(catalog.duplicate_physical_row_count, 345)
        self.assertEqual(catalog.experiment_sample_count, 6301)
        self.assertEqual(len(catalog.train_records), 3687)
        self.assertEqual(len(catalog.evaluation_records), 2614)

        membership_counts = [56, 88, 146, 283, 389, 261, 550, 446, 147]
        exact_single_counts = [45, 73, 145, 212, 331, 238, 250, 173, 147]
        with h5py.File(self.data_path, "r") as h5_file:
            for anomaly, membership, exact_single in zip(
                    defaults.anomalies,
                    membership_counts,
                    exact_single_counts):
                labels = h5_file[
                    f"anomaly_data/{anomaly}/labels"
                ][:].astype(str)
                self.assertEqual(len(labels), membership)
                self.assertEqual(np.sum(labels == anomaly), exact_single)

    def test_catalog_order_is_the_original_join_order(self):
        expected = []
        with h5py.File(self.data_path, "r") as h5_file:
            labels = h5_file["test_data/labels"][:].astype(str)
            sources = h5_file["test_data/source"][:].astype(str)
            sample_ids = h5_file["test_data/ids"][:].astype(str)
            expected.extend(zip(sample_ids, labels, sources))
            for anomaly in defaults.anomalies:
                group = f"anomaly_data/{anomaly}"
                labels = h5_file[f"{group}/labels"][:].astype(str)
                sources = h5_file[f"{group}/source"][:].astype(str)
                sample_ids = h5_file[f"{group}/ids"][:].astype(str)
                mask = labels == anomaly
                expected.extend(zip(sample_ids[mask], labels[mask],
                                    sources[mask]))

        actual = [
            (record.sample_id, record.label, record.source)
            for record in self.catalog.evaluation_records
        ]
        self.assertEqual(actual, expected)

    def test_representative_real_samples_match_legacy_normalise(self):
        selected = [self.catalog.train_records[0],
                    self.catalog.train_records[-1],
                    self.catalog.evaluation_records[0]]
        for anomaly in (defaults.anomalies[0], defaults.anomalies[4],
                        defaults.anomalies[5], defaults.anomalies[-1]):
            selected.append(next(
                record for record in self.catalog.evaluation_records
                if record.label == anomaly
            ))

        with h5py.File(self.data_path, "r") as h5_file:
            for record in selected:
                raw = h5_file[f"{record.group}/data"][
                    record.row:record.row + 1
                ]
                expected = legacy_normalise(raw, 0.1)
                actual = normalise_sample(raw[0], 0.1)[None, ...]
                np.testing.assert_array_equal(actual, expected)
                self.assertTrue(torch.equal(
                    torch.from_numpy(actual).to(torch.bfloat16),
                    torch.from_numpy(expected).to(torch.bfloat16),
                ))

    def test_seed_42_split_and_active_order_are_frozen(self):
        catalog = self.catalog
        evaluation_count = len(catalog.evaluation_records)
        test_positions, train_positions = train_test_split(
            np.arange(evaluation_count),
            test_size=0.5,
            random_state=42,
        )
        supervised_train, supervised_val = train_test_split(
            train_positions,
            test_size=0.05,
            random_state=42,
        )
        ssl_train, ssl_val = train_test_split(
            np.arange(len(catalog.train_records)),
            test_size=0.05,
            random_state=42,
        )
        self.assertEqual(
            (len(ssl_train), len(ssl_val), len(supervised_train),
             len(supervised_val)),
            (3502, 185, 1241, 66),
        )

        evaluation_labels = np.asarray([
            record.label for record in catalog.evaluation_records
        ], dtype=str)
        encoded = np.asarray([
            len(defaults.anomalies) if label == "" else
            defaults.anomalies.index(label)
            for label in evaluation_labels[test_positions]
        ])
        np.random.seed(42)
        selected = np.array([], dtype=int)
        normal_count = np.sum(encoded == len(defaults.anomalies))
        for label_index, anomaly in enumerate(
                defaults.percentage_comtamination):
            amount = int(
                normal_count * defaults.percentage_comtamination[anomaly]
            )
            candidates = np.flatnonzero(encoded == label_index)
            chosen = np.random.choice(candidates, amount, replace=False)
            selected = np.concatenate([selected, chosen])
        selected = np.concatenate([
            selected,
            np.flatnonzero(encoded == len(defaults.anomalies)),
        ]).astype(int)
        self.assertEqual(len(selected), 670)
        np.testing.assert_array_equal(
            np.bincount(encoded[selected], minlength=10),
            np.asarray([5, 5, 10, 20, 20, 30, 40, 30, 10, 500]),
        )

        active_ids = np.asarray([
            catalog.evaluation_records[index].sample_id
            for index in test_positions[selected]
        ], dtype=str)
        digest = hashlib.sha256("\0".join(active_ids).encode()).hexdigest()
        self.assertEqual(
            digest,
            "0c7d670195abcdc415b74e0ab0676d3d35d619bc0415a82d8d2e57d424e8c487",
        )

    def test_get_data_builds_five_index_views_over_one_store(self):
        class MetadataOnlyStore:
            def __init__(self, records):
                self.raw_labels = np.asarray([
                    record.label for record in records
                ], dtype=str)
                self.sources = np.asarray([
                    record.source for record in records
                ], dtype=str)
                self.sample_ids = np.asarray([
                    record.sample_id for record in records
                ], dtype=str)

            def get_data(self, index):
                raise AssertionError("dataset construction read image data")

        shared_store = MetadataOnlyStore(self.catalog.experiment_records)
        args = SimpleNamespace(
            data_path=str(self.data_path),
            data_cache_path=None,
            model_path=None,
            amount=0.1,
            percentage_data=0.5,
            seed=42,
            patch_size=32,
            resize_amount=0.05,
            ood=-1,
        )
        with mock.patch("data.NormalizedMemmapStore",
                        return_value=shared_store) as store_class:
            datasets = get_data(args)

        store_class.assert_called_once()
        self.assertTrue(all(dataset._store is shared_store
                            for dataset in datasets))
        self.assertTrue(all(dataset._materialized_data is None
                            for dataset in datasets))
        self.assertEqual(tuple(map(len, datasets)),
                         (3502, 185, 670, 1241, 66))

        expected_hashes = (
            "0cd3cae96c9dacd11974b0e2d420b5579d16700c694bd78e92dc69c2057b7465",
            "f0b5784e35d4e9b96609dd7b5e99cb5e6e3ef92dd8d1294f1cf6c08aa90074b0",
            "0c7d670195abcdc415b74e0ab0676d3d35d619bc0415a82d8d2e57d424e8c487",
            "ee6bc84a8fffdbb88c3bd13bd625b0cf6d50f71f42006d13eb30df96168e9e5c",
            "c89427f2c806bb54d077f9963101b74233ca0187a33506cc88e09b403a503052",
        )
        actual_hashes = tuple(
            hashlib.sha256(
                "\0".join(dataset.sample_ids.tolist()).encode("utf-8")
            ).hexdigest()
            for dataset in datasets
        )
        self.assertEqual(actual_hashes, expected_hashes)


if __name__ == "__main__":
    unittest.main()
