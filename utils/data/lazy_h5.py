"""Low-memory access to the ROAD HDF5 dataset.

The published HDF5 file stores a multi-label observation in every anomaly
group to which it belongs.  The training code, however, intentionally keeps
only labels that exactly match one anomaly name.  This module records both
counts, but builds the numerical cache only for the exact-label protocol so
that moving storage out of RAM does not change the experiment.
"""

from __future__ import annotations

import errno
import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import h5py
import numpy as np

if os.name == "nt":
    import msvcrt
else:
    import fcntl


CACHE_VERSION = 2
CACHE_LOCK_POLL_SECONDS = 2


@dataclass(frozen=True)
class SampleRecord:
    """Location and small metadata for one physical HDF5 sample."""

    group: str
    row: int
    sample_id: str
    label: str
    source: str


@dataclass(frozen=True)
class DatasetCatalog:
    """Metadata-only catalog for the full and legacy experiment datasets."""

    canonical_records: tuple[SampleRecord, ...]
    train_records: tuple[SampleRecord, ...]
    evaluation_records: tuple[SampleRecord, ...]
    physical_sample_count: int
    unique_sample_count: int
    normal_sample_count: int
    anomaly_sample_count: int
    multi_label_sample_count: int
    duplicate_physical_row_count: int

    @property
    def experiment_records(self) -> tuple[SampleRecord, ...]:
        return self.train_records + self.evaluation_records

    @property
    def experiment_sample_count(self) -> int:
        return len(self.train_records) + len(self.evaluation_records)


def _read_group_metadata(h5_file: h5py.File,
                         group: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    labels = h5_file[f"{group}/labels"][:].astype(str)
    sources = h5_file[f"{group}/source"][:].astype(str)
    sample_ids = h5_file[f"{group}/ids"][:].astype(str)
    return labels, sources, sample_ids


def _records_for_group(h5_file: h5py.File,
                       group: str) -> tuple[SampleRecord, ...]:
    labels, sources, sample_ids = _read_group_metadata(h5_file, group)
    return tuple(
        SampleRecord(group, row, sample_id, label, source)
        for row, (sample_id, label, source) in enumerate(
            zip(sample_ids, labels, sources)
        )
    )


def build_catalog(data_path: str,
                  anomaly_names: Sequence[str]) -> DatasetCatalog:
    """Build a metadata-only catalog without reading any image arrays.

    ``unique_sample_count`` is the ID-deduplicated publication count.  The
    experiment views retain the old ``label == anomaly`` filtering behavior,
    which excludes compound labels and therefore preserves the original
    single-label loss and class balance.
    """

    with h5py.File(data_path, "r") as h5_file:
        train_records = _records_for_group(h5_file, "train_data")
        normal_test_records = _records_for_group(h5_file, "test_data")

        all_records = list(train_records) + list(normal_test_records)
        evaluation_records = list(normal_test_records)

        for anomaly in anomaly_names:
            group = f"anomaly_data/{anomaly}"
            group_records = _records_for_group(h5_file, group)
            all_records.extend(group_records)
            evaluation_records.extend(
                record for record in group_records
                if record.label == anomaly
            )

    unique_records: dict[str, SampleRecord] = {}
    for record in all_records:
        previous = unique_records.get(record.sample_id)
        if previous is None:
            unique_records[record.sample_id] = record
            continue
        if (previous.label != record.label or
                previous.source != record.source):
            raise ValueError(
                "Duplicate ROAD sample ID has inconsistent metadata: "
                f"{record.sample_id}"
            )

    experiment_ids = [record.sample_id for record in train_records]
    experiment_ids.extend(record.sample_id for record in evaluation_records)
    if len(experiment_ids) != len(set(experiment_ids)):
        raise ValueError("The single-label ROAD experiment catalog is not unique")

    normal_count = len(train_records) + len(normal_test_records)
    multi_label_count = sum(
        "-" in record.label
        for record in unique_records.values()
        if record.label
    )

    return DatasetCatalog(
        canonical_records=tuple(unique_records.values()),
        train_records=tuple(train_records),
        evaluation_records=tuple(evaluation_records),
        physical_sample_count=len(all_records),
        unique_sample_count=len(unique_records),
        normal_sample_count=normal_count,
        anomaly_sample_count=len(unique_records) - normal_count,
        multi_label_sample_count=multi_label_count,
        duplicate_physical_row_count=len(all_records) - len(unique_records),
    )


def normalise_sample(data: np.ndarray, amount: float) -> np.ndarray:
    """Apply the original ROAD normalization to one sample.

    The intentionally implicit float64 output matches ``np.zeros(data.shape)``
    in the original implementation.  Keeping this intermediate precision
    avoids changing values before the existing model-side BF16 cast.
    """

    normalized = np.zeros(data.shape)
    for polarization in range(data.shape[-1]):
        minimum, maximum = np.percentile(
            data[..., polarization],
            [amount, 100 - amount],
        )
        temporary = np.clip(data[..., polarization], minimum, maximum)
        temporary = np.log(temporary)
        temporary = (
            (temporary - np.min(temporary)) /
            (np.max(temporary) - np.min(temporary))
        )
        normalized[..., polarization] = temporary
    return np.nan_to_num(normalized, 0)


def _cache_fingerprint(data_path: str,
                       records: Sequence[SampleRecord],
                       amount: float,
                       sample_shape: tuple[int, ...]) -> str:
    source_stat = os.stat(data_path)
    digest = hashlib.sha256()
    digest.update(str(CACHE_VERSION).encode())
    digest.update(os.path.abspath(data_path).encode())
    digest.update(str(source_stat.st_size).encode())
    digest.update(str(source_stat.st_mtime_ns).encode())
    digest.update(repr(float(amount)).encode())
    digest.update(repr(sample_shape).encode())
    for record in records:
        digest.update(record.group.encode())
        digest.update(str(record.row).encode())
        digest.update(record.sample_id.encode())
    return digest.hexdigest()


class NormalizedMemmapStore:
    """One normalized, disk-backed array shared by all dataset index views."""

    def __init__(self,
                 data_path: str,
                 records: Sequence[SampleRecord],
                 amount: float,
                 cache_dir: str):
        self.data_path = os.path.abspath(data_path)
        self.records = tuple(records)
        self.amount = float(amount)
        self.cache_dir = os.path.abspath(cache_dir)
        self._array = None
        self._array_pid = None
        self._h5_file = None
        self._h5_pid = None

        if not self.records:
            raise ValueError("Cannot build an empty ROAD data store")

        with h5py.File(self.data_path, "r") as h5_file:
            first = h5_file[f"{self.records[0].group}/data"]
            height, width, channels = first.shape[1:]
            first_frequency = h5_file[
                f"{self.records[0].group}/frequency_band"
            ]
            frequency_height, frequency_width, frequency_channels = (
                first_frequency.shape[1:]
            )
            self.frequency_dtype = first_frequency.dtype
        # Keep the normalized cache in the old NHWC layout.  Returning a
        # transposed view reproduces the channels-last strides created by the
        # original ``torch.from_numpy(...).permute(...)`` path, including for
        # validation tensors that bypass DataLoader collation.
        self.sample_shape = (height, width, channels)
        self.frequency_sample_shape = (
            frequency_height,
            frequency_width,
            frequency_channels,
        )

        fingerprint = _cache_fingerprint(
            self.data_path,
            self.records,
            self.amount,
            self.sample_shape,
        )
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
        stem = Path(self.data_path).stem
        filename = f"{stem}.normalized-f64.{fingerprint[:16]}.npy"
        self.cache_path = os.path.join(self.cache_dir, filename)
        self.metadata_path = f"{self.cache_path}.json"
        self._expected_metadata = {
            "cache_version": CACHE_VERSION,
            "fingerprint": fingerprint,
            "shape": [len(self.records), *self.sample_shape],
            "dtype": "float64",
            "amount": self.amount,
        }
        self._ensure_cache()

        self.raw_labels = np.asarray(
            [record.label for record in self.records],
            dtype=str,
        )
        self.sources = np.asarray(
            [record.source for record in self.records],
            dtype=str,
        )
        self.sample_ids = np.asarray(
            [record.sample_id for record in self.records],
            dtype=str,
        )

    def _cache_is_valid(self) -> bool:
        if not (os.path.exists(self.cache_path) and
                os.path.exists(self.metadata_path)):
            return False
        try:
            with open(self.metadata_path, "r", encoding="utf-8") as stream:
                metadata = json.load(stream)
            if metadata != self._expected_metadata:
                return False
            array = np.load(self.cache_path, mmap_mode="r")
            valid = (
                list(array.shape) == self._expected_metadata["shape"] and
                array.dtype == np.dtype(self._expected_metadata["dtype"])
            )
            del array
            return valid
        except (OSError, ValueError, json.JSONDecodeError):
            return False

    def _acquire_cache_lock(self, lock_path: str):
        announced_wait = False
        while True:
            descriptor = os.open(
                lock_path,
                os.O_CREAT | os.O_RDWR | getattr(os, "O_BINARY", 0),
                0o644,
            )
            try:
                if os.name == "nt":
                    if os.fstat(descriptor).st_size == 0:
                        os.write(descriptor, b"\0")
                    os.lseek(descriptor, 0, os.SEEK_SET)
                    msvcrt.locking(descriptor, msvcrt.LK_NBLCK, 1)
                else:
                    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                return descriptor
            except OSError as error:
                os.close(descriptor)
                if error.errno not in (errno.EACCES, errno.EAGAIN):
                    raise
                if self._cache_is_valid():
                    return None
                if not announced_wait:
                    print("ROAD cache: waiting for another cache builder")
                    announced_wait = True
                time.sleep(CACHE_LOCK_POLL_SECONDS)

    @staticmethod
    def _release_cache_lock(descriptor: int) -> None:
        try:
            if os.name == "nt":
                os.lseek(descriptor, 0, os.SEEK_SET)
                msvcrt.locking(descriptor, msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)

    def _ensure_cache(self) -> None:
        if self._cache_is_valid():
            return

        lock_path = f"{self.metadata_path}.lock"
        lock_descriptor = self._acquire_cache_lock(lock_path)
        if lock_descriptor is None:
            return
        try:
            # A different process may have completed between our initial
            # validity check and acquisition of the lock.
            if not self._cache_is_valid():
                self._remove_stale_temporary_files()
                self._build_cache()
        finally:
            self._release_cache_lock(lock_descriptor)

    def _remove_stale_temporary_files(self) -> None:
        filename_prefix = f"{os.path.basename(self.cache_path)}."
        with os.scandir(self.cache_dir) as entries:
            for entry in entries:
                if (entry.is_file() and
                        entry.name.startswith(filename_prefix) and
                        entry.name.endswith(".tmp")):
                    os.remove(entry.path)

    def _build_cache(self) -> None:

        temporary_path = f"{self.cache_path}.{os.getpid()}.tmp"
        temporary_metadata_path = f"{self.metadata_path}.{os.getpid()}.tmp"
        for path in (temporary_path, temporary_metadata_path):
            if os.path.exists(path):
                os.remove(path)

        shape = tuple(self._expected_metadata["shape"])
        records_by_group: dict[str, list[tuple[int, SampleRecord]]] = {}
        for cache_row, record in enumerate(self.records):
            records_by_group.setdefault(record.group, []).append(
                (cache_row, record)
            )

        completed = 0
        cache = None
        try:
            cache = np.lib.format.open_memmap(
                temporary_path,
                mode="w+",
                dtype=np.float64,
                shape=shape,
            )
            with h5py.File(self.data_path, "r") as h5_file:
                for group, indexed_records in records_by_group.items():
                    dataset = h5_file[f"{group}/data"]
                    if dataset.chunks:
                        chunk_rows = dataset.chunks[0]
                        block_size = chunk_rows * max(1, 64 // chunk_rows)
                    else:
                        block_size = 64
                    record_offset = 0
                    while record_offset < len(indexed_records):
                        first_row = indexed_records[record_offset][1].row
                        block_start = (first_row // block_size) * block_size
                        block_stop = min(block_start + block_size,
                                         len(dataset))
                        block = dataset[block_start:block_stop]
                        while record_offset < len(indexed_records):
                            cache_row, record = indexed_records[record_offset]
                            if record.row >= block_stop:
                                break
                            normalized = normalise_sample(
                                block[record.row - block_start],
                                self.amount,
                            )
                            cache[cache_row] = normalized
                            record_offset += 1
                            completed += 1
                        del block
                    print(
                        "ROAD cache: normalized "
                        f"{completed}/{len(self.records)} samples"
                    )
            cache.flush()
            cache = None
            os.replace(temporary_path, self.cache_path)
            with open(temporary_metadata_path, "w", encoding="utf-8") as stream:
                json.dump(self._expected_metadata, stream, indent=2)
            os.replace(temporary_metadata_path, self.metadata_path)
        except Exception:
            cache = None
            for path in (temporary_path, temporary_metadata_path):
                if os.path.exists(path):
                    os.remove(path)
            raise

    def _open_array(self) -> np.memmap:
        process_id = os.getpid()
        if self._array is None or self._array_pid != process_id:
            self._array = np.load(self.cache_path, mmap_mode="c")
            self._array_pid = process_id
        return self._array

    def get_data(self, index: int) -> np.ndarray:
        return np.asarray(
            self._open_array()[int(index)].transpose(2, 0, 1)
        )

    def materialize_data(self, indices: np.ndarray) -> np.ndarray:
        values = self._open_array()[np.asarray(indices, dtype=np.int64)]
        return np.asarray(values.transpose(0, 3, 1, 2))

    def _open_h5(self) -> h5py.File:
        process_id = os.getpid()
        if self._h5_file is None or self._h5_pid != process_id:
            if self._h5_file is not None:
                self._h5_file.close()
            self._h5_file = h5py.File(self.data_path, "r")
            self._h5_pid = process_id
        return self._h5_file

    def materialize_frequency(self, indices: np.ndarray) -> np.ndarray:
        indices = np.asarray(indices, dtype=np.int64)
        if len(indices) == 0:
            height, width, channels = self.frequency_sample_shape
            return np.empty(
                (0, channels, height, width),
                dtype=self.frequency_dtype,
            )
        h5_file = self._open_h5()
        values = [
            h5_file[f"{self.records[int(index)].group}/frequency_band"][
                self.records[int(index)].row
            ]
            for index in indices
        ]
        return np.stack(values).transpose(0, 3, 1, 2)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_array"] = None
        state["_array_pid"] = None
        state["_h5_file"] = None
        state["_h5_pid"] = None
        return state

    def close(self) -> None:
        self._array = None
        self._array_pid = None
        if self._h5_file is not None:
            self._h5_file.close()
        self._h5_file = None
        self._h5_pid = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
