"""
    Dataloaders for LOFAR
"""
import os
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

from utils.data import defaults
from utils.data.lazy_h5 import (NormalizedMemmapStore,
                                build_catalog,
                                normalise_sample)


def _fixed_resized_crop(task):
    """Apply one already-sampled context crop without consuming RNG state."""
    (output_index,
     source_patch,
     crop_parameters,
     size,
     interpolation,
     antialias) = task
    top, left, height, width = crop_parameters
    resized = TF.resized_crop(source_patch,
                              top,
                              left,
                              height,
                              width,
                              size,
                              interpolation,
                              antialias=antialias)
    return output_index, resized


def get_data(args,
             remove: str = None,
             transform=None) -> (Dataset, Dataset, Dataset, Dataset, Dataset):
    """
        Constructs datasets and loaders for training, validation and testing
        Test data for supervised and unsupervised must be the same

        Parameters
        ----------
        args: cmd args
        remove: name of class to be excluded from training set
        transform: transform for dataloader

        Returns
        -------
        train_dataset: ...
        val_dataset: ...
        test_dataset: ...
        supervised_train_dataset: ...
        supervised_val_dataset: ...
    """
    catalog = build_catalog(args.data_path, defaults.anomalies)
    print(
        "ROAD catalog: "
        f"{catalog.physical_sample_count} physical rows, "
        f"{catalog.unique_sample_count} unique samples, "
        f"{catalog.multi_label_sample_count} multi-label samples. "
        "Preserving the original exact-single-label protocol with "
        f"{catalog.experiment_sample_count} samples."
    )

    cache_dir = getattr(args, 'data_cache_path', None)
    if cache_dir is None:
        cache_root = getattr(args, 'model_path', None)
        if cache_root is None:
            cache_root = os.path.dirname(os.path.abspath(args.data_path))
        cache_dir = os.path.join(cache_root, '.road_cache')

    store = NormalizedMemmapStore(args.data_path,
                                  catalog.experiment_records,
                                  args.amount,
                                  cache_dir)

    n_train = len(catalog.train_records)
    n_evaluation = len(catalog.evaluation_records)
    ssl_indexes = np.arange(n_train, dtype=np.int64)
    evaluation_indexes = np.arange(n_train,
                                   n_train + n_evaluation,
                                   dtype=np.int64)

    # Preserve the original split direction and ordering exactly.  The first
    # return value is deliberately named test_indexes in the source paper.
    (test_positions,
     train_positions) = train_test_split(np.arange(n_evaluation),
                                         test_size=args.percentage_data,
                                         random_state=args.seed)
    test_indexes = evaluation_indexes[test_positions]
    train_indexes = evaluation_indexes[train_positions]

    if args.percentage_data != 0.5:  # always test on at most 50% of data
        test_indexes = test_indexes[:n_evaluation//2]

    (supervised_train_indexes,
     supervised_val_indexes) = train_test_split(train_indexes,
                                                test_size=0.05,
                                                random_state=args.seed)
    (ssl_train_indexes,
     ssl_val_indexes) = train_test_split(ssl_indexes,
                                         test_size=0.05,
                                         random_state=args.seed)

    supervised_train_dataset = LOFARDataset(
        store,
        supervised_train_indexes,
        args,
        test=False,
        transform=transform,
        remove=remove,
        roll=False,
        supervised=True,
    )
    supervised_val_dataset = LOFARDataset(
        store,
        supervised_val_indexes,
        args,
        test=False,
        transform=None,
        remove=remove,
        supervised=True,
    )
    train_dataset = LOFARDataset(
        store,
        ssl_train_indexes,
        args,
        test=False,
        transform=transform,
        roll=False,
        remove=None,
        supervised=False,
    )
    val_dataset = LOFARDataset(
        store,
        ssl_val_indexes,
        args,
        test=False,
        transform=None,
        remove=None,
        supervised=False,
    )
    test_dataset = LOFARDataset(
        store,
        test_indexes,
        args,
        test=True,
        transform=None,
        remove=None,
        supervised=False,
    )

    return (train_dataset,
            val_dataset,
            test_dataset,
            supervised_train_dataset,
            supervised_val_dataset)


class LOFARDataset(Dataset):
    def __init__(self,
                 store: NormalizedMemmapStore,
                 indexes: np.ndarray,
                 args,
                 test: bool,
                 transform=None,
                 remove=None,
                 roll=False,
                 supervised=False):

        self._store = store
        self._base_indexes = np.asarray(indexes, dtype=np.int64)
        self.supervised = supervised
        self.test = test
        self.test_seed = args.seed
        self.remove = remove
        self.args = args
        self.anomaly_mask = []
        self.original_anomaly_mask = []
        self.n_patches = int(defaults.SIZE[0]/args.patch_size)
        self.context_workers = int(getattr(args, 'context_workers', 1))
        if self.context_workers < 0:
            raise ValueError("context_workers must be non-negative")
        self._context_executor = None
        self._context_executor_pid = None

        raw_labels = self._store.raw_labels[self._base_indexes]
        sources = self._store.sources[self._base_indexes]
        sample_ids = self._store.sample_ids[self._base_indexes]

        if not self.test and args.ood != -1:
            np.random.seed(self.test_seed)
            classes = np.random.choice(defaults.anomalies,
                                       size=args.ood,
                                       replace=False)
            print(f'removing {classes}')
            mask = np.ones(len(raw_labels), dtype=bool)
            for c in classes:
                mask = np.logical_and(mask, raw_labels != c)
            # Deliberately preserve the published code's no-op: it computed
            # this OOD mask but never applied it.  Applying it here would
            # change the experiment rather than only its storage behavior.

        if remove is not None:
            mask = raw_labels != remove
            self._base_indexes = self._base_indexes[mask]
            raw_labels = raw_labels[mask]
            sources = sources[mask]
            sample_ids = sample_ids[mask]

        self._raw_labels = raw_labels
        self._labels = torch.from_numpy(self.encode_labels(raw_labels))
        self._source = sources
        self._sample_ids = sample_ids
        self._active_positions = np.arange(len(self._base_indexes),
                                           dtype=np.int64)
        self._materialized_data = None
        self._materialized_frequency_band = None

        self.resizer = T.RandomResizedCrop(scale=(1.0-args.resize_amount, 1.0),
                                           size=(args.patch_size,
                                                 args.patch_size),
                                           antialias=False)
        self.transform = transform
        self.set_anomaly_mask(-1)

    def _get_context_executor(self):
        """Return this process's lazily-created context crop thread pool."""
        if self.context_workers <= 1:
            return None

        process_id = os.getpid()
        if self._context_executor_pid != process_id:
            # A ThreadPoolExecutor cannot survive fork.  Do not call shutdown
            # on an executor inherited from the parent: its worker threads do
            # not exist in the child process.
            self._context_executor = None
            self._context_executor_pid = None

        if self._context_executor is None:
            self._context_executor = ThreadPoolExecutor(
                max_workers=self.context_workers,
                thread_name_prefix="road-context",
            )
            self._context_executor_pid = process_id
        return self._context_executor

    def _shutdown_context_executor(self, wait=True, cancel_futures=False):
        executor = getattr(self, '_context_executor', None)
        executor_pid = getattr(self, '_context_executor_pid', None)
        self._context_executor = None
        self._context_executor_pid = None
        if executor is not None and executor_pid == os.getpid():
            executor.shutdown(wait=wait, cancel_futures=cancel_futures)

    def __getstate__(self):
        """Keep Windows spawn/DataLoader pickling free of thread handles."""
        state = self.__dict__.copy()
        state['_context_executor'] = None
        state['_context_executor_pid'] = None
        return state

    def __del__(self):
        try:
            self._shutdown_context_executor(wait=False,
                                            cancel_futures=True)
        except Exception:
            pass

    def __len__(self):
        return len(self._active_positions)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        position = int(self._active_positions[idx])
        store_index = int(self._base_indexes[position])
        datum = torch.from_numpy(self._store.get_data(store_index))

        if self.supervised:
            label = self._labels[position]
            source = self._source[position]
            return datum, label, source

        else:
            label = np.repeat(self._labels[position], self.n_patches**2,
                              axis=0)
            datum = self.patch(datum.unsqueeze(0))

            (context_label,
             context_image_neighbour) = self.context_prediction(datum)

            if self.transform:
                datum = self.transform(datum)

            return datum, label, context_label, context_image_neighbour

    @property
    def active_store_indexes(self) -> np.ndarray:
        """Indices into the one shared cache for the current dataset view."""

        return self._base_indexes[self._active_positions]

    @property
    def data(self) -> torch.Tensor:
        """Materialize the active view only when legacy callers require it."""

        if self._materialized_data is None:
            values = self._store.materialize_data(self.active_store_indexes)
            self._materialized_data = torch.from_numpy(values)
        return self._materialized_data

    @property
    def frequency_band(self) -> torch.Tensor:
        """Load frequency arrays only if a caller actually asks for them."""

        if self._materialized_frequency_band is None:
            values = self._store.materialize_frequency(
                self.active_store_indexes
            )
            self._materialized_frequency_band = torch.from_numpy(values)
        return self._materialized_frequency_band

    @property
    def labels(self) -> torch.Tensor:
        return self._labels[self._active_positions]

    @property
    def source(self) -> np.ndarray:
        return self._source[self._active_positions]

    @property
    def sample_ids(self) -> np.ndarray:
        return self._sample_ids[self._active_positions]

    @property
    def raw_labels(self) -> np.ndarray:
        return self._raw_labels[self._active_positions]

    def _invalidate_materialized_views(self) -> None:
        self._materialized_data = None
        self._materialized_frequency_band = None

    def set_supervision(self, supervised: bool) -> None:
        """
            sets supervision flag

            Parameters
            ----------
            supervised: ...

            Returns
            -------
            None
        """
        self.supervised = supervised

    def remove_sources(self, remove: np.array):
        """
            removes data corresponding source

            Parameters
            ----------
            remove: array of sources to be removed

            Returns
            -------
            None
        """
        _, _, indxs = np.intersect1d(remove,
                                     self._source,
                                     assume_unique=True,
                                     return_indices=True)
        mask = [i not in indxs for i in range(len(self._source))]
        self._base_indexes = self._base_indexes[mask]
        self._raw_labels = self._raw_labels[mask]
        self._labels = self._labels[mask]
        self._source = self._source[mask]
        self._sample_ids = self._sample_ids[mask]
        self._invalidate_materialized_views()
        self.set_anomaly_mask(-1)

    def set_seed(self, seed: int) -> None:
        """
            sets test data seed

            Parameters
            ----------
            seed: seed for split

            Returns
            -------
            None
        """
        self.test_seed = seed
        self.set_anomaly_mask(-1)

    def set_anomaly_mask(self, anomaly: int):
        """
            Sets the mask for the dataloader to load only specific classes

            Parameters
            ----------
            anomaly: anomaly class for mask, -1 for all

            Returns
            -------
            None
        """

        assert (anomaly in np.arange(len(defaults.anomalies)) or
                anomaly == -1), "Anomaly not found"

        if self.test:
            subsample_mask = self.subsample(self._labels)
        else:
            subsample_mask = np.arange(len(self._base_indexes), dtype=np.int64)

        subsample_mask = np.asarray(subsample_mask, dtype=np.int64)
        subsample_labels = self._labels[subsample_mask]

        if anomaly == -1:
            self.anomaly_mask = [True]*len(subsample_mask)
        else:
            self.anomaly_mask = [((anomaly == _l) |
                                  (_l == len(defaults.anomalies)))
                                 for _l in subsample_labels]

        self._active_positions = subsample_mask[
            np.asarray(self.anomaly_mask, dtype=bool)
        ]
        self._invalidate_materialized_views()

    def subsample(self, labels: np.array) -> np.array:
        """
            Subsamples dataset to enforce percentage_comtamination

            Parameters
            ----------
            labels: numpy array containing labels
            seed: random seed for sampling

            Returns
            -------
            mask
        """

        np.random.seed(self.test_seed)
        mask = np.array([], dtype=int)
        _len_ = len(labels[labels == len(defaults.anomalies)])
        for i, a in enumerate(defaults.percentage_comtamination):
            _amount = int(_len_*defaults.percentage_comtamination[a])
            _indices = [j for j, x in enumerate(labels) if x == i]
            _indices = np.random.choice(_indices, _amount, replace=False)
            mask = np.concatenate([mask, _indices], axis=0)
        mask = np.concatenate([mask,
                              [i for i, x in enumerate(labels)
                                  if x == len(defaults.anomalies)]],
                              axis=0)

        return mask.astype(int)

    def encode_labels(self, labels: np.array) -> np.array:
        """
           encodes labels to integer notation

            Parameters
            ----------
            labels: array of strings

            Returns
            -------
            encoded_labels: array of ints

        """
        out = []
        for label in labels:
            if label == '':
                out.append(len(defaults.anomalies))
            else:
                out.append([i for i, a in enumerate(defaults.anomalies)
                            if a in label][0])
        return np.array(out)

    def normalise(self, data: np.array) -> np.array:
        """
            perpendicular polarisation normalisation

            Parameters
            ----------
            data: ...

            Returns
            -------
            normalised_data: ...

        """
        if len(data) == 0:
            return np.zeros(data.shape)
        return np.stack([
            normalise_sample(sample, self.args.amount)
            for sample in data
        ])

    def patch(self, _input: torch.tensor) -> torch.tensor:
        """
            Makes (N,C,h,w) shaped tensor into
                  (N*(h/size)*(w/size),C,h/size, w/size)
            Note: this only works for square patches sizes

            Parameters
            ----------
            input: (N,C,H,w) tensor

            Returns
            -------
            patches: tensor of patches reshaped to
                     (N*(h/size)*(w/size),C,h/size, w/size)

        """
        unfold = _input.data.permute(0, 2, 3, 1)
        unfold = unfold.unfold(1,
                               self.args.patch_size,
                               self.args.patch_size)
        unfold = unfold.unfold(2,
                               self.args.patch_size,
                               self.args.patch_size)
        unfold = unfold.contiguous()

        patches = unfold.view(-1,
                              _input.shape[1],
                              self.args.patch_size,
                              self.args.patch_size)
        return patches

    def context_prediction(self,
                           data: torch.tensor) -> (torch.tensor, torch.tensor):
        """
            Arranges a context prediction dataset

            Parameters
            ----------
            data: indexed patched data

            Returns
            -------
            context_neighbour: tensor of patches
            context_label: tensor of context labels

        """

        context_labels = np.ones([data.shape[0]], dtype='int')
        context_images_neighbour = np.zeros([data.shape[0],
                                             data.shape[1],
                                             self.args.patch_size,
                                             self.args.patch_size],
                                            dtype='float32')
        _indx = 0
        parallel_crops = self.context_workers > 1
        crop_tasks = [] if parallel_crops else None
        _locations = [-self.n_patches-1,
                      -self.n_patches,
                      -self.n_patches+1,
                      -1,
                      +1,
                      self.n_patches-1,
                      self.n_patches,
                      +self.n_patches+1]

        for _image_indx in range(self.n_patches**2,
                                 data.shape[0]+1,
                                 self.n_patches**2):
            temp_patches = data[_image_indx-self.n_patches**2:_image_indx,
                                ...]

            for _patch_index in range(self.n_patches**2):
                if _patch_index < self.n_patches:
                    # TOPRIGHT
                    if _patch_index % self.n_patches == self.n_patches-1:
                        context_labels[_indx] = np.random.choice([3, 5, 6])

                    # TOPLEFT
                    elif _patch_index % self.n_patches == 0:
                        context_labels[_indx] = np.random.choice([4, 6, 7])

                    else:  # TOPMIDDLE
                        context_labels[_indx] = np.random.choice([3, 4,
                                                                  5, 6, 7])

                elif _patch_index >= self.n_patches**2 - self.n_patches:
                    # BOTTOMRIGHT
                    if _patch_index % self.n_patches == self.n_patches-1:
                        context_labels[_indx] = np.random.choice([0, 1, 3])

                    # BOTTOMLEFT
                    elif _patch_index % self.n_patches == 0:
                        context_labels[_indx] = np.random.choice([1, 2, 4])

                    else:  # BOTTOMMIDDLE
                        context_labels[_indx] = np.random.choice([0, 1,
                                                                  2, 3, 4])
                # RIGHT
                elif _patch_index % self.n_patches == self.n_patches-1:
                    context_labels[_indx] = np.random.choice([0, 1, 3, 5, 6])
                # LEFT
                elif _patch_index % self.n_patches == 0:
                    context_labels[_indx] = np.random.choice([1, 2, 4, 6, 7])

                else:  # MIDDEL
                    context_labels[_indx] = np.random.choice([0, 1, 2, 3,
                                                              4, 5, 6, 7])

                _ni = _patch_index + _locations[context_labels[_indx]]
                source_patch = temp_patches[_ni]
                if parallel_crops:
                    # Keep this RNG call on the main thread and immediately
                    # after the matching NumPy label draw.  Moving it into a
                    # worker would make seeded runs scheduling-dependent.
                    crop_parameters = T.RandomResizedCrop.get_params(
                        source_patch,
                        self.resizer.scale,
                        self.resizer.ratio,
                    )
                    crop_tasks.append((_indx,
                                       source_patch,
                                       crop_parameters,
                                       self.resizer.size,
                                       self.resizer.interpolation,
                                       self.resizer.antialias))
                else:
                    # Preserve the published serial path exactly, including
                    # when crop errors happen relative to later RNG draws.
                    resized_patch = self.resizer(source_patch)
                    context_images_neighbour[_indx, :] = resized_patch
                _indx += 1

        if parallel_crops and crop_tasks:
            executor = self._get_context_executor()
            # Bound submitted work so a large validation set cannot retain a
            # second full set of completed crop tensors while waiting for the
            # main thread to write results in the original index order.
            pending = deque()
            max_pending = max(1, self.context_workers * 2)
            tasks = iter(crop_tasks)
            try:
                for _ in range(max_pending):
                    task = next(tasks, None)
                    if task is None:
                        break
                    pending.append(executor.submit(_fixed_resized_crop, task))

                while pending:
                    future = pending.popleft()
                    output_index, resized_patch = future.result()
                    context_images_neighbour[output_index, :] = resized_patch
                    task = next(tasks, None)
                    if task is not None:
                        pending.append(
                            executor.submit(_fixed_resized_crop, task)
                        )
            except BaseException:
                for future in pending:
                    future.cancel()
                self._shutdown_context_executor(wait=True,
                                                cancel_futures=True)
                raise
        context_labels = torch.from_numpy(context_labels)
        context_images_neighbour = torch.from_numpy(context_images_neighbour)

        return context_labels, context_images_neighbour
