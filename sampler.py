"""8 subjects x 4 classes x 3 examples = 96; 21 cross-subject positives per anchor."""
from __future__ import annotations

from itertools import combinations

import numpy as np
from torch.utils.data import Sampler


class SubjectClassBalancedBatchSampler(Sampler[list[int]]):
    """Enforces full subject-class grid to ensure valid PA-SCL positives.

    If the data lack an 8-subject x 4-class complete submatrix, raise an
    actionable error rather than silently breaking the paper's batch design.
    Samples WITH replacement within a subject/class when necessary.
    """

    def __init__(self, labels, subjects, subjects_per_batch=8,
                 classes_per_batch=4, examples_per_cell=3, steps_per_epoch=100,
                 seed=42):
        self.labels = np.asarray(labels, dtype=np.int64)
        self.subjects = np.asarray(subjects, dtype=np.int64)
        if len(self.labels) != len(self.subjects):
            raise ValueError("labels and subjects length mismatch")
        if min(subjects_per_batch, classes_per_batch, examples_per_cell, steps_per_epoch) < 1:
            raise ValueError("Sampler sizes must be positive")
        self.subjects_per_batch = subjects_per_batch
        self.classes_per_batch = classes_per_batch
        self.examples_per_cell = examples_per_cell
        self.steps_per_epoch = steps_per_epoch
        self.seed = seed
        self.epoch = 0
        self.lookup = {}
        for subject in np.unique(self.subjects):
            for klass in np.unique(self.labels[self.subjects == subject]):
                idx = np.flatnonzero((self.subjects == subject) & (self.labels == klass))
                self.lookup[(int(subject), int(klass))] = idx
        classes = np.unique(self.labels).tolist()
        if len(classes) < classes_per_batch:
            raise ValueError(f"Need at least {classes_per_batch} distinct classes")
        # Enumerate all possible class subsets, each with >= S eligible subjects.
        self.combos: list[tuple[tuple[int, ...], np.ndarray]] = []
        for selected in combinations(classes, classes_per_batch):
            eligible = [s for s in np.unique(self.subjects)
                        if all((int(s), int(c)) in self.lookup for c in selected)]
            if len(eligible) >= subjects_per_batch:
                self.combos.append((selected, np.asarray(eligible)))
        if not self.combos:
            raise ValueError(
                f"No complete {subjects_per_batch}-subject x {classes_per_batch}-class grid "
                "exists. Check labels, or explicitly reduce sampler composition."
            )

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def __len__(self):
        return self.steps_per_epoch

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch * 100003)
        for _ in range(self.steps_per_epoch):
            classes, eligible = self.combos[rng.integers(0, len(self.combos))]
            subjects = rng.choice(eligible, size=self.subjects_per_batch, replace=False)
            batch = []
            for s in subjects:
                for c in classes:
                    choices = self.lookup[(int(s), int(c))]
                    replace = len(choices) < self.examples_per_cell
                    batch.extend(rng.choice(choices, size=self.examples_per_cell, replace=replace).tolist())
            rng.shuffle(batch)
            yield batch
