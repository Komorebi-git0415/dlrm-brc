#!/usr/bin/env python3

"""
Generic dataset abstraction used by BRC frequency profiling.

Dataset-specific code is responsible for translating its canonical processed
representation into the effective table-local IDs that are actually consumed
by embedding lookup.

The generic BRC layer does not know the dataset name or embedding-table count.
"""

from abc import ABC, abstractmethod

from brc_frequency import count_table_frequencies


BRC_SAMPLE_SPLITS = ("train", "val", "test", "all")


def validate_sample_split(sample_split):
    if sample_split not in BRC_SAMPLE_SPLITS:
        raise ValueError(
            "sample_split must be one of {}".format(
                ", ".join(BRC_SAMPLE_SPLITS)
            )
        )


class BRCDatasetAdapter(ABC):
    """Interface between a dataset representation and generic BRC logic."""

    @property
    @abstractmethod
    def table_sizes(self):
        """Return the effective embedding-row count of every table."""

    @property
    def num_tables(self):
        return len(self.table_sizes)

    @abstractmethod
    def iter_categorical_chunks(self, sample_split):
        """
        Yield effective table-local lookup IDs.

        Every yielded array must have shape:

            (num_samples, num_tables)

        and contain the IDs actually supplied to embedding lookup.
        """

    def profile_frequencies(self, sample_split):
        """
        Count row-access frequencies for an explicitly selected sample split.
        """
        validate_sample_split(sample_split)

        return count_table_frequencies(
            self.iter_categorical_chunks(sample_split),
            self.table_sizes,
        )
