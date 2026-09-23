from __future__ import annotations

import os
import h5py
import numpy as np
import torch

from torch.utils.data import  Dataset


class SafetyValueDataset(Dataset):
    def __init__(self, dataset_path: str, split: str) -> None:
        self.dataset_path = os.path.abspath(dataset_path)
        self.split = split

        states_list = []
        g_values_list = []
        future_max_list = []
        current_indices = []
        next_indices = []
        state_offset = 0

        with h5py.File(self.dataset_path, "r") as file:
            split_group = file[split]
            for segment_key in sorted(split_group.keys()):
                group = split_group[segment_key]
                states = np.asarray(group["states"], dtype=np.float32)
                g_values = np.asarray(group["g_values"], dtype=np.float32)
                future_max_g = np.asarray(group["future_max_g"], dtype=np.float32)

                length = len(states)
                if length < 2:
                    continue

                states_list.append(states)
                g_values_list.append(g_values)
                future_max_list.append(future_max_g)

                indices = np.arange(state_offset, state_offset + length - 1, dtype=np.int64)
                current_indices.append(indices)
                next_indices.append(indices + 1)
                state_offset += length

        self.states = torch.from_numpy(np.concatenate(states_list, axis=0))
        self.g_values = torch.from_numpy(np.concatenate(g_values_list, axis=0))
        self.future_max_g = torch.from_numpy(np.concatenate(future_max_list, axis=0))
        self.current_indices = torch.from_numpy(np.concatenate(current_indices))
        self.next_indices = torch.from_numpy(np.concatenate(next_indices))

    def __len__(self) -> int:
        return len(self.current_indices)

    def __getitem__(self, index: int):
        current_index = self.current_indices[index]
        next_index = self.next_indices[index]
        return self.states[current_index], self.states[next_index], self.g_values[current_index], self.future_max_g[current_index]