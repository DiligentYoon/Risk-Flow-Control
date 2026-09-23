from __future__ import annotations

import json
import os
from typing import Any

import h5py
import torch


class SafetyRolloutCollector:
    """Collect batched safety rollouts and export complete episodes.

    The collector receives post-physics, pre-reset quantities from a vectorized
    environment. Each parallel environment owns one active episode buffer.
    When an environment terminates or times out, its trajectory is written to
    HDF5 and the corresponding buffer is reused for the next episode.

    """

    def __init__(
        self,
        file_path: str,
        num_envs: int,
        safety_state_dim: int,
        max_episode_length: int,
        metadata: dict[str, Any] | None = None,
        compression: str | None = "lzf",
    ) -> None:
        self.num_envs = num_envs
        self.safety_state_dim = safety_state_dim
        self.max_episode_length = max_episode_length
        self.compression = compression

        if file_path is not None:
            self.file_path = os.path.abspath(file_path)
            os.makedirs(os.path.dirname(self.file_path), exist_ok=True)

            self.file = h5py.File(self.file_path, "w")
            self.data_group = self.file.create_group("data")
            self._write_metadata(metadata or {})
        else:
            self.file = None
            self.file_path = None
            self.data_group = None

        # CPU buffers for currently active episodes.
        self.safety_state_buf = torch.empty((self.num_envs, self.max_episode_length, self.safety_state_dim), dtype=torch.float32, device="cpu")
        self.safety_value_buf = torch.empty((self.num_envs, self.max_episode_length), dtype=torch.float32, device="cpu")
        self.push_event_buf = torch.empty((self.num_envs, self.max_episode_length), dtype=torch.bool, device="cpu")
        self.terminated_buf = torch.empty((self.num_envs, self.max_episode_length), dtype=torch.bool, device="cpu")
        self.truncated_buf = torch.empty((self.num_envs, self.max_episode_length), dtype=torch.bool, device="cpu")

        # Current trajectory length of each parallel environment.
        self.episode_lengths = torch.zeros(self.num_envs, dtype=torch.long, device="cpu")

        self.episode_count = 0
        self.sample_count = 0
        self.closed = False

    def append(
        self,
        final_safety_states: torch.Tensor,
        final_safety_values: torch.Tensor,
        final_push_events: torch.Tensor,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
    ) -> int:
        """Append one vectorized environment step.

        Args:
            final_safety_states:
                Post-physics, pre-reset safety states. Shape: [E, D].
            final_safety_values:
                Safety values corresponding to final_safety_states.
                Shape: [E] or [E, 1].
            final_push_events:
                Push-segment marker associated with each recorded state.
                Shape: [E] or [E, 1].
            terminated:
                True terminal flags. Shape: [E] or [E, 1].
            truncated:
                Time-limit truncation flags. Shape: [E] or [E, 1].

        Returns:
            Number of complete episodes written during this append.
        """
        self._check_open()

        states = final_safety_states.detach().to(device="cpu", dtype=torch.float32)
        values = final_safety_values.detach().reshape(-1).to(device="cpu", dtype=torch.float32)
        push_events = final_push_events.detach().reshape(-1).to(device="cpu", dtype=torch.bool)
        terminated = terminated.detach().reshape(-1).to(device="cpu", dtype=torch.bool)
        truncated = truncated.detach().reshape(-1).to(device="cpu", dtype=torch.bool)

        self._validate_batch(states, values, push_events, terminated, truncated)

        if torch.any(self.episode_lengths >= self.max_episode_length):
            env_ids = torch.nonzero(self.episode_lengths >= self.max_episode_length, as_tuple=False).squeeze(-1)
            raise RuntimeError(f"Episode buffer overflow for env IDs: {env_ids.tolist()}")

        env_ids  = torch.arange(self.num_envs)
        step_ids = self.episode_lengths

        self.safety_state_buf[env_ids, step_ids] = states
        self.safety_value_buf[env_ids, step_ids] = values
        self.push_event_buf[env_ids, step_ids] = push_events
        self.terminated_buf[env_ids, step_ids] = terminated
        self.truncated_buf[env_ids, step_ids] = truncated

        self.episode_lengths += 1
        self.sample_count += self.num_envs

        done = terminated | truncated
        done_env_ids = torch.nonzero(done, as_tuple=False).squeeze(-1)

        # done env extraction
        if self.file is not None:
            num_written = 0
            for env_id in done_env_ids.tolist():
                self._write_episode(env_id, partial=False)
                # step index reset
                self.episode_lengths[env_id] = 0
                num_written += 1
        else:
            num_written = len(done_env_ids.tolist())

        return num_written

    def close(self, save_partial: bool = False) -> None:
        """Finalize and close the dataset.

        Args:
            save_partial:
                If True, trajectories that have not naturally terminated when
                collection stops are also written with ``partial=True``.
                The default is False because such trajectories are right-censored
                and are inconvenient for future-maximum evaluation.
        """
        if self.closed:
            return

        if save_partial:
            active_env_ids = torch.nonzero(self.episode_lengths > 0, as_tuple=False).squeeze(-1)

            for env_id in active_env_ids.tolist():
                self._write_episode(env_id, partial=True)

        if self.file is not None:
            self.file.attrs["num_episodes"] = self.episode_count
            self.file.attrs["num_samples_received"] = self.sample_count

            # HDF5 flush => file write
            self.file.flush()
            self.file.close()

        self.closed = True

    def _write_episode(self, env_id: int, partial: bool) -> None:
        # write data of ended episodes
        length = int(self.episode_lengths[env_id].item())

        if length <= 0:
            return

        episode_name = f"episode_{self.episode_count:08d}"
        group = self.data_group.create_group(episode_name)

        # save until ended index
        group.create_dataset("safety_states", data=self.safety_state_buf[env_id, :length].numpy(), compression=self.compression)
        group.create_dataset("safety_values", data=self.safety_value_buf[env_id, :length].numpy(), compression=self.compression)
        group.create_dataset("push_events", data=self.push_event_buf[env_id, :length].numpy(),  compression=self.compression)
        group.create_dataset("terminated", data=self.terminated_buf[env_id, :length].numpy(), compression=self.compression,)
        group.create_dataset("truncated", data=self.truncated_buf[env_id, :length].numpy(), compression=self.compression)

        group.attrs["env_id"] = env_id
        group.attrs["length"] = length
        group.attrs["partial"] = partial

        self.episode_count += 1

    def _validate_batch(
        self,
        states: torch.Tensor,
        values: torch.Tensor,
        push_events: torch.Tensor,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
    ) -> None:
        expected_state_shape = (self.num_envs, self.safety_state_dim)

        if states.shape != expected_state_shape:
            raise ValueError(f"Invalid safety-state shape: expected {expected_state_shape}, got {tuple(states.shape)}")

        for name, tensor in (("final_safety_values", values), ("final_push_events", push_events), ("terminated", terminated), ("truncated", truncated)):
            if tensor.shape != (self.num_envs,):
                raise ValueError(f"Invalid {name} shape: expected {(self.num_envs,)}, got {tuple(tensor.shape)}")

    def _write_metadata(
        self,
        metadata: dict[str, Any],
    ) -> None:
        self.file.attrs["num_envs"] = self.num_envs
        self.file.attrs["safety_state_dim"] = self.safety_state_dim
        self.file.attrs["max_episode_length"] = self.max_episode_length

        for key, value in metadata.items():
            if isinstance(value, (str, int, float, bool)):
                self.file.attrs[key] = value
            else:
                self.file.attrs[key] = json.dumps(value)

    def _check_open(self) -> None:
        if self.closed:
            raise RuntimeError("Collector is already closed.")

    def __enter__(self) -> SafetyRolloutCollector:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close(save_partial=False)