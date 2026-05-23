from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Any, Iterator


class SamplerTiming:
    def __init__(
        self,
        *,
        enabled: bool,
        backend: str,
        device: Any,
        batch_size: int = 1,
        cuda_events: bool = True,
        sync_cuda_for_wall: bool = False,
        record_chunk_timings: bool = False,
    ) -> None:
        self.enabled = bool(enabled)
        self.backend = str(backend)
        self.device = str(device)
        self.batch_size = int(batch_size)
        self.sync_cuda_for_wall = bool(sync_cuda_for_wall)
        self.record_chunk_timings = bool(record_chunk_timings)
        self._start_perf = time.perf_counter()
        self._wall: dict[str, float] = {}
        self._cuda_ms: dict[str, float] = {}
        self._counts: dict[str, int] = {
            "scalar_sync_count": 0,
            "full_sync_count": 0,
            "num_md_steps_completed": 0,
            "num_md_chunks": 0,
        }
        self._chunk_timings: list[dict[str, Any]] = []
        self._last_chunk_wall: dict[str, float] = {}
        self._last_chunk_cuda_s: dict[str, float] = {}
        self._torch = None
        self._cuda_enabled = False
        self._gpu_name = None
        if self.enabled and cuda_events and self.device.startswith("cuda"):
            try:
                import torch

                self._torch = torch
                self._cuda_enabled = bool(torch.cuda.is_available()) and hasattr(torch.cuda, "Event")
                if self._cuda_enabled:
                    try:
                        self._gpu_name = str(torch.cuda.get_device_name(torch.cuda.current_device()))
                    except Exception:
                        self._gpu_name = None
            except Exception:
                self._torch = None
                self._cuda_enabled = False

    @classmethod
    def from_sampler_config(
        cls,
        sampler_config: dict[str, Any],
        *,
        backend: str,
        device: Any,
        batch_size: int = 1,
    ) -> "SamplerTiming":
        raw = dict(sampler_config.get("timing") or {})
        return cls(
            enabled=bool(raw.get("enabled", False)),
            backend=backend,
            device=device,
            batch_size=int(batch_size),
            cuda_events=bool(raw.get("cuda_events", True)),
            sync_cuda_for_wall=bool(raw.get("sync_cuda_for_wall", False)),
            record_chunk_timings=bool(raw.get("record_chunk_timings", False)),
        )

    def increment(self, name: str, amount: int = 1) -> None:
        if not self.enabled:
            return
        self._counts[name] = int(self._counts.get(name, 0)) + int(amount)

    def set_count(self, name: str, value: int) -> None:
        if not self.enabled:
            return
        self._counts[name] = int(value)

    @contextmanager
    def scope(self, name: str, *, cuda: bool = False) -> Iterator[None]:
        if not self.enabled:
            yield
            return

        torch = self._torch
        use_cuda = bool(cuda and self._cuda_enabled and torch is not None)
        if use_cuda and self.sync_cuda_for_wall:
            torch.cuda.synchronize()
        start_wall = time.perf_counter()
        start_event = end_event = None
        if use_cuda:
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
        try:
            yield
        finally:
            if use_cuda and end_event is not None and start_event is not None:
                end_event.record()
                end_event.synchronize()
                self._cuda_ms[name] = self._cuda_ms.get(name, 0.0) + float(start_event.elapsed_time(end_event))
            elif use_cuda and self.sync_cuda_for_wall:
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start_wall
            self._wall[name] = self._wall.get(name, 0.0) + float(elapsed)

    def record_chunk(self, *, step: int, chunk_steps: int) -> None:
        if not (self.enabled and self.record_chunk_timings):
            return
        wall_snapshot = {
            "md_run": float(self._wall.get("md_run", 0.0)),
            "model_eval": float(self._wall.get("model_eval", 0.0)),
            "host_sync": float(self._wall.get("host_sync", 0.0)),
        }
        cuda_snapshot = {
            "md_run": float(self._cuda_ms.get("md_run", 0.0) / 1000.0),
            "model_eval": float(self._cuda_ms.get("model_eval", 0.0) / 1000.0),
        }
        self._chunk_timings.append(
            {
                "step": int(step),
                "chunk_steps": int(chunk_steps),
                "md_run_wall_s": wall_snapshot["md_run"] - self._last_chunk_wall.get("md_run", 0.0),
                "md_run_cuda_s": cuda_snapshot["md_run"] - self._last_chunk_cuda_s.get("md_run", 0.0),
                "model_eval_wall_s": wall_snapshot["model_eval"] - self._last_chunk_wall.get("model_eval", 0.0),
                "model_eval_cuda_s": cuda_snapshot["model_eval"] - self._last_chunk_cuda_s.get("model_eval", 0.0),
                "host_sync_wall_s": wall_snapshot["host_sync"] - self._last_chunk_wall.get("host_sync", 0.0),
            }
        )
        self._last_chunk_wall = wall_snapshot
        self._last_chunk_cuda_s = cuda_snapshot

    def metadata(self, *, total_wall_s: float | None = None, num_atoms: int | None = None) -> dict[str, Any] | None:
        if not self.enabled:
            return None
        total = float(total_wall_s if total_wall_s is not None else time.perf_counter() - self._start_perf)
        md_wall = float(self._wall.get("md_run", 0.0))
        md_cuda = float(self._cuda_ms.get("md_run", 0.0) / 1000.0)
        steps = int(self._counts.get("num_md_steps_completed", 0))
        batch_size = max(1, int(self.batch_size))
        atom_steps = steps * batch_size * int(num_atoms or 0)
        payload = {
            "backend": self.backend,
            "device": self.device,
            "gpu_name": self._gpu_name,
            "batch_size": batch_size,
            "total_wall_s": total,
            "md_run_wall_s": md_wall,
            "md_run_cuda_s": md_cuda,
            "model_eval_wall_s": float(self._wall.get("model_eval", 0.0)),
            "model_eval_cuda_s": float(self._cuda_ms.get("model_eval", 0.0) / 1000.0),
            "host_sync_wall_s": float(self._wall.get("host_sync", 0.0)),
            "metrics_wall_s": float(self._wall.get("metrics", 0.0)),
            "trajectory_io_wall_s": float(self._wall.get("trajectory_io", 0.0)),
            "num_md_steps_completed": steps,
            "num_md_chunks": int(self._counts.get("num_md_chunks", 0)),
            "steps_per_total_wall_s": float(steps / total) if total > 0.0 else None,
            "steps_per_md_wall_s": float(steps / md_wall) if md_wall > 0.0 else None,
            "steps_per_md_cuda_s": float(steps / md_cuda) if md_cuda > 0.0 else None,
            "systems_per_second": float((steps * batch_size) / md_wall) if md_wall > 0.0 else None,
            "atom_steps_per_second": float(atom_steps / md_wall) if md_wall > 0.0 and num_atoms else None,
            "scalar_sync_count": int(self._counts.get("scalar_sync_count", 0)),
            "full_sync_count": int(self._counts.get("full_sync_count", 0)),
            "cuda_events_enabled": bool(self._cuda_enabled),
            "record_chunk_timings": bool(self.record_chunk_timings),
        }
        if self.record_chunk_timings:
            payload["chunk_timings"] = list(self._chunk_timings)
        return payload
