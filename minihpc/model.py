"""Core data model: jobs, workers and the cluster they form.

A :class:`Job` is the unit of work the scheduler reasons about. A
:class:`Worker` is a compute node with a fixed number of cores and a fixed
amount of memory. A :class:`Cluster` is just a pool of workers plus the
book-keeping needed to know, at any instant, which jobs are running where and
how much capacity is still free.

Everything here is plain data and pure helpers -- no scheduling decisions and
no notion of time live in this module. That keeps the model reusable across
every policy in :mod:`minihpc.policies` and the discrete-event engine in
:mod:`minihpc.simulator`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional


@dataclass
class Job:
    """A single unit of work to be scheduled and run.

    Attributes:
        job_id: Stable identifier, unique within a workload.
        arrival: Tick at which the job becomes eligible to be scheduled.
        runtime: How long the job occupies a worker once started (in ticks).
        cores: Number of CPU cores the job needs while running.
        memory: Amount of memory the job needs while running (arbitrary units).
        priority: Lower number == more important. Used by the priority policy.
        payload: Optional callable for the real executor; ignored by the
            simulator. Lets the same Job feed both the sim and the pool.
    """

    job_id: str
    arrival: int = 0
    runtime: int = 1
    cores: int = 1
    memory: int = 1
    priority: int = 0
    payload: Optional[Callable[[], object]] = None

    def __post_init__(self) -> None:
        if self.runtime <= 0:
            raise ValueError(f"job {self.job_id}: runtime must be > 0")
        if self.cores <= 0:
            raise ValueError(f"job {self.job_id}: cores must be > 0")
        if self.memory <= 0:
            raise ValueError(f"job {self.job_id}: memory must be > 0")
        if self.arrival < 0:
            raise ValueError(f"job {self.job_id}: arrival must be >= 0")


@dataclass
class Worker:
    """A compute node with a fixed core/memory capacity.

    ``free_cores``/``free_memory`` track what is currently available; the
    capacity fields never change. A worker may run several jobs at once as long
    as their combined demand fits.
    """

    worker_id: int
    cores: int
    memory: int
    free_cores: int = field(init=False)
    free_memory: int = field(init=False)

    def __post_init__(self) -> None:
        self.free_cores = self.cores
        self.free_memory = self.memory

    def can_fit(self, job: Job) -> bool:
        return job.cores <= self.free_cores and job.memory <= self.free_memory

    def reserve(self, job: Job) -> None:
        if not self.can_fit(job):
            raise ValueError(
                f"job {job.job_id} does not fit on worker {self.worker_id}"
            )
        self.free_cores -= job.cores
        self.free_memory -= job.memory

    def release(self, job: Job) -> None:
        self.free_cores += job.cores
        self.free_memory += job.memory
        if self.free_cores > self.cores or self.free_memory > self.memory:
            raise ValueError(
                f"releasing job {job.job_id} over-frees worker {self.worker_id}"
            )


@dataclass
class Cluster:
    """A pool of homogeneous-or-not workers.

    The cluster owns the workers and offers a small amount of aggregate
    book-keeping (total/free capacity) used by the metrics layer.
    """

    workers: list[Worker]

    @classmethod
    def homogeneous(cls, n_workers: int, cores: int, memory: int) -> "Cluster":
        """Build a cluster of ``n_workers`` identical workers."""
        if n_workers <= 0:
            raise ValueError("n_workers must be > 0")
        return cls([Worker(i, cores, memory) for i in range(n_workers)])

    @property
    def total_cores(self) -> int:
        return sum(w.cores for w in self.workers)

    @property
    def total_memory(self) -> int:
        return sum(w.memory for w in self.workers)

    @property
    def free_cores(self) -> int:
        return sum(w.free_cores for w in self.workers)

    @property
    def used_cores(self) -> int:
        return self.total_cores - self.free_cores

    def first_fit(self, job: Job) -> Optional[Worker]:
        """Return the lowest-id worker that can currently host ``job``."""
        for w in self.workers:
            if w.can_fit(job):
                return w
        return None
