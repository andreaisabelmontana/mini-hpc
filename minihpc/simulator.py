"""Discrete-event simulator that runs a workload under a policy.

The model of time is a simple integer tick clock. On each tick the simulator:

1. **Releases** every job whose finish tick has arrived, freeing its worker's
   cores and memory.
2. **Admits** newly arrived jobs into the ready queue.
3. **Schedules**: asks the policy to order the ready queue, then walks that
   order placing each job on the first worker it fits. Strict policies stop at
   the first job that cannot be placed (head-of-line blocking); policies whose
   ``allows_skip()`` is true may look past a blocked job to backfill idle
   capacity.
4. **Advances** the clock to the next interesting tick (the soonest of: the
   next job arrival, or the next running-job completion) so we never spin
   through empty ticks.

The result is a :class:`Schedule` -- per-job start/finish records plus the
cluster :class:`Metrics` (makespan, throughput, average wait, average
turnaround, utilisation) computed from them.

Invariants the engine guarantees and the tests check:

* A job never starts before it arrives.
* At no instant does the set of running jobs exceed any worker's capacity --
  ``reserve``/``release`` make over-allocation impossible.
* Utilisation is core-seconds actually used divided by core-seconds available
  over the makespan, so it is always in [0, 1].
"""

from __future__ import annotations

from dataclasses import dataclass

from .model import Cluster, Job
from .policies import Policy


@dataclass
class JobRecord:
    """The realised timing of one job after a simulation run."""

    job: Job
    worker_id: int
    start: int
    finish: int

    @property
    def wait(self) -> int:
        """Ticks spent in the queue before starting."""
        return self.start - self.job.arrival

    @property
    def turnaround(self) -> int:
        """Total time from arrival to completion."""
        return self.finish - self.job.arrival


@dataclass
class Metrics:
    """Aggregate cluster metrics for a completed run."""

    makespan: int
    throughput: float
    avg_wait: float
    avg_turnaround: float
    utilisation: float
    n_jobs: int


@dataclass
class Schedule:
    """Full result of a simulation: per-job records plus aggregate metrics."""

    policy: str
    records: list[JobRecord]
    metrics: Metrics

    def by_job_id(self) -> dict[str, JobRecord]:
        return {r.job.job_id: r for r in self.records}

    def start_order(self) -> list[str]:
        """Job ids ordered by (start, worker_id, job_id) -- the run order."""
        ordered = sorted(
            self.records, key=lambda r: (r.start, r.worker_id, r.job.job_id)
        )
        return [r.job.job_id for r in ordered]


def _compute_metrics(records: list[JobRecord], cluster: Cluster) -> Metrics:
    if not records:
        return Metrics(0, 0.0, 0.0, 0.0, 0.0, 0)
    first_start = min(r.start for r in records)
    makespan = max(r.finish for r in records) - first_start
    n = len(records)
    avg_wait = sum(r.wait for r in records) / n
    avg_turn = sum(r.turnaround for r in records) / n
    throughput = n / makespan if makespan > 0 else float(n)
    # Utilisation: core-ticks used / core-ticks available over the makespan.
    used = sum(r.job.cores * r.job.runtime for r in records)
    available = cluster.total_cores * makespan
    utilisation = used / available if available > 0 else 0.0
    return Metrics(
        makespan=makespan,
        throughput=throughput,
        avg_wait=avg_wait,
        avg_turnaround=avg_turn,
        utilisation=utilisation,
        n_jobs=n,
    )


def simulate(
    jobs: list[Job],
    cluster: Cluster,
    policy: Policy,
    max_ticks: int = 1_000_000,
) -> Schedule:
    """Run ``jobs`` on ``cluster`` under ``policy`` and return the schedule.

    The cluster is mutated during the run (workers are reserved/released); pass
    a fresh cluster per policy if you are comparing policies on one workload.

    Raises:
        ValueError: if a job can never fit on any worker's full capacity, since
            it would block the queue forever. Validating up front turns a silent
            stall into a clear, early error.
    """
    for job in jobs:
        if not any(
            job.cores <= w.cores and job.memory <= w.memory
            for w in cluster.workers
        ):
            raise ValueError(
                f"job {job.job_id} (cores={job.cores}, memory={job.memory}) "
                f"exceeds every worker's capacity and can never be scheduled"
            )

    pending = sorted(jobs, key=lambda j: (j.arrival, j.job_id))
    ready: list[Job] = []
    running: list[tuple[int, Job, object]] = []  # (finish_tick, job, worker)
    records: list[JobRecord] = []

    now = 0
    idx = 0  # next index into pending not yet admitted
    ticks = 0

    while (idx < len(pending) or ready or running) and ticks < max_ticks:
        ticks += 1

        # 1. Release finished jobs.
        still_running = []
        for finish, job, worker in running:
            if finish <= now:
                worker.release(job)
            else:
                still_running.append((finish, job, worker))
        running = still_running

        # 2. Admit arrivals.
        while idx < len(pending) and pending[idx].arrival <= now:
            ready.append(pending[idx])
            idx += 1

        # 3. Schedule.
        if ready:
            ordered = policy.order(list(ready), now)
            placed: set[str] = set()
            for job in ordered:
                worker = cluster.first_fit(job)
                if worker is None:
                    if policy.allows_skip():
                        continue  # backfill: try the next candidate
                    break  # head-of-line blocking
                worker.reserve(job)
                finish = now + job.runtime
                running.append((finish, job, worker))
                records.append(JobRecord(job, worker.worker_id, now, finish))
                placed.add(job.job_id)
            if placed:
                ready = [j for j in ready if j.job_id not in placed]

        # 4. Advance the clock to the next interesting tick: the soonest of a
        #    running job finishing or the next job arriving. Jobs left in the
        #    ready queue (blocked / waiting on capacity) are reconsidered once
        #    one of those events frees a worker.
        candidates = []
        if running:
            candidates.append(min(f for f, _, _ in running))
        if idx < len(pending):
            candidates.append(pending[idx].arrival)
        if not candidates:
            # Nothing running and nothing left to arrive. If the ready queue is
            # non-empty here, those jobs never fit on any worker -- a malformed
            # workload -- so stop rather than spin forever.
            break
        nxt = min(candidates)
        now = nxt if nxt > now else now + 1

    metrics = _compute_metrics(records, cluster)
    return Schedule(policy=policy.name, records=records, metrics=metrics)
