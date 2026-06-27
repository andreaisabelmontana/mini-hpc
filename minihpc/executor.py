"""A real executor: run job payloads across a multiprocessing worker pool.

This makes miniHPC more than a simulation. Jobs whose ``payload`` is a callable
are dispatched to a pool of OS processes, respecting an aggregate core budget
(the total cores across the cluster) so we never run more concurrent work than
the cluster could hold. The dispatch order is decided by the same scheduling
policies used by the simulator, so the executor and the sim agree on *which*
job runs next -- the sim just models the timing instead of doing the work.

The result is a list of :class:`RunResult` carrying each job's return value and
wall-clock duration. This is intentionally simple: it is a faithful "the
scheduler really launched these" demonstration, not a production batch system.
"""

from __future__ import annotations

import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Optional

from .model import Cluster, Job
from .policies import Policy


@dataclass
class RunResult:
    job_id: str
    value: object
    duration: float


def _invoke(payload) -> object:
    return payload()


def run_pool(
    jobs: list[Job],
    cluster: Cluster,
    policy: Policy,
    max_workers: Optional[int] = None,
) -> list[RunResult]:
    """Execute every job's payload across a process pool.

    Jobs are ordered by ``policy`` (using now=0, the order at submission) and
    dispatched. Concurrency is capped at the cluster's total core count unless
    ``max_workers`` overrides it, so the pool never exceeds cluster capacity.
    """
    runnable = [j for j in jobs if j.payload is not None]
    if not runnable:
        return []
    ordered = policy.order(list(runnable), 0)
    cap = max_workers or max(1, cluster.total_cores)

    results: dict[str, RunResult] = {}
    with ProcessPoolExecutor(max_workers=cap) as pool:
        futures = {}
        for job in ordered:
            start = time.perf_counter()
            fut = pool.submit(_invoke, job.payload)
            futures[fut] = (job.job_id, start)
        for fut in as_completed(futures):
            job_id, start = futures[fut]
            value = fut.result()
            results[job_id] = RunResult(job_id, value, time.perf_counter() - start)

    # Preserve the policy's dispatch order in the returned list.
    return [results[j.job_id] for j in ordered if j.job_id in results]
