"""Tests against hand-computed schedules.

Each test fixes a tiny workload whose schedule can be worked out by hand, then
asserts the engine reproduces the exact order, waits, makespan and utilisation.
"""

from __future__ import annotations

import pytest

from minihpc import (
    Backfill,
    Cluster,
    FIFO,
    Job,
    Priority,
    RoundRobin,
    SJF,
    compare,
    demo_workload,
    simulate,
)


# --- A serial workload: one worker, capacity 1 core / 1 memory ---------------
# Jobs all arrive at t=0, so a single worker runs them strictly one at a time.
#   A: runtime 3, priority 2
#   B: runtime 1, priority 0
#   C: runtime 2, priority 1
def serial_jobs() -> list[Job]:
    return [
        Job("A", arrival=0, runtime=3, cores=1, memory=1, priority=2),
        Job("B", arrival=0, runtime=1, cores=1, memory=1, priority=0),
        Job("C", arrival=0, runtime=2, cores=1, memory=1, priority=1),
    ]


def serial_cluster() -> Cluster:
    return Cluster.homogeneous(1, cores=1, memory=1)


def test_fifo_orders_by_arrival_then_id():
    # All arrive at 0, so FIFO breaks ties by job_id: A, B, C.
    sched = simulate(serial_jobs(), serial_cluster(), FIFO())
    assert sched.start_order() == ["A", "B", "C"]
    r = sched.by_job_id()
    # Serial execution: A[0,3], B[3,4], C[4,6].
    assert (r["A"].start, r["A"].finish) == (0, 3)
    assert (r["B"].start, r["B"].finish) == (3, 4)
    assert (r["C"].start, r["C"].finish) == (4, 6)
    # Waits: A=0, B=3, C=4 -> average 7/3.
    assert sched.metrics.avg_wait == pytest.approx(7 / 3)


def test_sjf_orders_by_runtime():
    # Shortest first: B(1), C(2), A(3).
    sched = simulate(serial_jobs(), serial_cluster(), SJF())
    assert sched.start_order() == ["B", "C", "A"]
    r = sched.by_job_id()
    assert (r["B"].start, r["B"].finish) == (0, 1)
    assert (r["C"].start, r["C"].finish) == (1, 3)
    assert (r["A"].start, r["A"].finish) == (3, 6)
    # Waits: B=0, C=1, A=3 -> average 4/3.
    assert sched.metrics.avg_wait == pytest.approx(4 / 3)


def test_priority_respects_priority():
    # priority value: B=0, C=1, A=2 -> B, C, A.
    sched = simulate(serial_jobs(), serial_cluster(), Priority())
    assert sched.start_order() == ["B", "C", "A"]
    r = sched.by_job_id()
    # The most important job (B) starts first; the least (A) starts last.
    assert r["B"].start == 0
    assert r["A"].start == max(rec.start for rec in sched.records)


def test_priority_distinct_from_fifo_and_sjf():
    # A workload where priority order differs from both arrival and runtime.
    jobs = [
        Job("low", arrival=0, runtime=1, cores=1, memory=1, priority=5),
        Job("hi", arrival=0, runtime=4, cores=1, memory=1, priority=0),
    ]
    sched = simulate(jobs, Cluster.homogeneous(1, 1, 1), Priority())
    # Despite being the longer job and tied on arrival, 'hi' runs first.
    assert sched.start_order() == ["hi", "low"]


# --- Makespan / utilisation against hand computation -------------------------
def test_capacity_concurrency_and_metrics():
    # One worker, 2 cores. Three 1-core, runtime-2 jobs arriving at 0.
    # x and y run 0-2 together; z runs 2-4. Never more than 2 cores in use.
    jobs = [Job(j, arrival=0, runtime=2, cores=1, memory=1) for j in "xyz"]
    sched = simulate(jobs, Cluster.homogeneous(1, cores=2, memory=2), FIFO())
    r = sched.by_job_id()
    assert (r["x"].start, r["x"].finish) == (0, 2)
    assert (r["y"].start, r["y"].finish) == (0, 2)
    assert (r["z"].start, r["z"].finish) == (2, 4)
    assert sched.metrics.makespan == 4
    # used core-ticks = 3 jobs * 1 core * 2 ticks = 6; available = 2 * 4 = 8.
    assert sched.metrics.utilisation == pytest.approx(6 / 8)


def test_resource_limits_never_exceeded():
    # Reconstruct, for every tick, the set of running jobs from the records and
    # assert no worker ever hosts more than its capacity.
    jobs = demo_workload()
    cluster = Cluster.homogeneous(2, cores=4, memory=8)
    sched = simulate(jobs, cluster, FIFO())
    records = sched.records
    events = sorted({r.start for r in records} | {r.finish for r in records})
    for t in events:
        per_worker_cores: dict[int, int] = {}
        per_worker_mem: dict[int, int] = {}
        for r in records:
            if r.start <= t < r.finish:  # running at instant t
                per_worker_cores[r.worker_id] = (
                    per_worker_cores.get(r.worker_id, 0) + r.job.cores
                )
                per_worker_mem[r.worker_id] = (
                    per_worker_mem.get(r.worker_id, 0) + r.job.memory
                )
        for w in cluster.workers:
            assert per_worker_cores.get(w.worker_id, 0) <= w.cores
            assert per_worker_mem.get(w.worker_id, 0) <= w.memory


def test_concurrent_jobs_never_exceed_worker_count():
    # Across the whole demo run, the number of simultaneously running jobs must
    # never need more workers than exist (a coarser capacity check).
    jobs = demo_workload()
    n_workers = 2
    cluster = Cluster.homogeneous(n_workers, cores=4, memory=8)
    sched = simulate(jobs, cluster, FIFO())
    records = sched.records
    events = sorted({r.start for r in records})
    for t in events:
        busy_workers = {r.worker_id for r in records if r.start <= t < r.finish}
        assert len(busy_workers) <= n_workers


# --- The headline property: SJF minimises average wait -----------------------
def test_sjf_avg_wait_le_fifo_on_serial():
    fifo = simulate(serial_jobs(), serial_cluster(), FIFO())
    sjf = simulate(serial_jobs(), serial_cluster(), SJF())
    assert sjf.metrics.avg_wait <= fifo.metrics.avg_wait
    # And strictly better here.
    assert sjf.metrics.avg_wait < fifo.metrics.avg_wait


def test_sjf_avg_wait_le_fifo_on_demo_workload():
    cmp = compare(demo_workload(), n_workers=1, cores=4, memory=8)
    assert cmp.schedules["SJF"].metrics.avg_wait <= cmp.schedules["FIFO"].metrics.avg_wait


# --- Sanity / invariants -----------------------------------------------------
def test_no_job_starts_before_arrival():
    jobs = [
        Job("late", arrival=5, runtime=2, cores=1, memory=1),
        Job("early", arrival=0, runtime=2, cores=1, memory=1),
    ]
    sched = simulate(jobs, Cluster.homogeneous(1, 1, 1), FIFO())
    for r in sched.records:
        assert r.start >= r.job.arrival


def test_all_jobs_complete():
    for policy in (FIFO(), SJF(), Priority(), RoundRobin(), Backfill()):
        cluster = Cluster.homogeneous(2, 4, 8)
        sched = simulate(demo_workload(), cluster, policy)
        assert len(sched.records) == len(demo_workload())


def test_utilisation_in_unit_interval():
    for policy in (FIFO(), SJF(), Priority(), RoundRobin(), Backfill()):
        cluster = Cluster.homogeneous(1, 4, 8)
        sched = simulate(demo_workload(), cluster, policy)
        assert 0.0 <= sched.metrics.utilisation <= 1.0


def test_unschedulable_job_raises():
    # A job needing more cores than any worker has must be rejected, not stall.
    jobs = [Job("toobig", arrival=0, runtime=1, cores=8, memory=1)]
    with pytest.raises(ValueError):
        simulate(jobs, Cluster.homogeneous(1, cores=4, memory=8), FIFO())


def test_backfill_utilisation_ge_fifo():
    # Backfilling should never waste more capacity than strict FIFO ordering.
    cmp = compare(demo_workload(), n_workers=1, cores=4, memory=8)
    assert (
        cmp.schedules["Backfill"].metrics.utilisation
        >= cmp.schedules["FIFO"].metrics.utilisation
    )
