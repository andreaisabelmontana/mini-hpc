"""Tests for the real multiprocessing executor and the workload loader."""

from __future__ import annotations

from minihpc import Cluster, FIFO, Job, parse_workload, run_pool


class Square:
    """Top-level, picklable callable so the process pool can ship it across.

    Lambdas and local functions cannot be pickled (which the 'spawn' start
    method on Windows/macOS requires), so real payloads must be module-level.
    """

    def __init__(self, n: int) -> None:
        self.n = n

    def __call__(self) -> int:
        return self.n * self.n


def test_run_pool_executes_payloads():
    jobs = [
        Job(f"sq{i}", payload=Square(i), cores=1, memory=1)
        for i in range(6)
    ]
    cluster = Cluster.homogeneous(2, cores=2, memory=2)
    results = run_pool(jobs, cluster, FIFO())
    values = {r.job_id: r.value for r in results}
    assert values == {f"sq{i}": i * i for i in range(6)}


def test_run_pool_skips_jobs_without_payload():
    jobs = [Job("noop", cores=1, memory=1)]  # no payload
    results = run_pool(jobs, Cluster.homogeneous(1, 1, 1), FIFO())
    assert results == []


def test_parse_workload_with_header():
    text = """
    # comment
    job_id arrival runtime cores memory priority
    a 0 5 2 4 1
    b 1 2 1 2 0
    """
    jobs = parse_workload(text)
    assert [j.job_id for j in jobs] == ["a", "b"]
    a, b = jobs
    assert (a.arrival, a.runtime, a.cores, a.memory, a.priority) == (0, 5, 2, 4, 1)
    assert (b.arrival, b.runtime, b.cores, b.memory, b.priority) == (1, 2, 1, 2, 0)


def test_parse_workload_csv_and_defaults():
    jobs = parse_workload("x,0,3\ny,2,1")
    assert [j.job_id for j in jobs] == ["x", "y"]
    # Unspecified columns fall back to Job defaults.
    assert jobs[0].cores == 1 and jobs[0].memory == 1
