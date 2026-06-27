"""Scheduling policies.

A policy decides the *order* in which ready jobs are considered for placement.
The simulator (see :mod:`minihpc.simulator`) walks that order and places each
job on the first worker it fits, subject to the policy's blocking rule:

* Most policies are **strict ordering** policies: they hand back the ready
  queue in some order, and the simulator stops at the first job that cannot be
  placed (head-of-line blocking). FIFO, SJF and Priority all work this way --
  the difference is purely the sort key.
* **Round-robin** rotates which job is considered first each tick so no single
  job monopolises the head of the queue.
* **Backfill** relaxes the head-of-line rule: it keeps the priority order for
  the head job but lets smaller jobs jump ahead *as long as they do not delay
  the head job's reservation* -- the classic EASY-backfill idea.

Each policy is a small object with two methods so the simulator can treat them
uniformly:

    order(ready, now)   -> list[Job]   ordered candidates for this tick
    allows_skip()       -> bool        may the sim look past a blocked head?

Keeping the policies this thin makes them trivial to unit-test against a
hand-computed schedule.
"""

from __future__ import annotations

from typing import Protocol

from .model import Cluster, Job


class Policy(Protocol):
    name: str

    def order(self, ready: list[Job], now: int) -> list[Job]: ...

    def allows_skip(self) -> bool: ...


class FIFO:
    """First-in, first-out: order strictly by arrival, ties by job_id.

    The canonical baseline. Simple and fair in submission order, but a single
    large job at the head blocks everyone behind it (head-of-line blocking).
    """

    name = "FIFO"

    def order(self, ready: list[Job], now: int) -> list[Job]:
        return sorted(ready, key=lambda j: (j.arrival, j.job_id))

    def allows_skip(self) -> bool:
        return False


class SJF:
    """Shortest-Job-First: order by runtime, ties by arrival then job_id.

    Provably minimises average wait time among non-preemptive policies when
    all jobs are present, at the cost of potentially starving long jobs.
    """

    name = "SJF"

    def order(self, ready: list[Job], now: int) -> list[Job]:
        return sorted(ready, key=lambda j: (j.runtime, j.arrival, j.job_id))

    def allows_skip(self) -> bool:
        return False


class Priority:
    """Priority scheduling: lower ``priority`` value runs first.

    Ties broken by arrival then job_id so the order is fully deterministic.
    """

    name = "Priority"

    def order(self, ready: list[Job], now: int) -> list[Job]:
        return sorted(ready, key=lambda j: (j.priority, j.arrival, j.job_id))

    def allows_skip(self) -> bool:
        return False


class RoundRobin:
    """Round-robin: rotate which ready job is considered first each tick.

    Pure ordering policy with no per-job preference -- it just keeps cycling
    the head of the queue so no single job sits at the front forever. With
    multi-tick runtimes this behaves like a fair, starvation-resistant FIFO.
    """

    name = "RoundRobin"

    def __init__(self) -> None:
        self._offset = 0

    def order(self, ready: list[Job], now: int) -> list[Job]:
        base = sorted(ready, key=lambda j: (j.arrival, j.job_id))
        if not base:
            return base
        k = self._offset % len(base)
        self._offset += 1
        return base[k:] + base[:k]

    def allows_skip(self) -> bool:
        # Rotating the head already prevents one job from blocking forever, so
        # we let the sim look past a job that doesn't fit this tick.
        return True


class Backfill:
    """Priority order with EASY backfilling.

    Jobs are considered in priority order (lower value first). If the head job
    cannot be placed this tick, smaller/lower jobs behind it are allowed to run
    *as long as they fit right now* -- this fills idle capacity that strict
    head-of-line ordering would waste, without changing the head job's turn to
    be next in line. It trades a little fairness for markedly higher
    utilisation, which is exactly why real schedulers (Slurm, PBS) ship it.
    """

    name = "Backfill"

    def order(self, ready: list[Job], now: int) -> list[Job]:
        return sorted(ready, key=lambda j: (j.priority, j.arrival, j.job_id))

    def allows_skip(self) -> bool:
        return True


# Registry used by the CLI / demo so policies can be selected by name.
REGISTRY: dict[str, type] = {
    "fifo": FIFO,
    "sjf": SJF,
    "priority": Priority,
    "rr": RoundRobin,
    "roundrobin": RoundRobin,
    "backfill": Backfill,
}


def make_policy(name: str) -> Policy:
    key = name.strip().lower()
    if key not in REGISTRY:
        raise KeyError(
            f"unknown policy {name!r}; choose from {sorted(set(REGISTRY))}"
        )
    return REGISTRY[key]()
