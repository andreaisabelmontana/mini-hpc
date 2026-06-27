"""Run one workload under every policy and tabulate the results.

This is the layer that turns the simulator into a teaching tool: the same job
set is run under each policy on a *fresh* cluster, and the metrics are laid out
side by side so the trade-offs are obvious (e.g. SJF's lower average wait,
backfill's higher utilisation).
"""

from __future__ import annotations

from dataclasses import dataclass

from .model import Cluster, Job
from .policies import Backfill, FIFO, SJF, Priority, RoundRobin
from .simulator import Schedule, simulate

DEFAULT_POLICIES = (FIFO, SJF, Priority, RoundRobin, Backfill)


@dataclass
class Comparison:
    schedules: dict[str, Schedule]

    def table(self) -> str:
        cols = ("policy", "makespan", "throughput", "avg_wait",
                "avg_turn", "util%")
        widths = (10, 9, 11, 9, 9, 7)
        head = "".join(c.ljust(w) for c, w in zip(cols, widths))
        lines = [head, "-" * sum(widths)]
        for name, sched in self.schedules.items():
            m = sched.metrics
            row = [
                name,
                str(m.makespan),
                f"{m.throughput:.3f}",
                f"{m.avg_wait:.2f}",
                f"{m.avg_turnaround:.2f}",
                f"{m.utilisation * 100:.1f}",
            ]
            lines.append("".join(c.ljust(w) for c, w in zip(row, widths)))
        return "\n".join(lines)


def compare(
    jobs: list[Job],
    n_workers: int,
    cores: int,
    memory: int,
    policies=DEFAULT_POLICIES,
) -> Comparison:
    """Run every policy on its own fresh cluster and collect the schedules."""
    out: dict[str, Schedule] = {}
    for pol_cls in policies:
        cluster = Cluster.homogeneous(n_workers, cores, memory)
        policy = pol_cls()
        out[policy.name] = simulate(jobs, cluster, policy)
    return Comparison(out)
