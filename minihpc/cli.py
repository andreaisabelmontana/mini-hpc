"""Command-line interface: submit a workload and run it under a policy.

Examples:

    python -m minihpc run workload.txt --policy sjf --workers 2 --cores 4 --memory 8
    python -m minihpc compare workload.txt --workers 2 --cores 4 --memory 8 --gantt out.png
    python -m minihpc run workload.txt --policy fifo --gantt fifo.png
"""

from __future__ import annotations

import argparse
import sys

from .compare import compare
from .model import Cluster
from .policies import make_policy
from .simulator import simulate
from .workload import demo_workload, load_workload


def _add_cluster_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--workers", type=int, default=2, help="number of workers")
    p.add_argument("--cores", type=int, default=4, help="cores per worker")
    p.add_argument("--memory", type=int, default=8, help="memory per worker")


def _load(path: str):
    if path == "-" or path == "demo":
        return demo_workload()
    return load_workload(path)


def _print_schedule(sched) -> None:
    print(f"policy: {sched.policy}")
    print(f"{'job':10}{'worker':8}{'start':7}{'finish':8}{'wait':6}{'turn':6}")
    print("-" * 45)
    for r in sorted(sched.records, key=lambda r: (r.start, r.worker_id)):
        print(
            f"{r.job.job_id:10}{r.worker_id:<8}{r.start:<7}{r.finish:<8}"
            f"{r.wait:<6}{r.turnaround:<6}"
        )
    m = sched.metrics
    print("-" * 45)
    print(f"makespan={m.makespan}  throughput={m.throughput:.3f}  "
          f"avg_wait={m.avg_wait:.2f}  avg_turnaround={m.avg_turnaround:.2f}  "
          f"utilisation={m.utilisation * 100:.1f}%")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="minihpc", description="job scheduler + cluster simulator"
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run", help="run a workload under one policy")
    p_run.add_argument("workload", help="workload file, or 'demo'")
    p_run.add_argument("--policy", default="fifo",
                       help="fifo|sjf|priority|rr|backfill")
    _add_cluster_args(p_run)
    p_run.add_argument("--gantt", metavar="PNG", help="write a Gantt chart")

    p_cmp = sub.add_parser("compare", help="run under every policy")
    p_cmp.add_argument("workload", help="workload file, or 'demo'")
    _add_cluster_args(p_cmp)
    p_cmp.add_argument("--gantt", metavar="PNG",
                       help="write a Gantt chart for the FIFO run")

    args = parser.parse_args(argv)
    jobs = _load(args.workload)
    if not jobs:
        print("no jobs in workload", file=sys.stderr)
        return 1

    if args.cmd == "run":
        cluster = Cluster.homogeneous(args.workers, args.cores, args.memory)
        sched = simulate(jobs, cluster, make_policy(args.policy))
        _print_schedule(sched)
        if args.gantt:
            from .plot import gantt
            gantt(sched, args.gantt)
            print(f"wrote {args.gantt}")
        return 0

    if args.cmd == "compare":
        cmp = compare(jobs, args.workers, args.cores, args.memory)
        print(cmp.table())
        if args.gantt:
            from .plot import gantt
            gantt(cmp.schedules["FIFO"], args.gantt)
            print(f"wrote {args.gantt}")
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
