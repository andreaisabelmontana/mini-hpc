"""miniHPC: a small job scheduler + cluster simulator.

Public surface:

    Job, Worker, Cluster              -- the model
    FIFO, SJF, Priority, RoundRobin,
    Backfill, make_policy             -- scheduling policies
    simulate, Schedule, Metrics       -- the discrete-event engine
    compare, Comparison               -- run a workload under every policy
    run_pool                          -- real multiprocessing executor
    load_workload, demo_workload      -- workload I/O
"""

from .model import Cluster, Job, Worker
from .policies import (
    Backfill,
    FIFO,
    Policy,
    Priority,
    RoundRobin,
    SJF,
    make_policy,
)
from .simulator import JobRecord, Metrics, Schedule, simulate
from .compare import Comparison, compare
from .executor import RunResult, run_pool
from .workload import demo_workload, load_workload, parse_workload

__all__ = [
    "Cluster",
    "Job",
    "Worker",
    "FIFO",
    "SJF",
    "Priority",
    "RoundRobin",
    "Backfill",
    "Policy",
    "make_policy",
    "simulate",
    "Schedule",
    "Metrics",
    "JobRecord",
    "compare",
    "Comparison",
    "run_pool",
    "RunResult",
    "load_workload",
    "parse_workload",
    "demo_workload",
]

__version__ = "1.0.0"
