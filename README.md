# miniHPC

A small job scheduler and cluster simulator in Python. Submit a workload, run
it under a scheduling policy, and read back the schedule plus cluster metrics —
makespan, throughput, average wait, average turnaround and utilisation. There's
also a real multiprocessing executor that actually runs job payloads, so it's
more than a sim.

- **Live page:** https://andreaisabelmontana.github.io/mini-hpc/
- Pure standard library for the engine; `matplotlib` only for the optional Gantt chart.

## The model

- **Job** — `(id, arrival, runtime, cores, memory, priority, payload)`. Lower
  `priority` value means more important. `payload` is an optional callable used
  by the real executor.
- **Worker** — a compute node with fixed `cores` and `memory`. It can host
  several jobs at once as long as their combined demand fits; `reserve`/`release`
  make over-allocation impossible.
- **Cluster** — a pool of workers, with aggregate capacity book-keeping.

The model is pure data — no time, no scheduling decisions — so it's shared by
every policy and by the simulator.

## Scheduling policies

Each policy decides the *order* in which ready jobs are considered; the
simulator then places each job on the first worker it fits.

- **FIFO** — order by arrival (ties by id). Simple and fair in submission order,
  but a large job at the head blocks everyone behind it (head-of-line blocking).
- **SJF** (Shortest-Job-First) — order by runtime. Minimises average wait among
  non-preemptive policies, at the risk of starving long jobs.
- **Priority** — order by `priority` value, most-important first.
- **Round-Robin** — rotate which ready job is considered first each tick, so no
  single job sits at the head of the queue forever.
- **Backfill** — priority order, but smaller jobs may jump ahead to fill idle
  capacity that strict head-of-line ordering would waste (the EASY-backfill
  idea real schedulers like Slurm use). Trades a little fairness for higher
  utilisation.

## The simulator

A discrete-event engine over an integer tick clock. Each tick it: releases
finished jobs, admits new arrivals, asks the policy to order the ready queue and
places jobs first-fit, then advances the clock to the next arrival or completion
(no empty ticks). It records each job's start/finish and computes:

- **makespan** — first start to last finish
- **throughput** — jobs per tick over the makespan
- **avg wait** — mean time queued before starting
- **avg turnaround** — mean time from arrival to completion
- **utilisation** — core-ticks used / core-ticks available over the makespan

Guaranteed invariants (checked by the tests): a job never starts before it
arrives; no worker is ever over-allocated; a job that can never fit any worker
raises rather than stalling silently.

## Policy comparison (real numbers)

`demo.py` runs a fixed 8-job workload on **1 worker × 4 cores / 8 memory** under
every policy. The tight capacity is what makes the policies diverge:

```
policy    makespan throughput avg_wait avg_turn util%
-------------------------------------------------------
FIFO      21       0.381      9.12     12.50    83.3
SJF       19       0.421      1.88     5.25     92.1
Priority  20       0.400      2.38     5.75     87.5
RoundRobin18       0.444      8.62     12.00    97.2
Backfill  19       0.421      1.88     5.25     92.1
```

What this shows, on this workload:

- **SJF slashes average wait** — 1.88 vs FIFO's 9.12 — because FIFO's big head
  job (`big1`, 8 ticks) blocks the queue while SJF clears the short jobs first.
- **FIFO leaves capacity idle** (83.3% util) where backfill recovers it (92.1%)
  by letting short jobs slip into the gaps the big job leaves.
- **Round-Robin maxes utilisation** (97.2%) but keeps a high average wait
  (8.62): cycling the head keeps the cluster busy but doesn't favour short jobs.
- **Priority** reorders by importance, landing between FIFO and SJF here.

(SJF avg wait ≤ FIFO avg wait is asserted as a test on this workload.)

## Run it

```bash
pip install -r requirements.txt        # matplotlib + pytest; engine needs neither

python demo.py                         # comparison table + gantt_fifo/sjf.png

# Run one workload under one policy:
python -m minihpc run example_workload.txt --policy sjf --workers 1 --cores 4 --memory 8
# Compare every policy on a workload:
python -m minihpc compare example_workload.txt --workers 1 --cores 4 --memory 8 --gantt out.png
```

Workload files are one job per line, `# comments` allowed, whitespace- or
comma-separated columns `job_id arrival runtime cores memory priority` (only
`job_id` is required). See `example_workload.txt`.

### Real execution

`minihpc.run_pool` dispatches jobs whose `payload` is a callable across a
`ProcessPoolExecutor`, capped at the cluster's total core count, in the policy's
order — the same scheduling decision the simulator models, but actually run.

## Tests

```bash
python -m pytest -q
```

```
..................                                                       [100%]
18 passed in 0.28s
```

Tests verify exact schedules against hand computation: FIFO orders by arrival
and SJF by runtime (exact order + average wait), priority respects priority,
makespan and utilisation match hand-computed values, no worker is ever
over-allocated, and SJF average wait ≤ FIFO average wait on the workload.

## Layout

```
minihpc/
  model.py       Job, Worker, Cluster
  policies.py    FIFO, SJF, Priority, RoundRobin, Backfill
  simulator.py   discrete-event engine + metrics
  compare.py     run a workload under every policy
  executor.py    real multiprocessing executor
  plot.py        Gantt chart (matplotlib)
  workload.py    workload file loader + the demo workload
  cli.py         `python -m minihpc run|compare`
demo.py          fixed workload under all policies + Gantt
tests/           pytest suite
```
