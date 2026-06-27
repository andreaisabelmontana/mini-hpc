"""Run the fixed demo workload under every policy and print the comparison.

Also writes a Gantt chart (gantt.png) of the FIFO and SJF schedules so the
head-of-line blocking that FIFO suffers is visible next to SJF's tighter pack.

Run:  python demo.py
"""

from __future__ import annotations

from minihpc import demo_workload
from minihpc.compare import compare
from minihpc.plot import gantt

# A single, capacity-constrained worker (4 cores, 8 memory). The contention is
# what makes the policies diverge: with plenty of slack every policy finishes
# at once and the comparison is boring.
N_WORKERS, CORES, MEMORY = 1, 4, 8


def main() -> None:
    jobs = demo_workload()
    print(f"workload: {len(jobs)} jobs")
    print(f"cluster:  {N_WORKERS} worker(s) x {CORES} cores / {MEMORY} memory\n")

    cmp = compare(jobs, N_WORKERS, CORES, MEMORY)
    print(cmp.table())

    print("\nrun order (start sequence) per policy:")
    for name, sched in cmp.schedules.items():
        print(f"  {name:11} {' '.join(sched.start_order())}")

    fifo = cmp.schedules["FIFO"]
    sjf = cmp.schedules["SJF"]
    gantt(fifo, "gantt_fifo.png", title="miniHPC — FIFO (head-of-line blocking)")
    gantt(sjf, "gantt_sjf.png", title="miniHPC — SJF (shortest first)")
    print("\nwrote gantt_fifo.png and gantt_sjf.png")

    fw, sw = fifo.metrics.avg_wait, sjf.metrics.avg_wait
    print(f"\nSJF avg wait {sw:.2f} <= FIFO avg wait {fw:.2f}: {sw <= fw}")


if __name__ == "__main__":
    main()
