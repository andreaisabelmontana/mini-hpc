"""Render a schedule as a Gantt chart (matplotlib).

Each row is a worker; each bar is a job, positioned by its start/finish ticks.
Importing matplotlib lazily keeps the rest of the package dependency-free.
"""

from __future__ import annotations

from .simulator import Schedule


def gantt(schedule: Schedule, path: str, title: str | None = None) -> str:
    """Write a Gantt PNG for ``schedule`` to ``path`` and return the path."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    records = schedule.records
    worker_ids = sorted({r.worker_id for r in records})
    row = {w: i for i, w in enumerate(worker_ids)}

    fig, ax = plt.subplots(figsize=(9, 0.7 * len(worker_ids) + 1.5))
    cmap = plt.get_cmap("tab20")
    for k, r in enumerate(sorted(records, key=lambda r: r.start)):
        ax.barh(
            row[r.worker_id],
            r.finish - r.start,
            left=r.start,
            height=0.6,
            color=cmap(k % 20),
            edgecolor="white",
        )
        ax.text(
            (r.start + r.finish) / 2,
            row[r.worker_id],
            r.job.job_id,
            va="center",
            ha="center",
            fontsize=8,
            color="black",
        )

    ax.set_yticks(list(row.values()))
    ax.set_yticklabels([f"worker {w}" for w in worker_ids])
    ax.set_xlabel("tick")
    ax.set_title(title or f"miniHPC schedule — {schedule.policy}")
    ax.grid(axis="x", linestyle=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path
