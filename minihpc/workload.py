"""Load a workload from a simple file and build a few canned workloads.

The file format is one job per line, whitespace- or comma-separated columns,
``#`` comments allowed:

    # job_id arrival runtime cores memory priority
    j1 0 5 2 4 1
    j2 0 2 1 2 0

Only ``job_id`` is required; the rest fall back to Job's defaults. A header row
naming the columns may be given to reorder them.
"""

from __future__ import annotations

from pathlib import Path

from .model import Job

_FIELDS = ("job_id", "arrival", "runtime", "cores", "memory", "priority")
_INT_FIELDS = ("arrival", "runtime", "cores", "memory", "priority")


def parse_workload(text: str) -> list[Job]:
    """Parse workload text into a list of Jobs."""
    rows = []
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        rows.append(line)
    if not rows:
        return []

    # Detect an optional header naming the columns.
    header = None
    first = _split(rows[0])
    if first and first[0].lower() == "job_id":
        header = [c.lower() for c in first]
        rows = rows[1:]

    jobs: list[Job] = []
    for line in rows:
        cols = _split(line)
        names = header or list(_FIELDS)
        data = dict(zip(names, cols))
        kwargs = {"job_id": data["job_id"]}
        for f in _INT_FIELDS:
            if f in data and data[f] != "":
                kwargs[f] = int(data[f])
        jobs.append(Job(**kwargs))
    return jobs


def _split(line: str) -> list[str]:
    """Split a row on commas if present, else on any run of whitespace."""
    if "," in line:
        return [c.strip() for c in line.split(",")]
    return line.split()


def load_workload(path: str | Path) -> list[Job]:
    return parse_workload(Path(path).read_text(encoding="utf-8"))


def demo_workload() -> list[Job]:
    """A fixed, hand-tuned workload used by the demo and the docs.

    Mixes short and long jobs, varied priorities and resource demands so the
    policies visibly diverge: SJF should beat FIFO on average wait, priority
    should reorder by importance, and backfill should lift utilisation.
    """
    return [
        #    id     arr run cores mem prio
        Job("big1",   0, 8, 4, 6, priority=2),
        Job("short1", 0, 1, 1, 2, priority=1),
        Job("short2", 0, 2, 1, 2, priority=1),
        Job("med1",   0, 4, 2, 4, priority=0),
        Job("short3", 1, 1, 1, 2, priority=1),
        Job("med2",   2, 3, 2, 4, priority=0),
        Job("big2",   2, 6, 3, 5, priority=2),
        Job("short4", 3, 2, 1, 2, priority=1),
    ]
