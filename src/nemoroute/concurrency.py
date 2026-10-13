from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

from nemoroute.budget import BudgetExceeded


@dataclass
class TaskReport:
    done: int = 0
    failed: int = 0
    stopped_by_budget: bool = False
    errors: list[str] = field(default_factory=list)


def run_tasks(
    tasks: Iterable[Any],
    worker: Callable[[Any], Any],
    on_result: Callable[[Any, Any], None],
    concurrency: int,
    describe: Callable[[Any], str] = str,
) -> TaskReport:
    report = TaskReport()
    pool = ThreadPoolExecutor(max_workers=concurrency)
    try:
        futures = {pool.submit(worker, task): task for task in tasks}
        for future in as_completed(futures):
            task = futures[future]
            if future.cancelled():
                continue
            try:
                result = future.result()
            except BudgetExceeded as exc:
                if not report.stopped_by_budget:
                    print(f"Stopping: {exc}")
                report.stopped_by_budget = True
                for f in futures:
                    f.cancel()
                continue
            except Exception as exc:
                report.failed += 1
                message = f"{describe(task)}: {type(exc).__name__}: {exc}"
                report.errors.append(message[:300])
                print(f"  failed {message[:300]}")
                continue
            on_result(task, result)
            report.done += 1
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
    return report
