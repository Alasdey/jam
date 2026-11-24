"""
Worker processes of the online loop: each makes batches (core/batch.py) on
its own CUDA context, its CuPy pool capped at StreamConfig.worker_memory_mb.

Processes, not threads: a batch spends most of its time in Python between
small kernel launches, and threads share one GIL (measured: 4 threads on 4
streams ran 4x slower than one). Processes time-slice the GPU, so one's
kernels run while another is in Python.

Jobs ("create" then "test" of one batch, on the same worker) go to a worker
through its own queue; results come back through one shared queue, in
whatever order they finish (the coordinator inserts batches in order).
"""

import multiprocessing as mp
import queue
import traceback

from config import RunConfig


def _worker_main(wid: int, cfg: RunConfig, jobs, results) -> None:
    try:
        from core.batch import build_batch_maker
        from core.device import cupy

        cupy().get_default_memory_pool().set_limit(size=cfg.stream.worker_memory_mb * 2**20)
        maker = build_batch_maker(cfg)
        results.put(("ready", wid, None))
        while True:
            job = jobs.get()
            if job is None:
                return
            step = maker.create if job["kind"] == "create" else maker.test
            results.put(("done", wid, step(job)))
    except BaseException:
        results.put(("error", wid, traceback.format_exc()))


class WorkerPool:
    def __init__(self, cfg: RunConfig):
        ctx = mp.get_context("spawn")  # CUDA cannot survive a fork
        n = cfg.stream.n_workers
        self.results = ctx.Queue()
        self.jobs = [ctx.Queue() for _ in range(n)]
        self.procs = [ctx.Process(target=_worker_main, args=(w, cfg, self.jobs[w], self.results), daemon=True,
                                  name=f"jam-worker-{w}") for w in range(n)]
        for p in self.procs:
            p.start()
        try:
            for _ in range(n):
                self._get()  # "ready"
        except BaseException:
            self.close()
            raise

    def submit(self, wid: int, job: dict) -> None:
        self.jobs[wid].put(job)

    def result(self) -> tuple[int, dict]:
        """(worker, result) of the next job to finish."""
        return self._get()

    def _get(self):
        while True:
            try:
                kind, wid, payload = self.results.get(timeout=5)
            except queue.Empty:
                dead = [p.name for p in self.procs if not p.is_alive()]
                if dead:
                    raise RuntimeError(f"worker process(es) {dead} died without reporting an error "
                                       "(killed? out of host memory?)")
                continue
            if kind == "error":
                raise RuntimeError(f"worker {wid} failed:\n{payload}")
            return wid, payload

    def close(self) -> None:
        for q, p in zip(self.jobs, self.procs):
            if p.is_alive():
                q.put(None)
        for p in self.procs:
            p.join(timeout=10)
            if p.is_alive():
                p.terminate()
                p.join()

    def __enter__(self) -> "WorkerPool":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
