"""One process-wide lock between CUDA graph captures and other GPU work.

Why
---
``torch.cuda.graph`` captures in ``"global"`` mode. While one thread records a
graph, another thread that allocates, frees or synchronises on the device
invalidates the capture, and an eager ``torch.multinomial`` elsewhere trips
the CUDA generator's ``capturing`` flag ("Offset increment outside graph
capture encountered unexpectedly."). If ``cudaStreamEndCapture`` then fails,
PyTorch never clears that flag: every later sampling call and graph replay in
the process raises the same error until the process exits.

v3 Turbo captures lazily (a new batch bucket, cache length or sampling setting
mid-request), from whichever thread needs the graph, while other threads may
be prefilling, replaying, decoding audio or encoding a reference clip.

What
----
``CAPTURE_LOCK`` is a readers-writer lock. A capture (building a
``FusedFrame``/``StreamFrame``/``CudaGraphedFrame``, or dropping graphs) holds
it ``exclusive()``; every other piece of v3 Turbo GPU work holds it
``shared()``. Shared holders run side by side as before; a capture waits until
they are done, and while it waits new shared holders queue behind it, so a
busy stream worker cannot starve it.

Both modes are re-entrant per thread. A thread that holds it shared and asks
for ``exclusive()`` lets go of its shared hold while it waits and gets it back
afterwards (it does no GPU work in between), so that cannot deadlock either.

Rules for callers: keep shared sections to GPU work — never wait in one for
another thread (a queue, an event, another lock that a capturing thread may
hold), and never hold one across a ``yield``. The places in VieNeu that wait
for another thread (a batch engine's ``_fused_lock``, the stream scheduler's
lock and audio queues) call ``ensure_not_held`` first, so code that calls
VieNeu from inside its own ``shared()`` block gets an error at once instead of
a hang the first time a capture lands at the wrong moment. Pure
``threading``: importing this module does not pull torch.
"""
from __future__ import annotations

import functools
import threading
from contextlib import contextmanager
from typing import Dict, Iterator, Optional

__all__ = ["CaptureLock", "CAPTURE_LOCK", "gpu_work"]


class CaptureLock:
    """Readers-writer lock: ``exclusive()`` for captures, ``shared()`` for the rest."""

    def __init__(self) -> None:
        self._cond = threading.Condition(threading.Lock())
        self._readers: Dict[int, int] = {}      # thread id -> shared depth
        self._writer: Optional[int] = None      # thread id of the capturing thread
        self._writer_depth = 0
        self._writers_waiting = 0

    def held(self) -> Optional[str]:
        """How the calling thread holds the lock: ``"exclusive"``, ``"shared"``
        or ``None``. For assertions and tests."""
        me = threading.get_ident()
        with self._cond:
            if self._writer == me:
                return "exclusive"
            return "shared" if me in self._readers else None

    def ensure_not_held(self, what: str) -> None:
        """Raise if the calling thread holds the lock. For the places that wait
        for another thread, which may be one that needs a capture: holding the
        lock there can hang the process."""
        if self.held() is not None:
            raise RuntimeError(
                f"{what} was called while this thread holds CAPTURE_LOCK. Put only "
                "your own CUDA calls inside CAPTURE_LOCK.shared(): calling VieNeu or "
                "waiting for another thread in there can hang the process.")

    @contextmanager
    def shared(self) -> Iterator[None]:
        """GPU work that may run alongside other GPU work, but not a capture."""
        me = threading.get_ident()
        with self._cond:
            if self._writer != me and me not in self._readers:
                while self._writer is not None or self._writers_waiting:
                    self._cond.wait()
            self._readers[me] = self._readers.get(me, 0) + 1
        try:
            yield
        finally:
            with self._cond:
                depth = self._readers[me] - 1
                if depth:
                    self._readers[me] = depth
                else:
                    del self._readers[me]
                    self._cond.notify_all()

    @contextmanager
    def exclusive(self) -> Iterator[None]:
        """A capture: waits for all other GPU work to finish and keeps it out."""
        me = threading.get_ident()
        held = 0
        with self._cond:
            if self._writer == me:
                self._writer_depth += 1
            else:
                # Our own shared hold would make us wait for ourselves.
                held = self._readers.pop(me, 0)
                if held:
                    self._cond.notify_all()
                self._writers_waiting += 1
                try:
                    while self._writer is not None or self._readers:
                        self._cond.wait()
                except BaseException:
                    if held:
                        while self._writer is not None:
                            self._cond.wait()
                        self._readers[me] = held
                    raise
                finally:
                    self._writers_waiting -= 1
                    self._cond.notify_all()
                self._writer, self._writer_depth = me, 1
        try:
            yield
        finally:
            with self._cond:
                self._writer_depth -= 1
                if not self._writer_depth:
                    self._writer = None
                    if held:
                        self._readers[me] = held
                    self._cond.notify_all()


CAPTURE_LOCK = CaptureLock()


def gpu_work(fn):
    """Run ``fn`` holding ``CAPTURE_LOCK`` shared. For plain functions only:
    a generator would hold it only while it is created."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        with CAPTURE_LOCK.shared():
            return fn(*args, **kwargs)
    return wrapper
