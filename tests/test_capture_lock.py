"""CAPTURE_LOCK (vieneu/_v3_turbo_engine/capture_lock.py): a CUDA graph capture
runs alone, other GPU work runs side by side. Pure threading, no torch.

Every wait here has a deadline, so a regression fails instead of hanging.
"""
import sys
import threading
import time
from types import SimpleNamespace

import pytest

from vieneu._v3_turbo_engine.capture_lock import CAPTURE_LOCK, CaptureLock, gpu_work

WAIT = 5.0


def _start(fn, *args):
    t = threading.Thread(target=fn, args=args, daemon=True)
    t.start()
    return t


def _join(*threads):
    for t in threads:
        t.join(WAIT)
        assert not t.is_alive(), "deadlock: a thread never finished"


def _capture(lock, log=None):
    with lock.exclusive():
        if log is not None:
            log.append("capture")


def _until(cond):
    deadline = time.monotonic() + WAIT
    while not cond():
        assert time.monotonic() < deadline, "condition never became true"
        time.sleep(0.001)


def test_shared_holders_run_side_by_side():
    lock = CaptureLock()
    both_inside = threading.Barrier(2, timeout=WAIT)

    def work():
        with lock.shared():
            both_inside.wait()      # raises BrokenBarrierError if they were serialised

    _join(_start(work), _start(work))


def test_a_capture_waits_for_running_work_and_holds_new_work_back():
    lock = CaptureLock()
    log = []
    running, finish = threading.Event(), threading.Event()

    def running_work():
        with lock.shared():
            running.set()
            finish.wait(WAIT)
            log.append("work done")

    def late_work():
        with lock.shared():
            log.append("late work")

    a = _start(running_work)
    assert running.wait(WAIT)
    b = _start(_capture, lock, log)
    _until(lambda: lock._writers_waiting == 1)
    c = _start(late_work)            # arrives while the capture waits: must queue behind it
    assert log == []
    finish.set()
    _join(a, b, c)
    assert log == ["work done", "capture", "late work"]


def test_captures_do_not_overlap():
    lock = CaptureLock()
    inside, peak = [0], [0]
    guard = threading.Lock()

    def capture():
        with lock.exclusive():
            with guard:
                inside[0] += 1
                peak[0] = max(peak[0], inside[0])
            time.sleep(0.005)
            with guard:
                inside[0] -= 1

    _join(*[_start(capture) for _ in range(6)])
    assert peak[0] == 1


def test_both_modes_are_reentrant():
    lock = CaptureLock()
    seen = []

    def nested():
        with lock.shared():
            with lock.shared():
                seen.append(lock.held())
        with lock.exclusive():
            with lock.exclusive():
                with lock.shared():
                    seen.append(lock.held())
        seen.append(lock.held())

    _join(_start(nested))
    assert seen == ["shared", "exclusive", None]


def test_reentry_is_not_held_back_by_a_waiting_capture():
    # A shared holder that takes the lock again (a helper that also locks)
    # must not queue behind a capture that is waiting for it to finish.
    lock = CaptureLock()
    holding, go_on = threading.Event(), threading.Event()
    done = []

    def work():
        with lock.shared():
            holding.set()
            go_on.wait(WAIT)
            with lock.shared():
                done.append("inner")

    a = _start(work)
    assert holding.wait(WAIT)
    b = _start(_capture, lock)
    _until(lambda: lock._writers_waiting == 1)
    go_on.set()
    _join(a, b)
    assert done == ["inner"]


def test_a_shared_holder_can_capture_and_keeps_its_hold():
    lock = CaptureLock()
    other_in, other_out = threading.Event(), threading.Event()
    states = []

    def other_work():
        with lock.shared():
            other_in.set()
            other_out.wait(WAIT)

    def work_that_captures():
        with lock.shared():
            with lock.exclusive():       # waits for other_work only, not for itself
                states.append(lock.held())
            states.append(lock.held())

    a = _start(other_work)
    assert other_in.wait(WAIT)
    b = _start(work_that_captures)
    _until(lambda: lock._writers_waiting == 1)
    assert states == []                  # still waiting for other_work
    other_out.set()
    _join(a, b)
    assert states == ["exclusive", "shared"]


def test_gpu_work_holds_the_process_lock_shared():
    @gpu_work
    def job(x):
        return x, CAPTURE_LOCK.held()

    assert job(3) == (3, "shared")
    assert job.__name__ == "job"
    assert CAPTURE_LOCK.held() is None


def test_an_error_inside_releases_the_lock():
    lock = CaptureLock()
    for mode in (lock.shared, lock.exclusive):
        with pytest.raises(RuntimeError):
            with mode():
                raise RuntimeError("capture failed")
        assert lock.held() is None
    _join(_start(_capture, lock))      # nothing left behind: another thread gets it


def test_ensure_not_held_refuses_either_mode():
    # Where VieNeu waits for another thread it checks first: holding the lock
    # there would hang the process the first time a capture lands meanwhile.
    lock = CaptureLock()
    lock.ensure_not_held("infer")          # not held: nothing happens
    for mode in (lock.shared, lock.exclusive):
        with mode():
            with pytest.raises(RuntimeError, match="infer was called while this thread holds CAPTURE_LOCK"):
                lock.ensure_not_held("infer")
    assert lock.held() is None

    seen = []
    with lock.shared():                    # another thread's hold is not ours
        t = _start(lambda: seen.append(lock.ensure_not_held("infer")))
        _join(t)
    assert seen == [None]


def test_web_ui_gpu_cleanup_waits_for_captures(monkeypatch):
    # The Gradio app (vieneu-web) frees the CUDA cache and syncs after every
    # job, while another tab may be capturing a graph: that must hold the lock.
    from apps import ui_utils

    seen = []
    note = lambda what: lambda: seen.append((what, CAPTURE_LOCK.held()))   # noqa: E731
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: True, empty_cache=note("empty_cache"),
                             synchronize=note("synchronize")),
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False)),
        xpu=SimpleNamespace(is_available=lambda: False),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    ui_utils.cleanup_gpu_memory()
    assert seen == [("empty_cache", "shared"), ("synchronize", "shared")]
    assert CAPTURE_LOCK.held() is None
