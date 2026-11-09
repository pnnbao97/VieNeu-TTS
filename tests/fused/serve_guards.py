"""Failure handling of the GPU serving path, driven on the CPU with stand-ins.

The real ``V3TurboStreamScheduler`` / ``V3TurboBatchEngine`` code runs; only the
parts that need CUDA (the captured frames, the model) are replaced. Run through
tests/test_serve_guards.py in a fresh interpreter: other test files set
``sys.modules["torch"]`` to a MagicMock, which would turn the
``@torch.no_grad()`` methods under test into mocks.
"""
import math
import threading
import time
from types import SimpleNamespace

import pytest
import torch

from vieneu._v3_turbo_engine.capture_lock import CAPTURE_LOCK
from vieneu.v3_turbo_serve import engine as engine_mod
from vieneu.v3_turbo_serve import fused as fused_mod
from vieneu.v3_turbo_serve import stream as stream_mod


class _Any:
    """A tensor that is only sliced and passed along."""
    shape = (4, 8)

    def __getitem__(self, _):
        return self


# ── stream scheduler ─────────────────────────────────────────────────────────

class FakeFrame:
    """``StreamFrame`` without CUDA: takes rows, never finishes them."""

    def __init__(self, model, B, max_len, *, repetition_window):
        self.B = B
        self.bb = SimpleNamespace(layers=[0, 1])
        self.replay_error = None
        self.built_as = CAPTURE_LOCK.held()

    def admit(self, b, keys, values, h_last, spk, cap, **sampling):
        if spk == "bad-row":
            raise RuntimeError("row does not fit the frame")
        return 0

    def replay(self):
        if self.replay_error is not None:
            raise self.replay_error

    def status(self):
        return [0] * (2 * self.B)


class FakeBatchEngine:
    def __init__(self):
        self.config = SimpleNamespace(max_position_embeddings=1024)
        self.model = object()
        self.tts = SimpleNamespace(audio_tokenizer=SimpleNamespace(),
                                   _resolve_speaker_emb=lambda s: s)
        self.bb = SimpleNamespace(prefill=self.prefill)
        self.cache = SimpleNamespace(layers=[SimpleNamespace(keys=_Any(), values=_Any())] * 2)

    def _prompt_embeds(self, req):
        if req["phonemes"] == "bad-prompt":
            raise ValueError("ref_codes has the wrong shape")
        return _Any()

    def prefill(self, embeds):
        return [None] * len(embeds), self.cache, _Any(), None


class _NoThread:
    """The worker thread, never started: each test runs ``_admit``/``_run``
    itself, so nothing races the assertions."""

    def __init__(self, target=None, name=None, daemon=None):
        pass

    def start(self):
        pass

    def join(self, timeout=None):
        pass


@pytest.fixture
def make_sched(monkeypatch):
    monkeypatch.setattr(stream_mod, "StreamFrame", FakeFrame)
    monkeypatch.setattr(stream_mod, "threading", SimpleNamespace(
        Thread=_NoThread, Event=threading.Event, Condition=threading.Condition))

    def make(max_streams=2):
        return stream_mod.V3TurboStreamScheduler(FakeBatchEngine(), max_streams=max_streams)

    return make


def _req(**over):
    r = dict(phonemes="xin chào", speaker_emb="spk", ref_codes=None, use_ref_codes=True,
             temperature=0.8, top_k=25, top_p=0.95, repetition_penalty=1.2, max_new_frames=10)
    r.update(over)
    return r


def _queue(sched, **over):
    h = stream_mod.StreamHandle(sched, _req(**over))
    sched._pending.put(h)
    return h


def _answer(h):
    return h.q.get_nowait()


def test_bad_request_fails_alone(make_sched):
    sched = make_sched(max_streams=4)
    bad_prompt = _queue(sched, phonemes="bad-prompt")
    bad_row = _queue(sched, speaker_emb="bad-row")
    good = _queue(sched)
    sched._admit()
    assert isinstance(_answer(bad_prompt), ValueError)
    assert isinstance(_answer(bad_row), RuntimeError)
    assert good.slot is not None and sched._slots[good.slot] is good
    assert good.q.empty() and sched.error is None


def test_more_requests_than_free_slots_wait_their_turn(make_sched):
    # Used to take up to max_admit (8) requests and pop an empty free list:
    # IndexError, and the worker died for everyone.
    sched = make_sched(max_streams=1)
    first, second = _queue(sched), _queue(sched)
    sched._admit()
    assert sched._slots == [first]
    assert sched._pending.qsize() == 1 and second.slot is None
    assert sched.error is None


def test_close_tells_running_and_queued_requests(make_sched):
    # These used to wait forever on a queue no worker would fill again.
    sched = make_sched(max_streams=1)
    running, queued = _queue(sched), _queue(sched)
    sched._admit()
    sched.close()
    sched._run()                          # the worker's exit after close()
    for h in (running, queued):
        err = _answer(h)
        assert isinstance(err, RuntimeError) and "closed" in str(err)
    assert sched.error is None
    with pytest.raises(RuntimeError, match="closed"):
        sched.submit(**_req())


def test_worker_crash_reaches_every_waiter_and_blocks_new_work(make_sched):
    sched = make_sched(max_streams=2)
    sched.frame.replay_error = RuntimeError("CUDA error: an illegal memory access")
    h = sched.submit(**_req())
    sched._run()                          # admits h, then the first frame fails
    with pytest.raises(RuntimeError, match="illegal memory access"):
        list(h)
    assert isinstance(sched.error, RuntimeError)
    with pytest.raises(RuntimeError, match="down"):
        sched.submit(**_req())


def test_crash_while_admitting_reaches_requests_off_the_queue(make_sched):
    # Requests taken off the queue but not yet in a slot are known only to _admit.
    sched = make_sched(max_streams=2)

    class Broken:
        def __getitem__(self, _):
            raise RuntimeError("KV cache layout changed")

    sched.be.cache = SimpleNamespace(layers=[SimpleNamespace(keys=Broken(), values=Broken())] * 2)
    a, b = _queue(sched), _queue(sched)
    sched._run()
    for h in (a, b):
        assert "layout changed" in str(_answer(h))
    assert sched.error is not None


@pytest.mark.parametrize("bad", [
    dict(temperature=float("nan")), dict(temperature=-1.0), dict(top_p=float("inf")),
    dict(repetition_penalty=0.0), dict(top_k="many"),
])
def test_submit_rejects_bad_sampling_in_the_callers_thread(make_sched, bad):
    sched = make_sched()
    with pytest.raises(ValueError):
        sched.submit(**_req(**bad))
    assert sched._pending.empty() and sched.error is None


def test_int64_slot_overflow_is_not_a_value_error():
    # Why the old ``except ValueError`` around admit() could not contain it.
    t = torch.zeros(1, 1, dtype=torch.long)
    with pytest.raises((RuntimeError, OverflowError)):
        t[0] = 10 ** 30


def test_stream_frame_clamps_top_k_into_its_tensor():
    B = 2
    sf = stream_mod.StreamFrame.__new__(stream_mod.StreamFrame)
    sf.max_len, sf.vocab, sf.spk = 64, 1024, None
    sf.bb = SimpleNamespace(load_row=lambda b, k, v: None)
    sf.rep = SimpleNamespace(reset_rows=lambda rows: None)
    sf.h = torch.zeros(B, 4)
    sf.temperature, sf.top_p, sf.penalty = torch.ones(B, 1), torch.ones(B, 1), torch.ones(B, 1)
    sf.top_k = torch.full((B, 1), 25, dtype=torch.long)
    sf.frame = torch.zeros((), dtype=torch.long)
    sf.caps, sf.length = torch.zeros(B, dtype=torch.long), torch.zeros(B, dtype=torch.long)
    sf.finished = torch.ones(B, dtype=torch.bool)
    kv = [torch.zeros(1, 3, 2)]
    for b, k in ((0, 10 ** 30), (1, -5)):
        sf.admit(b, kv, kv, torch.zeros(4), None, 10, temperature=0.8, top_k=k,
                 top_p=0.95, repetition_penalty=1.2)
    assert sf.top_k.view(-1).tolist() == [1024, 0]


# ── batch engine ─────────────────────────────────────────────────────────────

class FakeFused:
    """``FusedFrame`` without CUDA; ``run`` reports how many runs overlap."""
    active = peak = 0
    guard = threading.Lock()
    built_as = []          # how CAPTURE_LOCK was held at each capture

    def __init__(self, model, B, max_len, max_frames, **sampling):
        self.B = B
        FakeFused.built_as.append(CAPTURE_LOCK.held())

    def run(self, h, cache, mask, pos, caps, spk, max_new_frames, on_frame=None):
        with FakeFused.guard:
            FakeFused.active += 1
            FakeFused.peak = max(FakeFused.peak, FakeFused.active)
        time.sleep(0.02)
        with FakeFused.guard:
            FakeFused.active -= 1
        return [f"codes-{b}" for b in range(self.B)]


def _engine(monkeypatch):
    monkeypatch.setattr(fused_mod, "FusedFrame", FakeFused)
    FakeFused.active = FakeFused.peak = 0
    FakeFused.built_as = []
    tts = SimpleNamespace(
        model=SimpleNamespace(semantic_backbone=None, config=SimpleNamespace(hidden_size=8)),
        config=SimpleNamespace(max_position_embeddings=4096),
    )
    eng = engine_mod.V3TurboBatchEngine(tts)
    eng.bb = SimpleNamespace(prefill=lambda embeds: (None, None, None, None))
    return eng


def _gen(eng, temperature=0.8, rows=1):
    return eng._generate_codes_fused(
        [_Any()] * rows, None, None, temperature=temperature, top_k=25, top_p=0.95,
        repetition_penalty=1.2, repetition_window=16, max_new_frames=10)


def test_fused_runs_do_not_overlap(monkeypatch):
    eng = _engine(monkeypatch)
    threads = [threading.Thread(target=_gen, args=(eng,)) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert FakeFused.peak == 1


def test_only_recent_sampling_settings_keep_their_graphs(monkeypatch):
    eng = _engine(monkeypatch)            # keeps 2 settings by default
    temps = lambda: sorted({k[2] for k in eng._fused})   # noqa: E731
    _gen(eng, 0.8)
    _gen(eng, 0.8, rows=4)                # same setting, another batch bucket
    _gen(eng, 0.5)
    assert temps() == [0.5, 0.8] and len(eng._fused) == 3
    _gen(eng, 0.3)                        # a third setting frees the oldest one
    assert temps() == [0.3, 0.5]
    _gen(eng, 0.5)
    _gen(eng, 0.9)                        # 0.5 was used again, so 0.3 goes
    assert temps() == [0.5, 0.9]


def test_graph_budget_comes_from_the_environment(monkeypatch):
    monkeypatch.setenv("VIENEU_FUSED_SAMPLINGS", "1")
    eng = _engine(monkeypatch)
    _gen(eng, 0.8)
    _gen(eng, 0.5)
    assert {k[2] for k in eng._fused} == {0.5}


def test_generate_batch_rejects_nan_before_any_gpu_work(monkeypatch):
    eng = _engine(monkeypatch)
    with pytest.raises(ValueError, match="temperature"):
        eng.generate_batch([{"phonemes": "a"}], temperature=math.nan)
    assert eng._fused == {}


# ── CAPTURE_LOCK: a capture never overlaps other GPU work ────────────────────
# A capture that another thread's cudaMalloc / sync / eager sampling breaks
# leaves the CUDA generator stuck in "capturing": every later sampling call or
# replay in the process fails with "Offset increment outside graph capture".

def _until(cond, wait=5.0):
    deadline = time.monotonic() + wait
    while not cond():
        assert time.monotonic() < deadline, "condition never became true"
        time.sleep(0.001)


def test_stream_graph_is_captured_alone_and_ticks_share_the_lock(make_sched):
    sched = make_sched()
    assert sched.frame.built_as == "exclusive"
    seen = []
    prompt = sched.be._prompt_embeds

    def logged_prompt(req):
        seen.append(("prompt", CAPTURE_LOCK.held()))
        return prompt(req)

    def replay():
        seen.append(("replay", CAPTURE_LOCK.held()))
        raise RuntimeError("stop the worker")

    sched.be._prompt_embeds = logged_prompt
    sched.frame.replay = replay
    sched.submit(**_req())
    sched._run()
    assert seen == [("prompt", "shared"), ("replay", "shared")]
    assert CAPTURE_LOCK.held() is None


def test_fused_run_holds_the_lock_frame_by_frame_but_not_in_the_hook():
    B, F = 2, 8
    ff = fused_mod.FusedFrame.__new__(fused_mod.FusedFrame)
    ff.B, ff.max_frames, ff.rep, ff.spk = B, F, None, None
    ff.h = torch.zeros(B, 4)
    ff.codes = torch.zeros(F, B, 3, dtype=torch.long)
    ff.finished = torch.zeros(B, dtype=torch.bool)
    ff.length = torch.zeros(B, dtype=torch.long)
    ff.caps = torch.zeros(B, dtype=torch.long)
    ff.frame = torch.zeros((), dtype=torch.long)
    ff.all_done = torch.zeros((), dtype=torch.bool)
    seen = []
    ff.bb = SimpleNamespace(load=lambda cache, mask, pos: seen.append(("load", CAPTURE_LOCK.held())))

    def replay():
        seen.append(("replay", CAPTURE_LOCK.held()))
        ff.frame += 1
        if int(ff.frame) == 3:
            ff.finished.fill_(True)
            ff.length.fill_(3)
            ff.all_done.fill_(True)

    ff.graph = SimpleNamespace(replay=replay)
    codes = ff.run(torch.ones(B, 4), None, None, None, [F, F], None, F,
                   on_frame=lambda: seen.append(("hook", CAPTURE_LOCK.held())))
    assert [len(c) for c in codes] == [3, 3]
    assert seen == [("load", "shared")] + [("hook", None), ("replay", "shared")] * 3
    assert CAPTURE_LOCK.held() is None


def test_batch_engine_captures_alone_and_shares_prompts_and_prefill(monkeypatch):
    eng = _engine(monkeypatch)
    eng.config = SimpleNamespace(max_position_embeddings=4096, n_vq=4, audio_pad_token_id=0,
                                 speech_generation_end_token_id=1,
                                 speech_generation_start_token_id=2)
    eng.tts.device = torch.device("cuda")     # takes the fused path; nothing here touches it
    seen = []

    def prompt(req):
        seen.append(("prompt", CAPTURE_LOCK.held()))
        return _Any()

    def speaker(spk):
        seen.append(("speaker", CAPTURE_LOCK.held()))

    def prefill(embeds):
        seen.append(("prefill", CAPTURE_LOCK.held()))
        return None, None, None, None

    eng._prompt_embeds = prompt
    eng.tts._resolve_speaker_emb = speaker
    eng.bb = SimpleNamespace(prefill=prefill)
    codes = eng._generate_codes_batch([{"phonemes": "a"}, {"phonemes": "b"}], max_new_frames=10)
    assert codes == ["codes-0", "codes-1"]
    assert FakeFused.built_as == ["exclusive"]
    assert seen == [("prompt", "shared"), ("prompt", "shared"),
                    ("speaker", "shared"), ("speaker", "shared"), ("prefill", "shared")]
    assert CAPTURE_LOCK.held() is None


def test_a_capture_waits_for_another_engines_gpu_work(monkeypatch):
    # The Gradio app keeps its own batch engine next to the one tts.infer()
    # uses; each has its own _fused_lock, so only CAPTURE_LOCK keeps one's
    # capture away from the other's prefill.
    a, b = _engine(monkeypatch), _engine(monkeypatch)
    _gen(a)                                   # a's graph exists: its next call only runs
    log, in_prefill, release = [], threading.Event(), threading.Event()

    def slow_prefill(embeds):
        in_prefill.set()
        release.wait(5)
        log.append("a prefill done")
        return None, None, None, None

    a.bb = SimpleNamespace(prefill=slow_prefill)
    build = FakeFused.__init__

    def logged_build(self, *args, **kwargs):
        log.append("b capture")
        build(self, *args, **kwargs)

    monkeypatch.setattr(FakeFused, "__init__", logged_build)
    ta = threading.Thread(target=_gen, args=(a,))
    ta.start()
    assert in_prefill.wait(5)
    tb = threading.Thread(target=_gen, args=(b, 0.5))
    tb.start()
    _until(lambda: CAPTURE_LOCK._writers_waiting == 1)
    assert log == []                          # b is queued behind a's prefill
    release.set()
    for t in (ta, tb):
        t.join(5)
        assert not t.is_alive()
    assert log == ["a prefill done", "b capture"]


def test_one_batch_engine_however_many_threads_ask(monkeypatch):
    # A server's loader thread (warm-up) and its first request could each
    # build an engine, and each engine captures and keeps its own graphs.
    from vieneu import v3_turbo_serve
    from vieneu.v3turbo import V3TurboVieNeuTTS

    built = []

    class SlowEngine:
        def __init__(self, tts):
            built.append(self)
            time.sleep(0.05)

    monkeypatch.setattr(v3_turbo_serve, "V3TurboBatchEngine", SlowEngine)
    tts = V3TurboVieNeuTTS.__new__(V3TurboVieNeuTTS)
    tts.backend, tts.engine, tts.babble_retries = "pytorch", object(), 1
    tts._batch_engine, tts._batch_lock = None, threading.Lock()
    go = threading.Barrier(8, timeout=5)
    got = []

    def ask():
        go.wait()
        got.append(tts._get_batch_engine())

    threads = [threading.Thread(target=ask) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(5)
    assert len(built) == 1 and len(got) == 8 and all(e is built[0] for e in got)
    assert built[0].babble_retries == 1


class _FakeV3Model:
    """Just enough of the v3 Turbo model for the single-path loops; every call
    records how CAPTURE_LOCK is held. Frame ``eos_at`` is the last one."""

    def __init__(self, eos_at):
        self.eos_at, self.frames, self.seen = eos_at, 0, []

    def _note(self, what):
        self.seen.append((what, CAPTURE_LOCK.held()))

    def _build_inputs_embeds(self, ids, speaker_emb=None):
        self._note("embeds")
        return torch.zeros(1, ids.shape[1], 4)

    def semantic_backbone(self, inputs_embeds, past_key_values=None, **kw):
        self._note("backbone")
        return SimpleNamespace(past_key_values=None, last_hidden_state=torch.zeros(1, inputs_embeds.shape[1], 4))

    def decode_one_frame(self, h, **kw):
        self._note("frame")
        self.frames += 1
        return torch.zeros(3, dtype=torch.long), torch.zeros(1, 1, 4)

    def text_lm_head(self, x):
        return torch.tensor([0.0, 1.0] if self.frames == self.eos_at else [1.0, 0.0])


def _single_path_engine(eos_at):
    from vieneu._v3_turbo_engine.inference_v3_turbo import VieNeuTTSv3Turbo

    eng = VieNeuTTSv3Turbo.__new__(VieNeuTTSv3Turbo)
    eng.device = torch.device("cpu")
    eng.model = _FakeV3Model(eos_at)
    eng.config = SimpleNamespace(speech_generation_end_token_id=1, speech_generation_start_token_id=2,
                                 n_vq=3, audio_pad_token_id=0, default_style_token_id=5)
    eng.use_speaker_embedding = False
    eng.babble_retries = 0
    eng._build_prompt_2d = lambda *a: torch.zeros(6, 4, dtype=torch.long)
    eng.audio_tokenizer = SimpleNamespace(
        decode=lambda c, return_dict=True: (eng.model._note("codec"),
                                            SimpleNamespace(audio=torch.zeros(1, 1, 8)))[1],
        batch_decode=lambda rows, **kw: (eng.model._note("codec"),
                                         SimpleNamespace(audio=torch.zeros(1, 1, 8), audio_lengths=None))[1],
    )
    return eng


def test_single_path_engine_holds_the_lock_for_device_work():
    eng = _single_path_engine(eos_at=4)
    wav = eng.infer(phonemes="xin chao", frame_cap=False, repetition_penalty=1.0)
    assert len(wav) == 8 and eng.model.frames == 4
    assert {held for _, held in eng.model.seen} == {"shared"}
    assert CAPTURE_LOCK.held() is None


def test_single_path_stream_never_holds_the_lock_across_a_yield():
    eng = _single_path_engine(eos_at=5)
    at_yield = []
    gen = eng._stream_generate(torch.zeros(6, 4, dtype=torch.long), None, 0.8, 25, 0.95,
                               max_new_frames=20, chunk_frames=2, repetition_penalty=1.0)
    for _wav in gen:
        at_yield.append(CAPTURE_LOCK.held())
    assert at_yield and set(at_yield) == {None}
    assert eng.model.frames == 5
    assert {held for _, held in eng.model.seen} == {"shared"}
    assert CAPTURE_LOCK.held() is None


# ── a capture waits for one frame, never for a whole job ─────────────────────

def test_per_op_loop_holds_the_lock_frame_by_frame_but_not_in_the_hook(monkeypatch):
    eng = _engine(monkeypatch)
    eng.use_fused = False                     # the per-op loop (VIENEU_FUSED_FRAME=0, CPU)
    eng.config = SimpleNamespace(max_position_embeddings=4096, n_vq=3, audio_pad_token_id=0,
                                 speech_generation_end_token_id=1,
                                 speech_generation_start_token_id=2)
    eng.tts.device = torch.device("cpu")
    eng._prompt_embeds = lambda req: torch.zeros(4, 8)
    eng.tts._resolve_speaker_emb = lambda spk: None
    seen, frames = [], []

    def prefill(embeds):
        seen.append(("prefill", CAPTURE_LOCK.held()))
        return torch.zeros(len(embeds), 8), None, None, None

    def decode_step(se, cache, mask, pos):
        seen.append(("step", CAPTURE_LOCK.held()))
        return torch.zeros(se.shape[0], 8), cache, mask, pos

    def frame(model, h, **kw):
        seen.append(("frame", CAPTURE_LOCK.held()))
        frames.append(1)
        return torch.zeros(h.shape[0], 3, dtype=torch.long), torch.zeros(h.shape[0], 1, 8)

    def text_head(x):                         # EOS (id 1) on the third frame
        return torch.tensor([[0.0, 1.0] if len(frames) == 3 else [1.0, 0.0]] * x.shape[0])

    eng.bb = SimpleNamespace(prefill=prefill, decode_step=decode_step)
    eng.model = SimpleNamespace(
        text_lm_head=text_head,
        _build_inputs_embeds=lambda slot, speaker_emb=None: torch.zeros(slot.shape[0], 1, 8))
    monkeypatch.setattr(engine_mod, "generate_frame_batched", frame)
    eng.frame_hook = lambda: seen.append(("hook", CAPTURE_LOCK.held()))
    codes = eng._generate_codes_batch([{"phonemes": "a"}, {"phonemes": "b"}],
                                      max_new_frames=10, repetition_penalty=1.0)
    assert [len(c) for c in codes] == [3, 3]
    assert seen == ([("prefill", "shared")]
                    + [("hook", None), ("frame", "shared"), ("step", "shared")] * 2
                    + [("hook", None), ("frame", "shared")])
    assert CAPTURE_LOCK.held() is None


def test_single_path_lets_a_capture_in_between_frames():
    # One utterance on the single path takes seconds; a capture elsewhere
    # (another tab's batch engine) must get in after the current frame.
    eng = _single_path_engine(eos_at=4)
    log, threads = [], []
    frame = eng.model.decode_one_frame

    def capture():
        with CAPTURE_LOCK.exclusive():
            log.append("capture")

    def decode_one_frame(h, **kw):
        out = frame(h, **kw)
        log.append(f"frame {eng.model.frames}")
        if eng.model.frames == 2:
            threads.append(threading.Thread(target=capture))
            threads[0].start()
            _until(lambda: CAPTURE_LOCK._writers_waiting == 1)
        return out

    eng.model.decode_one_frame = decode_one_frame
    codes = eng._generate_codes("xin chao", None, None, None, None, True, 0.8, 25, 0.95, 20,
                                repetition_penalty=1.0)
    threads[0].join(5)
    assert not threads[0].is_alive()
    assert len(codes) == 4
    assert log == ["frame 1", "frame 2", "capture", "frame 3", "frame 4"]
    assert CAPTURE_LOCK.held() is None


# ── calling VieNeu while holding the lock fails at once instead of hanging ───
# docs/streaming.md tells hosts to wrap their own CUDA work in
# CAPTURE_LOCK.shared(). A VieNeu call inside that block that waits for another
# thread (a _fused_lock holder, the stream worker) can wait for a thread that
# is itself waiting for the block to end before it captures: a hang.

def test_batch_generation_refuses_a_held_lock(monkeypatch):
    eng = _engine(monkeypatch)
    eng.tts.device = torch.device("cuda")     # warm_fused only looks at the type
    for mode in (CAPTURE_LOCK.shared, CAPTURE_LOCK.exclusive):
        with mode():
            with pytest.raises(RuntimeError, match="infer_batch.*holds CAPTURE_LOCK"):
                _gen(eng)
            with pytest.raises(RuntimeError, match="warm_fused.*holds CAPTURE_LOCK"):
                eng.warm_fused()
    assert eng._fused == {} and not eng._fused_lock.locked()
    _gen(eng)                                 # outside the block it runs
    assert len(eng._fused) == 1


def test_stream_waits_refuse_a_held_lock(make_sched):
    from vieneu.v3turbo import V3TurboVieNeuTTS

    sched = make_sched()
    handle = sched.submit(**_req())
    handle.q.put("audio")                     # without the check the test fails, not hangs
    with CAPTURE_LOCK.shared():
        with pytest.raises(RuntimeError, match="infer_stream.*holds CAPTURE_LOCK"):
            next(iter(handle))
    assert handle.cancelled                   # the row is given up, not left queued

    tts = V3TurboVieNeuTTS.__new__(V3TurboVieNeuTTS)
    tts._fused_available = lambda: True
    tts._stream_lock, tts._stream_sched = threading.Lock(), sched
    with CAPTURE_LOCK.shared():
        with pytest.raises(RuntimeError, match="infer_stream.*holds CAPTURE_LOCK"):
            tts._get_stream_scheduler()
    assert tts._get_stream_scheduler() is sched


def test_single_path_stream_refuses_a_held_lock():
    eng = _single_path_engine(eos_at=3)
    eng._lock = threading.RLock()

    def stream():
        return eng.infer_stream(phonemes="xin chao", frame_cap=False, repetition_penalty=1.0,
                                chunk_frames=2)

    with CAPTURE_LOCK.shared():
        with pytest.raises(RuntimeError, match="infer_stream.*holds CAPTURE_LOCK"):
            next(stream())
    assert len(list(stream())) == 2 and eng.model.frames == 3
    assert CAPTURE_LOCK.held() is None
