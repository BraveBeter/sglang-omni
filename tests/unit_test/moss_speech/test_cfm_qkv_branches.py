"""Fixed branch owners, stream handoff and failed cleanup retain resources."""

from types import SimpleNamespace as Obj

import pytest

from sglang_omni.models.moss_speech.components.cfm_qkv_multistream import QkvBranches

from .test_cfm_qkv_schedule import fixture


def branches():
    api, streams, contexts = fixture()
    api.current = streams["q"]
    events = []
    pool = iter((streams["k"], streams["v"]))
    api.Stream = lambda **kw: next(pool)
    api.synchronize = lambda device: events.append("sync")
    for kind, stream in streams.items():
        stream.wait_stream = lambda other, k=kind: events.append(
            ("wait", k, other.cuda_stream)
        )
    for kind, context in contexts.items():
        context.prepare_stream = lambda k=kind: events.append(("prepare", k))
        context.close = lambda k=kind: events.append(("close", k))
        context.owners = (context.workspace,)

    def factory(device):
        return next(contexts[k] for k in ("k", "v") if streams[k] is api.current)

    value = QkvBranches(api, factory, contexts["q"])
    return value, api, streams, contexts, events


def test_prepare_bind_keeps_branch_contexts_and_all_schedule_events():
    value, api, streams, contexts, events = branches()
    value.prepare()
    first = value.bind()
    assert len(value.owners) == 2
    newer = Obj(cuda_stream=7, device="cuda:0", wait_event=streams["q"].wait_event)
    api.current = newer
    with pytest.raises(ValueError, match="Main context"):
        value.bind()
    contexts["q"].stream = newer
    second = value.bind()
    assert value.contexts == {k: contexts[k] for k in ("k", "v")}
    assert value.schedules == [first, second]
    assert first.initial[0][0] == 1 and second.initial[0][0] == 7
    value.close()
    assert events[-3:] == ["sync", ("close", "k"), ("close", "v")]
    assert first.closed and second.closed
    assert value.closed and not value.owners


def test_partial_prepare_failure_is_owned_and_sync_failure_does_not_release():
    value, api, streams, contexts, events = branches()

    def fail():
        raise RuntimeError("prepare v failed")

    contexts["v"].prepare_stream = fail
    with pytest.raises(RuntimeError, match="prepare v"):
        value.prepare()
    assert set(value.contexts) == {"k", "v"}

    def syncfail(device):
        raise RuntimeError("sync failed")

    api.synchronize = syncfail
    with pytest.raises(RuntimeError, match="sync failed"):
        value.close()
    assert set(value.contexts) == {"k", "v"} and not value.closed
    assert not any(isinstance(v, tuple) and v[0] == "close" for v in events)
    api.synchronize = lambda device: events.append("sync")
    value.close()
    assert value.closed


def test_startup_control_joins_on_error_and_wrong_caller_cannot_run():
    value, api, streams, contexts, events = branches()
    value.prepare()

    def fail(actual):
        assert api.current is streams["k"]
        raise RuntimeError("original control failed")

    with pytest.raises(RuntimeError, match="original control"):
        value.call("k", fail, object())
    assert api.current is streams["q"]
    assert events[-1] == ("wait", "q", 2)
    api.current = streams["v"]
    with pytest.raises(ValueError, match="Wrong startup"):
        value.call("k", lambda value: None, object())
