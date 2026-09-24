"""Host dependency/ownership failures before any CUDA implementation executes."""

import ctypes as C
import unittest
from contextlib import contextmanager
from types import SimpleNamespace as Obj


class Api:
    def __init__(self):
        self.current = Obj(cuda_stream=1, device="cuda:0")
        self.events = []
        self.log = []

    def current_stream(self, device=None):
        return self.current

    def is_current_stream_capturing(self):
        return False

    def Event(self, **kwargs):
        self.assert_event = kwargs
        api = self

        class Event:
            def __init__(self):
                self.name = "e" + str(len(api.events))
                api.events.append(self)

            def record(self, stream):
                api.log.append(("record", self.name, stream.cuda_stream))

        return Event()

    @contextmanager
    def stream(self, stream):
        old = self.current
        self.current = stream
        try:
            yield
        finally:
            self.current = old


def fixture():
    api = Api()
    streams = {
        k: Obj(cuda_stream=i, device="cuda:0") for i, k in enumerate(("q", "k", "v"), 1)
    }
    for stream in streams.values():
        stream.wait_event = lambda event, s=stream: api.log.append(
            ("wait", s.cuda_stream, event.name)
        )
    contexts = {
        k: Obj(
            stream=streams[k],
            device="cuda:0",
            handle=C.c_void_p(1234),
            version=120804,
            path="/qualified/libcublasLt.so.12",
            workspace=Obj(data_ptr=lambda i=i: 1048576 * (i + 1)),
            closed=False,
            failed=False,
        )
        for i, k in enumerate(streams)
    }
    return api, streams, contexts


class ScheduleTests(unittest.TestCase):
    def make(self):
        from sglang_omni.models.moss_speech.components.cfm_qkv_multistream import (
            QkvSchedule,
        )

        api, streams, contexts = fixture()
        return api, streams, contexts, QkvSchedule(api, streams, contexts)

    def test_default_main_stream_zero_keeps_real_fork_join(self):
        from sglang_omni.models.moss_speech.components.cfm_qkv_multistream import (
            QkvSchedule,
        )

        api, streams, contexts = fixture()
        streams["q"].cuda_stream = 0
        api.current = streams["q"]
        schedule = QkvSchedule(api, streams, contexts)
        self.assertEqual(schedule.initial[0][0], 0)
        api.log.clear()
        seen = []
        value = object()
        schedule.run(value, lambda k, v: seen.append((k, v, api.current.cuda_stream)))
        self.assertEqual({k: stream for k, _, stream in seen}, {"q": 0, "k": 2, "v": 3})
        self.assertTrue(all(v is value for _, v, _ in seen))
        self.assertEqual(api.current.cuda_stream, 0)
        self.assertEqual(schedule.calls, 1)
        for done in [x for x in api.log if x[0] == "record" and x[2] in (2, 3)]:
            self.assertIn(("wait", 0, done[1]), api.log)
        schedule.close()
        self.assertIsNone(schedule.input_owner)

    def test_zero_is_only_valid_for_main_not_shared_branch_or_storage(self):
        from sglang_omni.models.moss_speech.components.cfm_qkv_multistream import (
            QkvSchedule,
        )

        for change in (
            lambda s, c: setattr(s["q"], "cuda_stream", -1),
            lambda s, c: setattr(s["k"], "cuda_stream", 0),
            lambda s, c: setattr(s["v"], "cuda_stream", 0),
            lambda s, c: setattr(c["k"].handle, "value", 0),
            lambda s, c: setattr(c["v"].workspace, "data_ptr", lambda: 0),
            lambda s, c: setattr(c["q"], "device", "cuda:1"),
        ):
            api, streams, contexts = fixture()
            streams["q"].cuda_stream = 0
            api.current = streams["q"]
            change(streams, contexts)
            with (
                self.subTest(change=change),
                self.assertRaises((ValueError, TypeError)),
            ):
                QkvSchedule(api, streams, contexts)

    def test_fork_each_copy_then_join_and_same_input(self):
        api, streams, contexts, schedule = self.make()
        api.log.clear()
        value = object()
        seen = []

        def enqueue(kind, actual):
            self.assertIs(actual, value)
            seen.append(kind)
            api.log.append(("work", kind, api.current.cuda_stream))

        schedule.run(value, enqueue)
        self.assertEqual(set(seen), {"q", "k", "v"})
        self.assertEqual(len(seen), 3)
        self.assertEqual(api.current.cuda_stream, 1)
        self.assertIs(schedule.input_owner, value)
        work = [x for x in api.log if x[0] == "work"]
        self.assertEqual({k: s for _, k, s in work}, {"q": 1, "k": 2, "v": 3})
        fork = next(x for x in api.log if x[0] == "record" and x[2] == 1)
        for kind in ("k", "v"):
            wait = ("wait", streams[kind].cuda_stream, fork[1])
            self.assertLess(
                api.log.index(wait),
                api.log.index(("work", kind, streams[kind].cuda_stream)),
            )
            done = next(
                x
                for x in api.log
                if x[0] == "record" and x[2] == streams[kind].cuda_stream
            )
            self.assertLess(
                api.log.index(("work", kind, streams[kind].cuda_stream)),
                api.log.index(done),
            )
            self.assertLess(api.log.index(done), api.log.index(("wait", 1, done[1])))
        self.assertFalse(schedule.busy)

    def test_workspace_handle_stream_aliases_rejected(self):
        from sglang_omni.models.moss_speech.components.cfm_qkv_multistream import (
            QkvSchedule,
        )

        for mutate in (
            lambda s, c: c["v"].workspace.__setattr__(
                "data_ptr", c["k"].workspace.data_ptr
            ),
            lambda s, c: c["v"].workspace.__setattr__(
                "data_ptr", lambda: c["k"].workspace.data_ptr() + 64
            ),
            lambda s, c: c["v"].__setattr__("handle", c["k"].handle),
            lambda s, c: c.__setitem__("v", c["k"]),
            lambda s, c: c["v"].__setattr__("version", 120803),
            lambda s, c: c["v"].__setattr__("path", "/foreign/libcublasLt.so.12"),
            lambda s, c: s["v"].__setattr__("cuda_stream", s["k"].cuda_stream),
            lambda s, c: c["v"].__setattr__("stream", s["q"]),
            lambda s, c: c["v"].__setattr__("closed", True),
        ):
            api, streams, contexts = fixture()
            mutate(streams, contexts)
            with self.subTest(mutate=mutate), self.assertRaises(ValueError):
                QkvSchedule(api, streams, contexts)

    def test_exception_restores_stream_and_joins_started_branches(self):
        api, _, _, schedule = self.make()

        def enqueue(kind, value):
            api.log.append(("work", kind, api.current.cuda_stream))
            if kind == "v":
                raise RuntimeError("operator failed")

        with self.assertRaisesRegex(RuntimeError, "operator failed"):
            schedule.run(object(), enqueue)
        self.assertEqual(api.current.cuda_stream, 1)
        self.assertFalse(schedule.busy)
        for done in [x for x in api.log if x[0] == "record" and x[2] in (2, 3)]:
            self.assertIn(("wait", 1, done[1]), api.log)

    def test_observed_opaque_alias_is_not_resource_alias(self):
        from sglang_omni.models.moss_speech.components.cfm_qkv_multistream import (
            identity,
        )

        api, streams, contexts, schedule = self.make()
        self.assertEqual(len({c.handle.value for c in contexts.values()}), 1)
        self.assertEqual(len({C.addressof(c.handle) for c in contexts.values()}), 3)
        self.assertEqual(identity(streams, contexts), schedule.initial)
        schedule.run(object(), lambda *args: None)
        self.assertEqual(schedule.calls, 1)

    def test_wrong_stream_reentrant_closed_and_changed_owner_rejected(self):
        api, streams, contexts, schedule = self.make()
        api.current = streams["k"]
        with self.assertRaises(ValueError):
            schedule.run(object(), lambda *a: None)
        api.current = streams["q"]
        schedule.busy = True
        with self.assertRaises(ValueError):
            schedule.run(object(), lambda *a: None)
        schedule.busy = False
        contexts["k"].handle.value = 999
        with self.assertRaises(ValueError):
            schedule.run(object(), lambda *a: None)
        _, _, _, schedule = self.make()
        schedule.close()
        with self.assertRaises(ValueError):
            schedule.run(object(), lambda *a: None)
        schedule.close()
        self.assertIsNone(schedule.input_owner)


if __name__ == "__main__":
    unittest.main()
