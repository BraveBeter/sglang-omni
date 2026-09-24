"""Full-CFM graph storage may cross startup/executor threads, never overlap."""

import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

path = (
    Path(__file__).resolve().parents[3]
    / "sglang_omni/models/moss_speech/components/cfm_graph_lifetime.py"
)
spec = spec_from_file_location("cfm_graph_lifetime", path)
module = module_from_spec(spec)
spec.loader.exec_module(module)
ReplayLease = module.ReplayLease


def lease():
    storage = {"pointer": 10, "shape": (2, 8, 1280, 64), "version": 0}
    return ReplayLease((storage,), lambda t: tuple(t.items()), stream=7), storage


class ReplayLifetimeTests(unittest.TestCase):
    def test_startup_to_executor_thread_and_owned_storage(self):
        item, storage = lease()
        creator = threading.get_ident()

        def run():
            self.assertNotEqual(threading.get_ident(), creator)
            with item.use(stream=7):
                self.assertEqual(item.owners, (storage,))
            return True

        with ThreadPoolExecutor(max_workers=1) as pool:
            self.assertTrue(pool.submit(run).result(timeout=5))
        item.close()
        self.assertEqual(item.owners, ())

    def test_concurrent_use_and_close_are_rejected_then_recover(self):
        item, _ = lease()
        entered, release = threading.Event(), threading.Event()

        def run():
            with item.use(stream=7):
                entered.set()
                self.assertTrue(release.wait(5))

        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(run)
            self.assertTrue(entered.wait(5))
            try:
                with self.assertRaisesRegex(RuntimeError, "busy"):
                    with item.use(stream=7):
                        pass
                with self.assertRaisesRegex(RuntimeError, "busy"):
                    item.close()
            finally:
                release.set()
            future.result(timeout=5)
        with item.use(stream=7):
            pass

    def test_exceptions_and_wrong_stream_do_not_lock_bank(self):
        item, _ = lease()
        with self.assertRaisesRegex(RuntimeError, "stream"):
            with item.use(stream=8):
                pass
        with self.assertRaisesRegex(ValueError, "work"):
            with item.use(stream=7):
                raise ValueError("work")
        with item.use(stream=7):
            with self.assertRaisesRegex(RuntimeError, "busy"):
                with item.use(stream=7):
                    pass

    def test_storage_change_permanently_invalidates(self):
        for key, value in [
            ("pointer", 11),
            ("shape", (2, 8, 1279, 64)),
            ("version", 1),
        ]:
            with self.subTest(key=key):
                item, storage = lease()
                original = storage[key]
                storage[key] = value
                with self.assertRaisesRegex(RuntimeError, "storage"):
                    with item.use(stream=7):
                        pass
                storage[key] = original
                with self.assertRaisesRegex(RuntimeError, "invalid"):
                    with item.use(stream=7):
                        pass
                item.close()
                item.close()
                with self.assertRaisesRegex(RuntimeError, "closed"):
                    with item.use(stream=7):
                        pass

    def test_metadata_error_invalidates_and_releases_guard(self):
        broken = False

        def describe(value):
            if broken:
                raise ValueError("metadata")
            return value

        item = ReplayLease((1,), describe, stream=7)
        broken = True
        with self.assertRaisesRegex(ValueError, "metadata"):
            with item.use(stream=7):
                pass
        broken = False
        with self.assertRaisesRegex(RuntimeError, "invalid"):
            with item.use(stream=7):
                pass
        item.close()


if __name__ == "__main__":
    unittest.main()
