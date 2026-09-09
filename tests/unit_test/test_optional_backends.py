# SPDX-License-Identifier: Apache-2.0
"""Missing optional backends must not look like startup failures on other devices."""

import subprocess
import sys

import pytest


@pytest.mark.parametrize("backend", ["musa", "nixl"])
def test_missing_backend_is_quiet_until_selected(backend):
    code = """
import logging
import sys

logging.basicConfig(level=logging.WARNING)
sys.modules['torchada'] = None
sys.modules['nixl'] = None
sys.modules['nixl._api'] = None
if sys.argv[1] == 'musa':
    from sglang_omni.platforms.cuda import CUDAOmniPlatform
    assert CUDAOmniPlatform().device_type == 'cuda'
else:
    from sglang_omni.relay.base import create_relay
    relay = create_relay('shm', engine_id='test', device='cpu')
    relay.close()
    try:
        create_relay('nixl', engine_id='test', device='cpu')
    except ImportError as exc:
        assert 'nixl' in str(exc).lower()
        assert isinstance(exc.__cause__, ImportError)
    else:
        raise AssertionError('Selecting unavailable NIXL must fail')
"""
    result = subprocess.run(
        [sys.executable, "-c", code, backend],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stderr
    assert "WARNING:sglang_omni.platforms.musa:" not in result.stderr
    assert "ERROR:sglang_omni.relay.nixl:" not in result.stderr
