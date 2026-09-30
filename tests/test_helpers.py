import contextlib
import subprocess
import sys
import threading

import pytest

from tests.helpers import PROCESS_GROUP_FLAGS, decode_text, finish_process


def test_decode_text_uses_universal_newlines() -> None:
    raw = b'alpha\r\nbeta\rgamma\ndelta'

    assert decode_text(raw) == 'alpha\nbeta\ngamma\ndelta'


def test_finish_process_timeout_reaps_child_and_reports_output() -> None:
    process = subprocess.Popen(
        [
            sys.executable,
            '-u',
            '-c',
            (
                'import sys, threading; '
                'print("sentinel", flush=True); '
                'print("ready", file=sys.stderr, flush=True); '
                'threading.Event().wait(30)'
            ),
        ],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding='utf-8',
        start_new_session=True,
        creationflags=PROCESS_GROUP_FLAGS,
    )
    stderr = process.stderr
    assert stderr is not None
    marker: list[str] = []
    ready = threading.Event()

    def read_ready() -> None:
        marker.append(stderr.readline())
        ready.set()

    reader = threading.Thread(target=read_ready, daemon=True)
    reader.start()
    try:
        assert ready.wait(10)
        assert marker == ['ready\n']
        with pytest.raises(AssertionError) as failure:
            finish_process(process, timeout=0.2)
        transcript = str(failure.value)
        assert 'note: timed out after 0.2 s' in transcript
        assert '  | sentinel' in transcript
        assert process.returncode is not None
    finally:
        with contextlib.suppress(OSError):
            process.kill()
        process.wait(timeout=2)
        reader.join(timeout=2)
        assert not reader.is_alive()
        process.communicate(timeout=2)
