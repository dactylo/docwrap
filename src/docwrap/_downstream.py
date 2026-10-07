"""Absorb closure of a downstream reader of standard output."""

import contextlib
import errno
import io
import os
import sys
from collections.abc import Generator, Iterable
from typing import TextIO, cast

_WINDOWS = sys.platform == 'win32'


class ClosableTextOutput:
    """Text output that discards writes after its downstream reader closes.

    Attributes other than writing and flushing come from the wrapped stream, so
    encoding, terminal detection, and reconfiguration behave as on that stream.
    """

    def __init__(self, stream: TextIO) -> None:
        """Wrap a text stream.

        Args:
            stream: Stream written through until its reader closes.
        """
        self._closed_downstream = False
        self._stream = stream

    def __getattr__(self, name: str) -> object:
        """Return an attribute of the wrapped stream."""
        return getattr(self._stream, name)

    @property
    def closed_downstream(self) -> bool:
        """Whether the downstream reader has closed the stream."""
        return self._closed_downstream

    def flush(self) -> None:
        """Flush the wrapped stream unless its reader has closed.

        Raises:
            OSError: If flushing fails for a reason other than closure.
        """
        if self._closed_downstream:
            return
        try:
            self._stream.flush()
        except OSError as error:
            if not _is_downstream_closure(error):
                raise
            self._closed_downstream = True

    def write(self, text: str) -> int:
        """Write text, or discard it after the reader has closed.

        Args:
            text: Text to write.

        Returns:
            The number of characters accepted.

        Raises:
            OSError: If writing fails for a reason other than closure.
        """
        if self._closed_downstream:
            return len(text)
        try:
            return self._stream.write(text)
        except OSError as error:
            if not _is_downstream_closure(error):
                raise
            self._closed_downstream = True
            return len(text)

    def writelines(self, lines: Iterable[str]) -> None:
        """Write lines until the reader closes, then stop consuming them.

        Args:
            lines: Text lines in output order.

        Raises:
            OSError: If writing fails for a reason other than closure.
        """
        if self._closed_downstream:
            return
        for line in lines:
            self.write(line)
            if self._closed_downstream:
                return


def _is_downstream_closure(error: OSError) -> bool:
    """Report whether a write or flush failed because the reader closed the stream."""
    return isinstance(error, BrokenPipeError) or (_WINDOWS and error.errno == errno.EINVAL)


def _silence_descriptor(stream: TextIO) -> None:
    """Point a stream's descriptor at the null device when it has one."""
    try:
        target = stream.fileno()
    except (OSError, ValueError):
        return
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, target)
    finally:
        os.close(devnull)


@contextlib.contextmanager
def closable_stdout() -> Generator[ClosableTextOutput]:
    """Route standard output through a closure-absorbing wrapper.

    On every exit, including `SystemExit` from argument parsing, pending output
    is flushed through the wrapper and the original stream is restored. If the
    reader closed, the original descriptor is pointed at the null device so the
    interpreter's final flush cannot fail again. The closure latch remains
    because display functions write from many places, and the documented exit
    status reflects the work completed when the reader closes.

    Yields:
        The installed wrapper.
    """
    original = sys.stdout
    output = ClosableTextOutput(original)
    sys.stdout = cast('TextIO', output)
    try:
        yield output
    finally:
        try:
            output.flush()
        finally:
            sys.stdout = original
            if output.closed_downstream:
                _silence_descriptor(original)


def ensure_utf8_stdio() -> None:
    """Reconfigure supported standard streams to UTF-8 on every platform.

    Each text stream retains its existing error handler.
    """
    # Transitional: PEP 686 makes UTF-8 mode the default in Python 3.15. Remove this
    # function when requires-python is >=3.15; PYTHONUTF8=0 then remains a deliberate opt-out.
    for stream in (sys.stdin, sys.stdout, sys.stderr):
        if isinstance(stream, io.TextIOWrapper):
            stream.reconfigure(encoding='utf-8', errors=stream.errors)


def stdout_closed() -> bool:
    """Report whether standard output's downstream reader has closed.

    Returns:
        Whether the installed wrapper has observed closure.
    """
    output = sys.stdout
    return isinstance(output, ClosableTextOutput) and output.closed_downstream
