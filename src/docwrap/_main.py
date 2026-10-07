"""Dispatch the console script and `python -m docwrap`."""

import sys
from collections.abc import Sequence

from docwrap._downstream import closable_stdout, ensure_utf8_stdio
from docwrap._hook_front import run_hook_command

_EXIT_ERROR = 2


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI command and emit its output and diagnostics.

    Report an unavailable stdout and do nothing else. Otherwise, configure
    standard streams for UTF-8 where supported before processing arguments. If
    stdout closes, discard remaining output, examine no further files, and
    return the status of work completed so far.

    Args:
        argv: Command-line arguments, or `None` to use `sys.argv[1:]`.

    Returns:
        For file and statistics commands, 0 on success, including successful
        writes; 1 when changes remain pending; or 2 on processing errors.
        Hook commands use host-specific exit statuses, or 2 when no functioning
        adapter is available. Every command returns 2 when stdout is
        unavailable at startup.

    Raises:
        SystemExit: When argument parsing displays help or rejects arguments.
    """
    if sys.stdout is None:
        # Started with standard output closed: nothing can be delivered, and the
        # closed-stdout wrapper cannot wrap a missing stream.
        print('error: standard output is unavailable', file=sys.stderr)
        return _EXIT_ERROR
    ensure_utf8_stdio()
    with closable_stdout():
        args = sys.argv[1:] if argv is None else list(argv)
        if args and args[0] == 'hook':
            return run_hook_command(args[1:])
        from docwrap._cli import run_cli  # noqa: PLC0415

        return run_cli(args)
