"""Shared argument helpers for CLI and hook dispatch."""

import argparse


def incremental_override(args: argparse.Namespace) -> bool | None:
    """Map CLI incremental switches to a tri-state override.

    Args:
        args: Parsed options containing incremental and no_incremental flags.

    Returns:
        True to enable incremental processing, False to disable it, or None to
        defer to configuration.
    """
    if args.no_incremental:
        return False
    if args.incremental:
        return True
    return None
