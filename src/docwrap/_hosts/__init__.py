"""Discover and load hook host adapters through entry points."""

from importlib.metadata import entry_points

from docwrap.hooks import API_VERSION, HookAdapter

_ENTRY_POINT_GROUP = 'docwrap.hook_adapters'


class AdapterLoadError(Exception):
    """Report that no functioning adapter exists for the selected host."""


def _registered_hosts() -> list[str]:
    """Return sorted names of registered hook adapters."""
    return sorted({entry.name for entry in entry_points(group=_ENTRY_POINT_GROUP)})


def load_adapter(name: str) -> HookAdapter:
    """Load the adapter registered under a host name.

    Args:
        name: Host name given to `--agent`.

    Returns:
        A new adapter instance.

    Raises:
        AdapterLoadError: If the name is unregistered or registered more than
            once, the entry point cannot be imported or instantiated, or the
            adapter's `api_version` differs from `API_VERSION`.
    """
    matches = [entry for entry in entry_points(group=_ENTRY_POINT_GROUP) if entry.name == name]
    if not matches:
        hosts = ', '.join(_registered_hosts()) or 'none'
        raise AdapterLoadError(f"unknown host '{name}'; registered hosts: {hosts}")
    if len(matches) > 1:
        raise AdapterLoadError(f"host '{name}' is registered {len(matches)} times")
    try:
        factory = matches[0].load()
        adapter = factory()
    except Exception as error:
        raise AdapterLoadError(f"host '{name}' failed to load: {error}") from error
    version = getattr(adapter, 'api_version', None)
    if version != API_VERSION:
        raise AdapterLoadError(
            f"host '{name}' implements hook adapter API {version!r}; "
            f'docwrap requires {API_VERSION}'
        )
    return adapter
