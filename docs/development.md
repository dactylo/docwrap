# Development

## Local environment

Run `make setup` from the checkout with uv 0.12.23 and a supported `python` on `PATH`. It
synchronizes the hashed development lock into `.venv` and installs the checkout in editable mode.
`make quality` runs the read-only formatting, lock, type, dead-code, manifest, and whitespace
checks; `make check` adds the full test suite. `make test` runs that suite alone, and `make format`
applies the repository formatters. Project commands refuse a missing or outdated setup. Biome,
Taplo, and shfmt are separate native tools on `PATH` for formatting.

Use `make install` to install the command-line tool with `uv tool`, or `make reinstall` to replace
it. `make format-check`, `make lock-check`, `make type-check`, `make dead-code-check`,
`make manifest-check`, and `make whitespace-check` run the corresponding read-only checks.

## Continuous integration

The [CI workflow](../.github/workflows/ci.yml) runs `quality` first. The `test` and `hooks` jobs
then run in parallel. `package` starts only after those jobs pass; `hosted-validation` reports the
outcome of a hosted run, including failures.

CI uses GitHub-hosted runners unless a run explicitly requests local runners. In a private
repository, the Actions variables `CI_LINUX_RUNNER`, `CI_MACOS_RUNNER`, and `CI_WINDOWS_RUNNER`
select local runners for runs started from Actions > CI > Run workflow with the hosted option
disabled. If a variable is absent, that operating system uses a hosted runner. Pushes, pull
requests, public repositories, and runs with the hosted option enabled use hosted runners for every
job and run every test.

A staging repository sets the Actions variable `CI_RUN_ON_PUSH` to `false`. Its pushes then appear
as `CI / push skipped` and run no jobs; CI runs only on request. A newer requested run on a branch
cancels an older one. Push runs are never cancelled, because the Release workflow needs the tagged
commit's push run.

| Requested run                                                 | Actions > CI > Run workflow | Title               |
| ------------------------------------------------------------- | --------------------------- | ------------------- |
| Quick: local runners, slow tests omitted on macOS and Windows | hosted off, slow tests off  | `CI / local`        |
| Full: local runners, every test                               | hosted off, slow tests on   | `CI / local (full)` |
| Hosted: GitHub-hosted runners, every test                     | hosted on                   | `CI / hosted`       |

The `test` matrix covers Python 3.13 to 3.15 on Linux and Python 3.13 on macOS and Windows. All
test jobs have a one-hour hang guard, on local and hosted runners, in quick and full runs. The
`hooks` job runs both pre-commit hooks against a sample file in a temporary repository, the way a
user would, without building distributions. `package` also loads the bundled plugin entry points
from the installed wheel.

CI runs tests in parallel with pytest-xdist. Set `PYTEST_XDIST_AUTO_NUM_WORKERS` when a runner's
visible CPU count exceeds its CPU quota. For the same scheduling locally, run:

```sh
make test TEST_FLAGS='-n auto --dist worksteal --durations=15'
```

The manual [Release workflow](../.github/workflows/release.yml) publishes the package to PyPI after
checking its version tag and a successful push CI run for the tagged commit.

## Dependency updates

`pyproject.toml` declares dependency ranges. `requirements-ci.txt` records exact versions and
hashes for the development tools, test dependencies, and build backend. `requirements-test.txt`
selects the test dependencies and Hatchling build backend from those same pins. Quality checks
install the full set, including `ci-build` tools; test jobs install the smaller test set. The built
wheel installs in a clean environment with runtime dependencies constrained by the CI pins.
Published packages keep the ranges declared in `pyproject.toml`.

Use the uv version pinned in [the CI workflow](../.github/workflows/ci.yml) when regenerating the
files. After changing dependencies, regenerate both files with:

```sh
make lock-update
```

This retains existing compatible pins. To update to newer compatible versions deliberately:

```sh
make lock-update LOCK_FLAGS=--upgrade
```

Review and commit both generated files with the dependency changes. `make lock-check` checks that
they are current without rewriting them; `make check` includes that check. Generation and checking
resolve dependencies but do not install them.

The `ci-build` dependency group supplies build tools for third-party dependencies that need a
source build on a supported CI interpreter. CI verifies hashes when installing the pinned tools and
build backend.

The Markdown formatter uses Taplo for TOML code blocks and shfmt for shell code blocks. CI
downloads their official Linux binaries for the quality runner's architecture and verifies pinned
SHA-256 checksums. The versions and checksums live in the workflow. uv excludes `mdformat-config`'s
Taplo Python dependency because that distribution lacks Linux and Windows ARM wheels; the native
executable supplies the formatter instead. For local Markdown formatting, provide `taplo` and
`shfmt` on `PATH`. Test jobs do not need them.

## Test selection

Workers use the available CPU count. Use `-n 0` for a serial debugging run; plain `make test`
remains serial.
