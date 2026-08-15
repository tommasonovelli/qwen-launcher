"""Install, describe, and launch the managed DeepSeek Harness interface.

DeepSeek Harness (`dsh`) is an upstream program this launcher starts; it is not part of the
distribution and it is never modified. Everything bora decides travels through the child process
environment and through one overlay bora owns and rewrites on every launch, so nothing is written
into the interface's own storage, no credential is held inside it, and its API is never called
(D-097, keeping the rule D-094 established for the interface it replaces).

The command line below was read at npm version `0.1.0-rc.6` and measured on Ubuntu. Two of its
properties decided the shape of this module: `--patch` is a launcher flag and must precede the web
app's own arguments, and the harness publishes no readiness endpoint at all, so `GET /` answering
`200 text/html` is the only signal there is (specification section 5.9).
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from bora_workbench.paths import harness_dir, node_package_script
from bora_workbench.process import ReadinessContract

DSH_VERSION = "0.1.0-rc.6"
DSH_PACKAGE = "@deepseek-ai/dsh"
DSH_REQUIREMENT = f"{DSH_PACKAGE}@{DSH_VERSION}"

# `--host` is the only place the bind address is decided, and it is a constant here so no
# configuration key can ever widen it (specification section 5.12, D-015). Upstream refuses
# `0.0.0.0` itself, but bora states the address rather than inheriting a default it does not own.
LOOPBACK_HOST = "127.0.0.1"

# The harness declares `^22.19.0 || >=24.0.0`; a Node below that boots into failures that read as
# harness bugs rather than as an unmet prerequisite, so it is refused before anything is installed.
NODE_MINIMUM = (22, 19, 0)
NODE_MINIMUM_TEXT = "22.19.0"

# The harness serves its single-page application from the fallback route, so every unmatched path
# answers `200 text/html` and no health path exists to poll. Readiness is therefore the first
# successful response on `/`, and a first boot was measured at about one second.
READY_PATH = "/"
_STARTUP_TIMEOUT_SECONDS = 5 * 60.0

_MARKER_NAME = "installed.json"
_OVERLAY_NAME = "bora.patch.yml"
_CREDENTIAL_VARIABLE = "BORA_LOCAL_KEY"
_LOCAL_CREDENTIAL = "bora-local"
_PROVIDER_ROUTE = "bora"
# One of upstream's three shipped presets. `read-only` still asks before acting; `workspace-write`
# would let a session edit the launch directory without the user having chosen that.
_PERMISSION_MODE = "read-only"
_VERSION_PATTERN = re.compile(r"^v?(\d+)\.(\d+)\.(\d+)")


class HarnessError(RuntimeError):
    """Report an actionable failure to install, inspect, or configure the managed interface."""


@dataclass(frozen=True, slots=True)
class HarnessStatus:
    """Describe the managed interface installation without starting or repairing anything."""

    root: Path
    version: str | None
    script: Path | None

    @property
    def is_installed(self) -> bool:
        """Report whether the pinned version is present and its entry module exists."""
        return self.version == DSH_VERSION and self.script is not None


@dataclass(frozen=True, slots=True)
class HarnessLaunch:
    """Group everything one managed interface process needs, resolved and validated."""

    script: Path
    port: int
    llama_port: int
    home_dir: Path
    model_alias: str
    is_vision: bool = False


def environment_dir(root: Path) -> Path:
    """Return the npm prefix holding the pinned harness and its dependency tree."""
    return root / "node"


def interface_data_dir(root: Path) -> Path:
    """Return the directory the harness owns: `$DSH_HOME`, with its profiles and sessions."""
    return root / "home"


def overlay_path(root: Path) -> Path:
    """Return the launch overlay bora owns, which is rewritten before every start."""
    return root / _OVERLAY_NAME


def _marker_path(root: Path) -> Path:
    """Return the file recording which version the managed installation actually holds."""
    return environment_dir(root) / _MARKER_NAME


def _installed_version(root: Path) -> str | None:
    """Read the recorded version, treating an absent or unreadable marker as not installed.

    The marker is written last, so an interrupted installation leaves no version behind and the
    next `bora ui install` rebuilds the tree instead of trusting a partial one.
    """
    try:
        recorded = json.loads(_marker_path(root).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    version = recorded.get("version") if isinstance(recorded, dict) else None
    return version if isinstance(version, str) else None


def entry_script(root: Path) -> Path:
    """Return the harness entry module inside the managed npm prefix."""
    return node_package_script(environment_dir(root), DSH_PACKAGE, "lib/bin.js")


def inspect_harness(root: Path | None = None) -> HarnessStatus:
    """Describe the managed installation without creating directories or starting a process."""
    selected = harness_dir() if root is None else root
    script = entry_script(selected)
    present = script if script.is_file() else None
    return HarnessStatus(selected, _installed_version(selected), present)


def _require_executable(name: str, remedy: str) -> str:
    """Locate one required external program, naming what to install when it is absent."""
    located = shutil.which(name)
    if located is None:
        raise HarnessError(f"{name} is required to install DeepSeek Harness; {remedy}")
    return located


def _parse_version(text: str) -> tuple[int, int, int] | None:
    """Read the leading `major.minor.patch` of a version string, ignoring any suffix."""
    match = _VERSION_PATTERN.match(text.strip())
    if match is None:
        return None
    return (int(match.group(1)), int(match.group(2)), int(match.group(3)))


def _node_version(node: str) -> tuple[int, int, int]:
    """Ask the located Node for its own version, which decides whether the harness can run."""
    try:
        result = subprocess.run(
            [node, "--version"], check=False, capture_output=True, text=True, timeout=30
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise HarnessError(f"cannot run {node} --version: {error}") from error
    parsed = _parse_version(result.stdout) if result.returncode == 0 else None
    if parsed is None:
        raise HarnessError(f"{node} --version did not report a usable version")
    return parsed


def require_node() -> str:
    """Return a Node new enough for the pinned harness, or explain exactly what is missing."""
    node = _require_executable("node", "install Node.js from https://nodejs.org/ and try again")
    version = _node_version(node)
    if version < NODE_MINIMUM:
        found = ".".join(str(part) for part in version)
        raise HarnessError(
            f"DeepSeek Harness requires Node {NODE_MINIMUM_TEXT} or newer; {node} is {found}"
        )
    return node


def _run(command: list[str], description: str) -> None:
    """Run one installer step without a shell, leaving its own progress visible to the user.

    The output is not captured, because an installation this large is worth watching: npm prints
    which package it is resolving, and one native dependency is compiled from source on Linux.
    """
    try:
        result = subprocess.run(command, check=False)
    except OSError as error:
        raise HarnessError(f"cannot {description}: {error}") from error
    if result.returncode != 0:
        raise HarnessError(f"could not {description}: npm exited with code {result.returncode}")


def _write_marker(root: Path) -> None:
    """Record the installed version only once the entry module is present."""
    payload = {"version": DSH_VERSION, "package": DSH_PACKAGE}
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    try:
        _marker_path(root).write_text(text, encoding="utf-8")
    except OSError as error:
        raise HarnessError(f"cannot record the DeepSeek Harness installation: {error}") from error


def install_harness(root: Path | None = None, *, force: bool = False) -> HarnessStatus:
    """Create the managed npm prefix and install exactly the pinned harness release.

    The version is pinned here rather than resolved, so two machines running the same bora install
    the same interface; `latest` is forbidden by specification section 2. Install scripts are run
    rather than skipped, because the harness mounts `node-pty` as a boot-time plugin and a tree
    installed with `--ignore-scripts` fails to load at all rather than degrading (D-097).
    """
    selected = harness_dir() if root is None else root
    current = inspect_harness(selected)
    if current.is_installed and not force:
        return current
    node = require_node()
    npm = _require_executable("npm", "it ships with Node.js; reinstall Node and try again")
    del node
    environment = environment_dir(selected)
    _marker_path(selected).unlink(missing_ok=True)
    environment.mkdir(parents=True, exist_ok=True)
    _run(
        [
            npm,
            "install",
            "--prefix",
            str(environment),
            "--save-exact",
            "--no-audit",
            "--no-fund",
            DSH_REQUIREMENT,
        ],
        f"install {DSH_REQUIREMENT}",
    )
    if not entry_script(selected).is_file():
        raise HarnessError(f"the DeepSeek Harness installation produced no {DSH_PACKAGE} entry")
    _write_marker(selected)
    return inspect_harness(selected)


def directory_size(path: Path) -> int:
    """Sum the bytes of one managed directory, so a removal can report what it frees.

    Entries that vanish or cannot be read while walking are skipped rather than failing the
    report: the number exists to inform a decision, not to be an audit.
    """
    if not path.is_dir():
        return 0
    total = 0
    for item in path.rglob("*"):
        try:
            if item.is_file() and not item.is_symlink():
                total += item.stat().st_size
        except OSError:
            continue
    return total


def _remove_confined(path: Path, root: Path) -> bool:
    """Delete one directory proven to sit inside the managed root, and report whether it existed.

    A symlink is refused rather than followed, so removal can never escape the managed tree
    (specification sections 5.10 and 5.12).
    """
    absolute, managed = path.absolute(), root.absolute()
    if absolute.parent != managed:
        raise HarnessError(f"refusing to remove a path outside the managed root: {absolute}")
    if path.is_symlink():
        raise HarnessError(f"refusing to remove a symlinked managed path: {absolute}")
    if not path.exists():
        return False
    try:
        shutil.rmtree(path)
    except OSError as error:
        raise HarnessError(f"cannot remove {absolute}: {error}") from error
    return True


def remove_environment(root: Path | None = None) -> bool:
    """Delete the managed harness installation, leaving the interface's own data untouched.

    The two are separated because they are different kinds of thing: the installation is bytes bora
    fetched and can fetch again, while the harness home is the user's sessions and workspaces.
    """
    selected = harness_dir() if root is None else root
    return _remove_confined(environment_dir(selected), selected)


def remove_interface_data(root: Path | None = None) -> bool:
    """Delete the harness home: its profiles, sessions and storages, which are user content."""
    selected = harness_dir() if root is None else root
    return _remove_confined(interface_data_dir(selected), selected)


def _provider_route(launch: HarnessLaunch) -> dict[str, object]:
    """Describe the managed llama-server as one pi-ai provider route.

    The model id is the alias `engine.lock` already makes `/v1/models` report (D-080), so the
    picker names the model without anything being provisioned into the interface.
    """
    modalities = ["text", "image"] if launch.is_vision else ["text"]
    return {
        "displayName": "bora (managed llama-server)",
        # llama-server ignores the key and the field is required, as in D-081; it is passed by
        # environment-variable reference, so no credential is ever written to disk.
        "apiKeyEnv": _CREDENTIAL_VARIABLE,
        "api": "openai-completions",
        "baseURL": f"http://{LOOPBACK_HOST}:{launch.llama_port}/v1",
        "defaultInput": modalities,
        # A hand-entered model is text-only until it says otherwise, so vstudio states its
        # modalities on the model itself rather than relying on the route fallback alone.
        "models": [{"id": launch.model_alias, "input": modalities}],
    }


def overlay_rows(launch: HarnessLaunch) -> list[dict[str, object]]:
    """Return every composition row bora decides, with the reason it decides it."""
    return [
        # The one route this distribution serves, and the model new sessions open on.
        {"id": "llm-pi-ai", "config": {"providers": {_PROVIDER_ROUTE: _provider_route(launch)}}},
        {
            "id": "agent-default-model",
            "config": {"provider": _PROVIDER_ROUTE, "model": launch.model_alias},
        },
        # The upstream cloud route would otherwise sit in the picker of a local distribution and
        # ask for an API key nobody here has.
        {"id": "llm-deepseek", "disabled": True},
        # A local distribution does not phone home: the search tool reaches a hosted endpoint.
        {"id": "web-search-deepseek", "disabled": True},
        # A second completion per turn, on the one calibrated slot, serialized behind the stream
        # the user is waiting on. Only the model-backed titler is dropped: the row that owns title
        # state stays, so a session is still named from its first prompt by the upstream fallback
        # rather than losing its title (D-097, restoring what D-094 W5 settled).
        {"id": "session-title-llm", "disabled": True},
    ]


def render_overlay(launch: HarnessLaunch) -> str:
    """Render the launch overlay as JSON, which is valid YAML and needs no serializer.

    Emitting JSON keeps every value the alias may contain correctly quoted without adding a YAML
    dependency for one generated file (AGENTS.md dependency rule).
    """
    header = (
        "# Generated by bora on every launch; edits are overwritten.\n"
        "# The interface's own layers stay untouched: this is bora's overlay, passed by argv.\n"
    )
    return header + json.dumps(overlay_rows(launch), indent=2) + "\n"


def write_overlay(launch: HarnessLaunch, root: Path | None = None) -> Path:
    """Write the launch overlay atomically inside the managed root and return its path."""
    selected = harness_dir() if root is None else root
    target = overlay_path(selected)
    temporary = target.with_suffix(".tmp")
    try:
        selected.mkdir(parents=True, exist_ok=True)
        temporary.write_text(render_overlay(launch), encoding="utf-8")
        temporary.replace(target)
    except OSError as error:
        raise HarnessError(f"cannot write the harness launch overlay {target}: {error}") from error
    return target


def serve_command(launch: HarnessLaunch, node: str, overlay: Path) -> tuple[str, ...]:
    """Build the launch command, with the launcher's own flags ahead of the web app's.

    `--patch` belongs to the launcher and the first token it does not recognize starts the app's
    arguments, so `web --patch ...` reaches the app instead and is rejected. The profile is
    therefore named explicitly rather than through the `web` alias.
    """
    return (
        node,
        str(launch.script),
        "--profile",
        "web",
        "--patch",
        str(overlay),
        "--host",
        LOOPBACK_HOST,
        "--port",
        str(launch.port),
    )


def readiness_contract(port: int) -> ReadinessContract:
    """Describe how the interface reports readiness, and how long a first start may take.

    The harness publishes no health or readiness route: its single-page application answers every
    unmatched path, so `GET /ready` and `GET /health` both return the same page and would report a
    dead server as healthy if the body were trusted. The contract therefore checks the status only,
    and the body is deliberately absent rather than invented (D-097).
    """
    return ReadinessContract(
        f"http://{LOOPBACK_HOST}:{port}{READY_PATH}",
        200,
        None,
        (404, 503),
        _STARTUP_TIMEOUT_SECONDS,
        "DeepSeek Harness",
    )


def _managed_settings(launch: HarnessLaunch) -> dict[str, str]:
    """Return every environment value bora decides, with the reason it decides it."""
    return {
        # The profiles, sessions and storages stay inside a managed root, so `uninstall` reaches
        # them (specification section 5.10).
        "DSH_HOME": str(launch.home_dir),
        _CREDENTIAL_VARIABLE: _LOCAL_CREDENTIAL,
        # Any non-empty value is upstream's authoritative hard opt-out; a local distribution does
        # not export session text, tool arguments, or workspace paths.
        "DSH_TELEMETRY_DISABLED": "1",
        # `studio` is documented as a place to talk to the model, so a session opens unable to
        # change anything rather than able to write across the directory bora happened to be
        # launched from. The interface shows the preset and the user raises it per session, which
        # is the shape every other destructive step here already has (D-097).
        "DSH_PERMISSION_MODE": _PERMISSION_MODE,
    }


def launch_environment(launch: HarnessLaunch) -> dict[str, str]:
    """Assemble the whole child environment in one place, so `doctor` can show all of it.

    `DEEPSEEK_API_KEY` is never set and an inherited one is removed: the routes that would use it
    are disabled by the overlay, and a key left in the environment would quietly re-enable a hosted
    endpoint in a distribution whose whole point is that it runs locally.
    """
    environment = dict(os.environ)
    environment.pop("DEEPSEEK_API_KEY", None)
    environment.update(_managed_settings(launch))
    return environment
