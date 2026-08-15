"""Tests for the managed DeepSeek Harness installation, overlay, environment, and readiness.

Nothing here installs, downloads, or starts anything: `npm` and `node` are replaced by recorders,
so the tests assert on the commands, the overlay, and the environment bora would produce rather
than on a real interface.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import pytest

import bora_workbench.harness as harness_module
from bora_workbench.harness import (
    DSH_PACKAGE,
    DSH_REQUIREMENT,
    DSH_VERSION,
    HarnessError,
    HarnessLaunch,
    inspect_harness,
    install_harness,
    launch_environment,
    overlay_rows,
    readiness_contract,
    render_overlay,
    serve_command,
    write_overlay,
)


def _launch(*, port: int = 3080, llama_port: int = 8080, vision: bool = False) -> HarnessLaunch:
    """Build one resolved launch without touching the filesystem."""
    return HarnessLaunch(
        Path("/managed/node/node_modules/@deepseek-ai/dsh/lib/bin.js"),
        port,
        llama_port,
        Path("/managed/home"),
        "Qwen 3.6",
        vision,
    )


def _install_tree(root: Path, *, version: str = DSH_VERSION) -> Path:
    """Create the exact shape a completed installation leaves behind."""
    package = root / "node" / "node_modules" / DSH_PACKAGE / "lib"
    package.mkdir(parents=True)
    script = package / "bin.js"
    script.write_text("#!/usr/bin/env node\n", encoding="utf-8")
    (root / "node" / "installed.json").write_text(
        json.dumps({"version": version, "package": DSH_PACKAGE}), encoding="utf-8"
    )
    return script


def _row(rows: list[dict[str, object]], identifier: str) -> dict[str, object]:
    """Return the one overlay row with the given id, failing loudly when it is absent."""
    matches = [row for row in rows if row["id"] == identifier]
    assert len(matches) == 1, f"expected exactly one {identifier!r} row, got {len(matches)}"
    return matches[0]


def test_serve_command_always_binds_loopback() -> None:
    """Never read a bind host from anywhere, so no configuration key can widen it."""
    command = serve_command(_launch(port=9090), "/usr/bin/node", Path("/managed/bora.patch.yml"))

    assert command[-4:] == ("--host", "127.0.0.1", "--port", "9090")
    assert "0.0.0.0" not in command


def test_launcher_flags_precede_the_web_app_arguments() -> None:
    """Name the profile explicitly: after the `web` alias, `--patch` would reach the app instead.

    The launcher stops parsing at the first token it does not recognize, so `web --patch ...` is
    rejected by the web app rather than applied as an overlay.
    """
    command = serve_command(_launch(), "/usr/bin/node", Path("/managed/bora.patch.yml"))

    assert command[2:6] == ("--profile", "web", "--patch", "/managed/bora.patch.yml")
    assert command.index("--patch") < command.index("--host")
    assert "web" not in command[:2]


def test_the_command_runs_the_module_rather_than_a_shim() -> None:
    """Launch through Node itself, because the Windows shim is a batch file needing a shell."""
    command = serve_command(_launch(), "/usr/bin/node", Path("/managed/bora.patch.yml"))

    assert command[0] == "/usr/bin/node"
    assert command[1].endswith("lib/bin.js")


def test_readiness_checks_the_status_only_because_no_health_route_exists() -> None:
    """Trust the status alone: every unmatched path returns the same page, including `/health`."""
    contract = readiness_contract(3080)

    assert contract.url == "http://127.0.0.1:3080/"
    assert contract.ready_status == 200
    assert contract.ready_body is None
    assert 404 in contract.transient_statuses


def test_environment_carries_every_settled_value() -> None:
    """Assemble the whole configuration in one place, including each switch with a reason."""
    environment = launch_environment(_launch())

    assert environment["DSH_HOME"] == str(Path("/managed/home"))
    assert environment["DSH_TELEMETRY_DISABLED"] == "1"
    assert environment["DSH_PERMISSION_MODE"] == "read-only"
    assert environment["BORA_LOCAL_KEY"]


def test_an_inherited_cloud_key_is_removed_from_the_child(monkeypatch) -> None:
    """Never let a key in the shell quietly re-enable a hosted endpoint in a local distribution."""
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-inherited")

    environment = launch_environment(_launch())

    assert "DEEPSEEK_API_KEY" not in environment


def test_the_environment_inherits_the_parent_so_the_child_can_start(monkeypatch) -> None:
    """Add the managed settings to the inherited environment instead of replacing it.

    A process needs more than the variables bora chooses: Windows cannot start one at all without
    `SystemRoot`, and Node needs `PATH` to find its own runtime.
    """
    monkeypatch.setenv("BORA_TEST_INHERITED", "kept")

    environment = launch_environment(_launch())

    assert environment["BORA_TEST_INHERITED"] == "kept"
    assert set(os.environ) - {"DEEPSEEK_API_KEY"} <= set(environment)


def test_the_overlay_points_the_only_route_at_the_managed_engine() -> None:
    """Serve exactly one provider, on loopback, at the configured engine port."""
    route = _row(overlay_rows(_launch(llama_port=8080)), "llm-pi-ai")

    provider = route["config"]["providers"]["bora"]
    assert provider["baseURL"] == "http://127.0.0.1:8080/v1"
    assert provider["api"] == "openai-completions"
    assert provider["apiKeyEnv"] == "BORA_LOCAL_KEY"


def test_no_credential_value_is_ever_written_into_the_overlay() -> None:
    """Pass the key by environment-variable reference, so nothing secret reaches the disk."""
    text = render_overlay(_launch())

    assert "BORA_LOCAL_KEY" in text
    assert "bora-local" not in text


def test_the_model_named_is_the_alias_the_engine_already_reports() -> None:
    """Name the model with D-080's alias, so nothing has to be provisioned into the interface."""
    rows = overlay_rows(_launch())

    assert _row(rows, "agent-default-model")["config"] == {
        "provider": "bora",
        "model": "Qwen 3.6",
    }
    assert _row(rows, "llm-pi-ai")["config"]["providers"]["bora"]["models"] == [
        {"id": "Qwen 3.6", "input": ["text"]}
    ]


def test_only_vstudio_declares_the_image_modality() -> None:
    """Keep a text mode text-only: an image is refused before it is sent, naming the model."""
    text_route = _row(overlay_rows(_launch(vision=False)), "llm-pi-ai")
    vision_route = _row(overlay_rows(_launch(vision=True)), "llm-pi-ai")

    assert text_route["config"]["providers"]["bora"]["defaultInput"] == ["text"]
    assert vision_route["config"]["providers"]["bora"]["defaultInput"] == ["text", "image"]
    assert vision_route["config"]["providers"]["bora"]["models"][0]["input"] == ["text", "image"]


def test_every_route_that_would_leave_this_machine_is_disabled() -> None:
    """Disable the hosted routes rather than relying on a key merely being absent."""
    rows = overlay_rows(_launch())

    assert _row(rows, "llm-deepseek")["disabled"] is True
    assert _row(rows, "web-search-deepseek")["disabled"] is True


def test_a_session_opens_unable_to_change_anything() -> None:
    """Open on the preset that asks, not the one that may write the launch directory."""
    assert launch_environment(_launch())["DSH_PERMISSION_MODE"] == "read-only"


def test_the_extra_completion_per_turn_is_disabled_without_losing_titles() -> None:
    """Keep the single calibrated slot for the stream the user is waiting on.

    Only the model-backed titler is dropped. The row owning title state stays, so upstream still
    names a session from its first prompt through its own fallback.
    """
    rows = overlay_rows(_launch())

    assert _row(rows, "session-title-llm")["disabled"] is True
    assert not [row for row in rows if row["id"] == "session-title"]


def test_the_overlay_is_valid_json_so_it_needs_no_yaml_serializer() -> None:
    """Emit JSON, which YAML is a superset of, so every alias is quoted without a dependency."""
    text = render_overlay(_launch())
    body = "\n".join(line for line in text.splitlines() if not line.startswith("#"))

    assert json.loads(body) == overlay_rows(_launch())


def test_the_overlay_is_written_inside_the_managed_root(tmp_path) -> None:
    """Keep bora's own file in bora's own root, never in the interface's storage."""
    written = write_overlay(_launch(), tmp_path)

    assert written == tmp_path / "bora.patch.yml"
    assert written.read_text(encoding="utf-8") == render_overlay(_launch())


def test_rewriting_the_overlay_replaces_it_without_leaving_a_temporary(tmp_path) -> None:
    """Rewrite on every launch, so a stale port or alias can never survive into a new run."""
    write_overlay(_launch(llama_port=8080), tmp_path)
    write_overlay(_launch(llama_port=9999), tmp_path)

    assert "9999" in (tmp_path / "bora.patch.yml").read_text(encoding="utf-8")
    assert list(tmp_path.glob("*.tmp")) == []


def test_absent_installation_is_reported_as_not_installed(tmp_path) -> None:
    """Report a machine that never ran the install without creating anything on it."""
    status = inspect_harness(tmp_path / "deepseek-harness")

    assert not status.is_installed
    assert status.version is None
    assert not (tmp_path / "deepseek-harness").exists()


def test_a_different_recorded_version_is_not_the_pinned_one(tmp_path) -> None:
    """Treat a tree left by another release as absent rather than as usable."""
    _install_tree(tmp_path, version="0.1.0-rc.5")

    assert not inspect_harness(tmp_path).is_installed


def test_a_complete_installation_is_reported_with_its_entry_script(tmp_path) -> None:
    """Require both the recorded version and a real entry module before claiming installed."""
    script = _install_tree(tmp_path)

    status = inspect_harness(tmp_path)

    assert status.is_installed
    assert status.version == DSH_VERSION
    assert status.script == script


def _record_npm(monkeypatch, tmp_path, *, fail: bool = False) -> list[list[str]]:
    """Replace node and npm with recorders that create the tree the installer verifies."""
    commands: list[list[str]] = []

    class _Result:
        """Stand in for a completed process with only the fields the installer reads."""

        def __init__(self, returncode: int, stdout: str = "") -> None:
            """Retain the exit status and any version output the installer inspects."""
            self.returncode = returncode
            self.stdout = stdout

    def run(command, **kwargs):
        """Record one step, answering the version probe and simulating the install."""
        del kwargs
        if command[1:] == ["--version"]:
            return _Result(0, "v22.23.0\n")
        commands.append(list(command))
        if fail:
            return _Result(1)
        _install_tree(tmp_path)
        return _Result(0)

    monkeypatch.setattr(harness_module.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(harness_module.subprocess, "run", run)
    return commands


def test_install_pins_the_exact_version_and_never_resolves_a_range(tmp_path, monkeypatch) -> None:
    """Install one pinned release, so two machines running the same bora get the same interface."""
    commands = _record_npm(monkeypatch, tmp_path)

    status = install_harness(tmp_path)

    assert status.is_installed
    assert commands[0][:2] == ["/usr/bin/npm", "install"]
    assert DSH_REQUIREMENT in commands[0]
    assert "latest" not in " ".join(commands[0])
    assert "--save-exact" in commands[0]


def test_install_runs_scripts_because_a_skipped_build_cannot_boot(tmp_path, monkeypatch) -> None:
    """Never pass `--ignore-scripts`: the harness mounts a native plugin at boot and fails hard.

    A tree installed without them is not a degraded interface, it is one whose whole plugin tree
    refuses to load, so the flag would turn a working install into an unbootable one (D-097).
    """
    commands = _record_npm(monkeypatch, tmp_path)

    install_harness(tmp_path)

    assert "--ignore-scripts" not in commands[0]


def test_an_interrupted_install_leaves_nothing_that_reports_as_installed(
    tmp_path, monkeypatch
) -> None:
    """Write the version marker last, so a failed step cannot be mistaken for a usable one."""
    _record_npm(monkeypatch, tmp_path, fail=True)

    with pytest.raises(HarnessError, match="install"):
        install_harness(tmp_path)

    assert not inspect_harness(tmp_path).is_installed


def test_install_is_idempotent_and_runs_nothing_when_already_pinned(tmp_path, monkeypatch) -> None:
    """Skip a several-hundred-megabyte reinstall when the pinned version is already present."""
    _install_tree(tmp_path)
    commands = _record_npm(monkeypatch, tmp_path)

    status = install_harness(tmp_path)

    assert status.is_installed
    assert commands == []


def test_install_without_node_names_the_missing_prerequisite(tmp_path, monkeypatch) -> None:
    """Report the one prerequisite by name instead of failing inside a subprocess call."""
    monkeypatch.setattr(harness_module.shutil, "which", lambda name: None)

    with pytest.raises(HarnessError, match="node is required"):
        install_harness(tmp_path)


def test_an_old_node_is_refused_before_anything_is_downloaded(tmp_path, monkeypatch) -> None:
    """Refuse an unsupported runtime up front, rather than after several hundred megabytes."""

    class _Result:
        """Stand in for the version probe alone."""

        returncode = 0
        stdout = "v20.11.0\n"

    monkeypatch.setattr(harness_module.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(harness_module.subprocess, "run", lambda command, **kwargs: _Result())

    with pytest.raises(HarnessError, match=re.escape("22.19.0 or newer")):
        install_harness(tmp_path)


def test_removing_the_installation_keeps_the_harness_home(tmp_path) -> None:
    """Free the reinstallable bytes without touching what the user wrote."""
    _install_tree(tmp_path)
    home = tmp_path / "home"
    home.mkdir()
    (home / "settings.yaml").write_text("sessions", encoding="utf-8")

    assert harness_module.remove_environment(tmp_path)

    assert not (tmp_path / "node").exists()
    assert (home / "settings.yaml").read_text(encoding="utf-8") == "sessions"
    assert not inspect_harness(tmp_path).is_installed


def test_removing_the_harness_home_keeps_the_installation(tmp_path) -> None:
    """Keep the two removals independent, because they delete different kinds of thing."""
    _install_tree(tmp_path)
    (tmp_path / "home").mkdir()

    assert harness_module.remove_interface_data(tmp_path)

    assert not (tmp_path / "home").exists()
    assert inspect_harness(tmp_path).is_installed


def test_removal_is_idempotent_on_an_absent_installation(tmp_path) -> None:
    """Report nothing removed rather than failing, so a repeated removal stays safe."""
    assert not harness_module.remove_environment(tmp_path)
    assert not harness_module.remove_interface_data(tmp_path)


@pytest.mark.skipif(os.name == "nt", reason="symlink creation needs a privileged Windows account")
def test_a_symlinked_managed_path_is_refused_rather_than_followed(tmp_path) -> None:
    """Never let a removal escape the managed tree by following a link out of it."""
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "precious.txt").write_text("keep", encoding="utf-8")
    root = tmp_path / "deepseek-harness"
    root.mkdir()
    (root / "node").symlink_to(outside, target_is_directory=True)

    with pytest.raises(HarnessError, match="symlinked"):
        harness_module.remove_environment(root)

    assert (outside / "precious.txt").exists()


def test_directory_size_reports_what_a_removal_would_free(tmp_path) -> None:
    """Measure the managed tree so a removal can name the space it reclaims."""
    nested = tmp_path / "node" / "node_modules"
    nested.mkdir(parents=True)
    (nested / "a.bin").write_bytes(b"x" * 100)
    (nested / "b.bin").write_bytes(b"y" * 40)

    assert harness_module.directory_size(tmp_path) == 140
    assert harness_module.directory_size(tmp_path / "absent") == 0
