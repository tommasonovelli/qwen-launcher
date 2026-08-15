"""Tests for the `bora ui` commands and for keeping every credential out of every output."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from typer.testing import CliRunner

import bora_workbench._cli_diagnostics as diagnostics_cli
import bora_workbench._cli_harness as harness_cli
import bora_workbench._cli_services as service_cli
import bora_workbench.snapshot as snapshot_module
from bora_workbench.cli import app
from bora_workbench.harness import DSH_VERSION, HarnessError, render_overlay
from bora_workbench.harness import HarnessLaunch as Launch
from tests.test_cli_update import flat

runner = CliRunner()


def _status(installed: bool, root: Path = Path("/managed/deepseek-harness")) -> SimpleNamespace:
    """Return one inspection result without touching the filesystem."""
    return SimpleNamespace(is_installed=installed, root=root, script=None)


def test_status_reports_an_absent_installation_and_the_remedy(monkeypatch) -> None:
    """Tell a user who never installed it exactly which command changes that."""
    monkeypatch.setattr(harness_cli, "inspect_harness", lambda: _status(False))

    result = runner.invoke(app, ["ui", "status"])

    assert result.exit_code == 0
    assert "not installed" in result.stdout
    assert "bora ui install" in result.stdout
    assert "3080" in result.stdout


def test_status_reports_the_pinned_version_when_present(monkeypatch) -> None:
    """Name the exact release, so two machines can be compared without guessing."""
    monkeypatch.setattr(harness_cli, "inspect_harness", lambda: _status(True))

    result = runner.invoke(app, ["ui", "status"])

    assert result.exit_code == 0
    assert DSH_VERSION in result.stdout


def test_this_interface_needs_no_stored_secret_at_all(monkeypatch) -> None:
    """Store nothing: the harness signs no session cookie, so there is no key to keep or leak."""
    monkeypatch.setattr(harness_cli, "inspect_harness", lambda: _status(True))

    status = runner.invoke(app, ["ui", "status"])
    doctor = runner.invoke(app, ["doctor"])

    overlay = render_overlay(Launch(Path("bin.js"), 3080, 8080, Path("/home"), "Qwen 3.6"))
    assert "bora-local" not in overlay
    assert "bora-local" not in status.stdout
    assert "bora-local" not in doctor.stdout


def test_install_refuses_while_a_managed_service_is_running(monkeypatch) -> None:
    """Never replace an environment a live interface still has open."""
    live = SimpleNamespace(services=(SimpleNamespace(pid=1),), warnings=(), stopped=())
    monkeypatch.setattr(service_cli, "service_roots", lambda: (Path("state"),))
    monkeypatch.setattr(service_cli, "status_services", lambda root: live)
    monkeypatch.setattr(
        harness_cli, "install_harness", lambda **kwargs: _forbidden("install must not run")
    )

    result = runner.invoke(app, ["ui", "install"])

    assert result.exit_code == 1
    assert "run bora stop" in flat(result.stderr)


def _forbidden(message: str) -> None:
    """Raise inside a lambda so a forbidden call fails the test where it happens."""
    raise AssertionError(message)


def _no_services(monkeypatch) -> None:
    """Report an idle machine so an installation is allowed to proceed."""
    empty = SimpleNamespace(services=(), warnings=(), stopped=())
    monkeypatch.setattr(service_cli, "service_roots", lambda: (Path("state"),))
    monkeypatch.setattr(service_cli, "status_services", lambda root: empty)


def test_install_states_the_cost_and_whose_program_it_is(monkeypatch) -> None:
    """Say what the command spends and what it starts, before it spends it."""
    _no_services(monkeypatch)
    monkeypatch.setattr(harness_cli, "inspect_harness", lambda: _status(False))
    monkeypatch.setattr(harness_cli, "install_harness", lambda **kwargs: _status(True))

    result = runner.invoke(app, ["ui", "install"])

    assert result.exit_code == 0
    assert "360 MB" in result.stdout
    assert "separate program" in result.stdout
    assert DSH_VERSION in result.stdout


def test_install_maps_a_failure_to_exit_one_without_a_traceback(monkeypatch) -> None:
    """Report an actionable installation failure the way every other command does."""
    _no_services(monkeypatch)
    monkeypatch.setattr(harness_cli, "inspect_harness", lambda: _status(False))

    def fail(**kwargs):
        """Fail the installation the way a missing prerequisite would."""
        del kwargs
        raise HarnessError("node is required to install DeepSeek Harness")

    monkeypatch.setattr(harness_cli, "install_harness", fail)

    result = runner.invoke(app, ["ui", "install"])

    assert result.exit_code == 1
    assert "node is required" in flat(result.stderr)
    assert "Traceback" not in result.stderr


def test_doctor_names_the_interface_state_and_its_port(monkeypatch) -> None:
    """Show on one line whether studio will open the harness or the built-in interface."""
    monkeypatch.setattr(snapshot_module, "inspect_harness", lambda: _status(False))

    result = runner.invoke(app, ["doctor"])

    assert result.exit_code == 0
    assert "not installed; studio opens the integrated interface" in flat(result.stdout)
    assert "3080" in result.stdout


def test_engine_install_acquires_the_interface_by_default(monkeypatch) -> None:
    """Put the interface where a first setup already spends gigabytes and waits."""
    acquired: list[bool] = []
    monkeypatch.setattr(
        diagnostics_cli, "install_managed_harness", lambda force, stdout: acquired.append(force)
    )
    _stub_engine(monkeypatch)

    result = runner.invoke(app, ["engine", "install", "--no-model"])

    assert result.exit_code == 0
    assert acquired == [False]


def test_no_ui_declines_the_interface_without_removing_anything(monkeypatch) -> None:
    """Leave a machine that only wants the API free of a Node dependency tree."""
    monkeypatch.setattr(
        diagnostics_cli,
        "install_managed_harness",
        lambda force, stdout: _forbidden("--no-ui must acquire nothing"),
    )
    _stub_engine(monkeypatch)

    result = runner.invoke(app, ["engine", "install", "--no-model", "--no-ui"])

    assert result.exit_code == 0


def _stub_engine(monkeypatch) -> None:
    """Replace engine acquisition so no build, download, or hardware probe happens."""
    hardware = SimpleNamespace(backend="cpu", warnings=())
    status = SimpleNamespace(
        is_active=True, release="b10011", backend="cpu", executable=Path("server"), differences=()
    )
    monkeypatch.setattr(diagnostics_cli, "detect_hardware", lambda: hardware)
    monkeypatch.setattr(
        diagnostics_cli,
        "_install_with_progress",
        lambda backend, force, stdout: SimpleNamespace(status=status, is_installed=True),
    )
    monkeypatch.setattr(diagnostics_cli, "_report_install", lambda result, stdout: None)


def test_remove_frees_the_installation_and_asks_about_sessions_separately(monkeypatch) -> None:
    """Ask two questions, because reinstallable bytes and a user's own content differ."""
    removed: list[str] = []
    root = Path("/managed/deepseek-harness")
    monkeypatch.setattr(harness_cli, "inspect_harness", lambda: _status(True, root))
    monkeypatch.setattr(harness_cli, "directory_size", lambda path: 1024)
    monkeypatch.setattr(Path, "is_dir", lambda self: True)
    monkeypatch.setattr(harness_cli, "remove_environment", lambda: removed.append("environment"))
    monkeypatch.setattr(harness_cli, "remove_interface_data", lambda: removed.append("data"))
    _no_services(monkeypatch)

    result = runner.invoke(app, ["ui", "remove"], input="y\nn\n")

    assert result.exit_code == 0
    assert "Remove the managed DeepSeek Harness installation?" in result.stdout
    assert "Also remove the harness home?" in result.stdout
    assert "sessions, workspaces" in result.stdout
    assert removed == ["environment"]


def test_declining_both_questions_removes_nothing(monkeypatch) -> None:
    """Keep a default of no on both questions, so an accidental Enter deletes nothing."""
    monkeypatch.setattr(harness_cli, "inspect_harness", lambda: _status(True))
    monkeypatch.setattr(harness_cli, "directory_size", lambda path: 1024)
    monkeypatch.setattr(Path, "is_dir", lambda self: True)
    monkeypatch.setattr(
        harness_cli, "remove_environment", lambda: _forbidden("nothing may be removed")
    )
    monkeypatch.setattr(
        harness_cli, "remove_interface_data", lambda: _forbidden("nothing may be removed")
    )
    _no_services(monkeypatch)

    result = runner.invoke(app, ["ui", "remove"], input="\n\n")

    assert result.exit_code == 0
    assert "Installation kept." in result.stdout
    assert "Harness home kept" in result.stdout


def test_remove_refuses_while_a_managed_service_is_running(monkeypatch) -> None:
    """Never delete an environment a live interface still has open."""
    live = SimpleNamespace(services=(SimpleNamespace(pid=1),), warnings=(), stopped=())
    monkeypatch.setattr(service_cli, "service_roots", lambda: (Path("state"),))
    monkeypatch.setattr(service_cli, "status_services", lambda root: live)

    result = runner.invoke(app, ["ui", "remove"])

    assert result.exit_code == 1
    assert "run bora stop" in flat(result.stderr)


def test_uninstall_names_the_interface_content_it_deletes(monkeypatch) -> None:
    """Say that a full uninstall takes the sessions with it, since nothing asks about them."""
    _no_services(monkeypatch)
    monkeypatch.setattr(
        service_cli,
        "inspect_tool_installation",
        lambda: SimpleNamespace(is_managed_by_uv=False, environment=Path("tool")),
    )

    result = runner.invoke(app, ["uninstall"], input="n\n")

    assert "DeepSeek Harness" in flat(result.stdout)
    assert "bora ui remove" in flat(result.stdout)
