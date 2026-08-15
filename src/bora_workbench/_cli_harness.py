"""Present the commands that own the managed DeepSeek Harness installation.

The interface arrives with the engine, because `bora engine install` is already the step where a
first setup spends gigabytes and waits. Removing it is a separate command, and it asks about the
installation and about the user's own sessions as two different questions, because they are two
different kinds of thing (D-096, kept by D-097).
"""

from __future__ import annotations

import typer
from rich.console import Console

from bora_workbench._cli_services import require_services_stopped
from bora_workbench._cli_theme import (
    format_bytes,
    print_error,
    print_heading,
    print_note,
    print_success,
    print_warning,
)
from bora_workbench.config import ConfigError, load_config
from bora_workbench.harness import (
    DSH_REQUIREMENT,
    DSH_VERSION,
    HarnessError,
    HarnessStatus,
    directory_size,
    environment_dir,
    inspect_harness,
    install_harness,
    interface_data_dir,
    remove_environment,
    remove_interface_data,
)


def _show_status(status: HarnessStatus, stdout: Console) -> None:
    """Describe the installation without naming anything the user did not ask about."""
    installed = f"{DSH_VERSION} installed" if status.is_installed else "not installed"
    print_note(stdout, "DeepSeek Harness", installed)
    print_note(stdout, "Installation", str(environment_dir(status.root)))
    print_note(stdout, "Harness home", str(interface_data_dir(status.root)))


def show_harness_status(stdout: Console, stderr: Console) -> None:
    """Report whether the pinned interface is installed, and on which port it would listen."""
    try:
        config = load_config()
    except ConfigError as error:
        print_error(stderr, "Configuration error", str(error))
        raise typer.Exit(code=2) from error
    status = inspect_harness()
    print_heading(stdout, "Managed DeepSeek Harness")
    _show_status(status, stdout)
    print_note(stdout, "Port", str(config.ui_port))
    if not status.is_installed:
        stdout.print("Run `bora ui install` to use DeepSeek Harness in studio and vstudio.")


def install_managed_harness(force: bool, stdout: Console) -> HarnessStatus:
    """Install the pinned interface, showing npm's own progress and what the download costs.

    Shared with `engine install`, so the interface arrives the same way and says the same thing
    whether it is acquired with the engine or on its own.
    """
    current = inspect_harness()
    if current.is_installed and not force:
        print_success(stdout, "DeepSeek Harness", f"{DSH_VERSION} is already installed")
        return current
    print_heading(stdout, "Installing DeepSeek Harness")
    stdout.print(f"Installing {DSH_REQUIREMENT} into {environment_dir(current.root)}")
    stdout.print("This downloads about 360 MB and needs Node.js 22.19 or newer on PATH.")
    stdout.print("One dependency has no Linux prebuilt binary and is compiled during install,")
    stdout.print("so a C/C++ toolchain must be available there.")
    stdout.print("DeepSeek Harness is a separate program; bora starts it and never modifies it.")
    status = install_harness(force=force)
    print_success(stdout, "DeepSeek Harness", f"{DSH_VERSION} installed")
    return status


def run_harness_install(force: bool, stdout: Console, stderr: Console) -> None:
    """Install the interface on its own, refusing while a service still holds it open."""
    require_services_stopped("Install error", stderr)
    try:
        status = install_managed_harness(force, stdout)
    except HarnessError as error:
        print_error(stderr, "Install error", str(error))
        raise typer.Exit(code=1) from error
    _show_status(status, stdout)


def _offer_environment(status: HarnessStatus, stdout: Console) -> None:
    """Ask about the reinstallable bytes first, naming what the answer frees."""
    environment = environment_dir(status.root)
    if not environment.is_dir():
        stdout.print("No managed DeepSeek Harness installation is present.")
        return
    size = format_bytes(directory_size(environment))
    stdout.print(f"Installation: {environment} ({size})")
    stdout.print("Removing it frees that space; `bora ui install` puts it back.")
    if not typer.confirm("Remove the managed DeepSeek Harness installation?", default=False):
        stdout.print("Installation kept.")
        return
    if remove_environment():
        print_success(stdout, "Removed", f"{size} freed")


def _offer_interface_data(status: HarnessStatus, stdout: Console) -> None:
    """Ask about the user's own content separately, because deleting it is not reversible.

    This is the question D-079 established for weights: content a user made gets its own request
    rather than being swept up in another one.
    """
    data = interface_data_dir(status.root)
    if not data.is_dir():
        return
    size = format_bytes(directory_size(data))
    stdout.print(f"Harness home: {data} ({size})")
    stdout.print("This is your own content: sessions, workspaces, and settings. It is not backed")
    stdout.print("up anywhere and removing it cannot be undone.")
    if not typer.confirm("Also remove the harness home?", default=False):
        stdout.print("Harness home kept; a later install finds your sessions where they were.")
        return
    if remove_interface_data():
        print_success(stdout, "Removed", f"{size} freed")


def run_harness_removal(stdout: Console, stderr: Console) -> None:
    """Remove the managed interface, asking about its installation and its content separately."""
    require_services_stopped("Removal error", stderr)
    status = inspect_harness()
    print_heading(stdout, "Remove DeepSeek Harness")
    try:
        _offer_environment(status, stdout)
        _offer_interface_data(status, stdout)
    except HarnessError as error:
        print_error(stderr, "Removal error", str(error))
        raise typer.Exit(code=1) from error
    except (KeyboardInterrupt, typer.Abort) as error:
        print_warning(stderr, f"Removal cancelled: {error}")
        raise typer.Exit(code=130) from error
