"""Typer entry point exposing every SampleSort command.

Command bodies import their heavy dependencies lazily so that ``--help`` stays
fast and works even when the optional learning stack is not installed.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated

import typer

from samplesort import __version__
from samplesort.config import ConfigError, SampleSortConfig, load_config
from samplesort.logging_setup import setup_logging

logger = logging.getLogger(__name__)

app = typer.Typer(
    name="samplesort",
    help="Vision-guided robotic arm system that automates lab sample sorting.",
    no_args_is_help=True,
    add_completion=False,
)

ConfigDirOpt = Annotated[
    Path | None,
    typer.Option("--config-dir", "-c", help="Directory holding the YAML config files."),
]
SeedOpt = Annotated[int | None, typer.Option("--seed", help="RNG seed for reproducible runs.")]


def _version_callback(value: bool) -> None:
    """Print the package version and exit."""
    if value:
        typer.echo(f"samplesort {__version__}")
        raise typer.Exit


@app.callback()
def main_callback(
    version: Annotated[  # noqa: ARG001 - consumed by the eager Typer callback
        bool,
        typer.Option("--version", callback=_version_callback, is_eager=True, help="Show version."),
    ] = False,
    log_level: Annotated[
        str, typer.Option("--log-level", help="Logging level (DEBUG, INFO, WARNING, ERROR).")
    ] = "INFO",
) -> None:
    """Configure logging before any subcommand runs."""
    setup_logging(log_level)


def _load(config_dir: Path | None, seed: int | None = None) -> SampleSortConfig:
    """Load the config bundle, turning config errors into clean CLI failures."""
    try:
        return load_config(config_dir, seed=seed)
    except ConfigError as exc:
        typer.secho(f"Configuration error:\n{exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2) from exc


@app.command("config")
def show_config(config_dir: ConfigDirOpt = None) -> None:
    """Validate the configuration files and print a summary."""
    cfg = _load(config_dir)
    typer.echo(cfg.model_dump_json(indent=2))


@app.command("sim-demo")
def sim_demo(
    config_dir: ConfigDirOpt = None,
    seed: SeedOpt = None,
    num_tubes: Annotated[int, typer.Option("--num-tubes", "-n", help="Tubes to spawn.")] = 6,
    gui: Annotated[bool, typer.Option("--gui", help="Open the PyBullet GUI window.")] = False,
) -> None:
    """Build the simulated workspace and render the overhead view."""
    from samplesort.hal.factory import build_backend

    cfg = _load(config_dir, seed)
    backend = build_backend(cfg, gui=gui, seed=seed if seed is not None else cfg.seed)
    assert backend.world is not None
    with backend:
        backend.arm.go_home()
        tubes = backend.world.spawn_tubes(num_tubes)
        frame = backend.camera.read()
        typer.echo(f"Spawned {len(tubes)} tubes; camera frame {frame.shape}")
        for tube in tubes:
            x, y = backend.world.tube_xy(tube.body_id)
            typer.echo(f"  {tube.label:<7} at ({x:+.3f}, {y:+.3f})")


def main() -> None:
    """Console-script entry point."""
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
