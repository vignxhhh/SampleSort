"""Typer entry point exposing every SampleSort command.

Command bodies import their heavy dependencies lazily so that ``--help`` stays
fast and works even when the optional learning stack is not installed.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import typer

from samplesort import __version__
from samplesort.config import ConfigError, SampleSortConfig, load_config
from samplesort.logging_setup import setup_logging

if TYPE_CHECKING:  # imported lazily at runtime to keep `--help` fast
    from samplesort.control.scripted import ScriptedController
    from samplesort.hal.factory import Backend
    from samplesort.pipeline import PipelineReport

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
    """Spawn a randomised tray of tubes in simulation and sort them into racks."""
    from samplesort.hal.factory import build_backend
    from samplesort.pipeline import SortPipeline

    cfg = _load(config_dir, seed)
    resolved_seed = seed if seed is not None else cfg.seed
    backend = build_backend(cfg, gui=gui, seed=resolved_seed)
    assert backend.world is not None

    with backend:
        tubes = backend.world.spawn_tubes(num_tubes)
        typer.echo(f"Spawned {len(tubes)} tubes (seed {resolved_seed}):")
        for tube in tubes:
            x, y = backend.world.tube_xy(tube.body_id)
            typer.echo(
                f"  {tube.label:<7} at ({x:+.3f}, {y:+.3f}) -> {cfg.rack_for_class(tube.label)}"
            )

        with SortPipeline(cfg, backend) as pipeline:
            report = pipeline.run()
            typer.echo("")
            for line in report.summary_lines():
                typer.echo(line)
            typer.echo(f"  rack occupancy    : {pipeline.rack_state.summary()}")
            typer.echo(f"  sort log          : {cfg.database_path}")

    _exit_on_failure(report)


@app.command("run")
def run(
    config_dir: ConfigDirOpt = None,
    seed: SeedOpt = None,
    mode: Annotated[
        str, typer.Option("--mode", help="Control mode: scripted or learned.")
    ] = "scripted",
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Perceive and plan only; do not move.")
    ] = False,
    num_tubes: Annotated[
        int, typer.Option("--num-tubes", "-n", help="Tubes to spawn (sim mode only).")
    ] = 6,
    gui: Annotated[bool, typer.Option("--gui", help="Open the PyBullet GUI window.")] = False,
    policy: Annotated[
        Path | None, typer.Option("--policy", help="ACT checkpoint for --mode learned.")
    ] = None,
) -> None:
    """Run the sort loop against the configured backend."""
    from samplesort.hal.factory import build_backend
    from samplesort.pipeline import SortPipeline

    if mode not in ("scripted", "learned"):
        typer.secho(
            f"Unknown mode '{mode}'; expected 'scripted' or 'learned'.",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=2)

    cfg = _load(config_dir, seed)
    cfg.pipeline.control_mode = mode  # type: ignore[assignment]
    if policy is not None:
        cfg.pipeline.policy_path = policy

    backend = build_backend(cfg, gui=gui, seed=seed if seed is not None else cfg.seed)
    controller = _build_controller(cfg, backend)

    with backend:
        if backend.world is not None:
            backend.world.spawn_tubes(num_tubes)
        with SortPipeline(cfg, backend, controller=controller) as pipeline:
            report = pipeline.run(dry_run=dry_run)
            for line in report.summary_lines():
                typer.echo(line)

    _exit_on_failure(report)


def _build_controller(cfg: SampleSortConfig, backend: Backend) -> ScriptedController:
    """Build the controller the configured control mode calls for.

    Raises:
        typer.Exit: If learned mode was selected without a policy checkpoint.
    """
    from samplesort.control.scripted import ScriptedController

    if cfg.pipeline.control_mode == "scripted":
        return ScriptedController(cfg, backend.arm, backend.world)

    from samplesort.control.learned import LearnedController

    if cfg.pipeline.policy_path is None:
        typer.secho(
            "--mode learned needs a policy checkpoint: pass --policy PATH or set "
            "pipeline.policy_path in configs/default.yaml.",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=2)
    learned: ScriptedController = LearnedController(
        cfg,
        backend.arm,
        backend.camera,
        world=backend.world,
        policy_path=cfg.pipeline.policy_path,
    )
    return learned


def _exit_on_failure(report: PipelineReport) -> None:
    """Exit non-zero when a run attempted jobs and none of them succeeded.

    Raises:
        typer.Exit: With code 1 when every attempted job failed.
    """
    if report.attempted and not report.succeeded:
        raise typer.Exit(code=1)


def main() -> None:
    """Console-script entry point."""
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
