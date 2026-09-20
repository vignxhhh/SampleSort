"""Typer entry point exposing every SampleSort command.

Command bodies import their heavy dependencies lazily so that ``--help`` stays
fast and works even when the optional learning stack is not installed.
"""

from __future__ import annotations

import logging
import os
import sys
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


@app.command("calibrate")
def calibrate(
    config_dir: ConfigDirOpt = None,
    image: Annotated[
        Path | None, typer.Option("--image", "-i", help="Fit this saved frame instead.")
    ] = None,
    output: Annotated[
        Path | None, typer.Option("--output", "-o", help="Where to save the calibration.")
    ] = None,
    save_frame: Annotated[
        Path | None, typer.Option("--save-frame", help="Also save the frame that was used.")
    ] = None,
    generate_board: Annotated[
        bool, typer.Option("--generate-board", help="Render the printable ArUco board and exit.")
    ] = False,
    board_output: Annotated[
        Path, typer.Option("--board-output", help="Where to write the board PNG.")
    ] = Path("docs/aruco_board.png"),
    dpi: Annotated[int, typer.Option("--dpi", help="Print resolution for the board.")] = 300,
) -> None:
    """Fit the camera-to-table homography from the printed ArUco board.

    In sim mode the calibration is derived analytically from the known camera
    pose, so this is only needed for real hardware. Pass ``--generate-board``
    first to print the target.
    """
    import subprocess

    cfg = _load(config_dir)
    scripts = Path(__file__).resolve().parent.parent / "scripts"

    if generate_board:
        command = [
            sys.executable,
            str(scripts / "generate_aruco_board.py"),
            "--config-dir",
            str(cfg.config_dir),
            "--output",
            str(board_output),
            "--dpi",
            str(dpi),
        ]
        raise typer.Exit(code=subprocess.call(command))

    command = [
        sys.executable,
        str(scripts / "calibrate_camera.py"),
        "--config-dir",
        str(cfg.config_dir),
    ]
    if image is not None:
        command += ["--image", str(image)]
    if output is not None:
        command += ["--output", str(output)]
    if save_frame is not None:
        command += ["--save-frame", str(save_frame)]

    raise typer.Exit(code=subprocess.call(command))


@app.command("record")
def record(
    config_dir: ConfigDirOpt = None,
    seed: SeedOpt = None,
    episodes: Annotated[
        int, typer.Option("--episodes", "-e", help="Demonstration episodes to record.")
    ] = 20,
    num_tubes: Annotated[
        int, typer.Option("--num-tubes", "-n", help="Tubes spawned per episode.")
    ] = 3,
    dataset: Annotated[
        Path | None, typer.Option("--dataset", "-d", help="Where to write the dataset.")
    ] = None,
    overwrite: Annotated[
        bool, typer.Option("--overwrite", help="Replace an existing dataset at that path.")
    ] = False,
    keep_failures: Annotated[
        bool, typer.Option("--keep-failures", help="Keep episodes whose demo failed.")
    ] = False,
) -> None:
    """Record demonstration episodes into a LeRobot dataset."""
    from samplesort.learning.record import record_episodes

    cfg = _load(config_dir, seed)
    try:
        stats = record_episodes(
            cfg,
            episodes=episodes,
            num_tubes=num_tubes,
            dataset_dir=dataset,
            overwrite=overwrite,
            seed=seed if seed is not None else cfg.seed,
            keep_failures=keep_failures,
        )
    except FileExistsError as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2) from exc

    typer.echo(f"Recorded {stats.episodes} episode(s), {stats.frames} frames")
    typer.echo(f"Successful demonstrations: {stats.successful_episodes}/{stats.episodes}")
    typer.echo(f"Dataset: {stats.dataset_dir}")
    if stats.is_empty:
        raise typer.Exit(code=1)


@app.command("train")
def train(
    config_dir: ConfigDirOpt = None,
    dataset: Annotated[
        Path | None, typer.Option("--dataset", "-d", help="Dataset to train on.")
    ] = None,
    output: Annotated[
        Path | None, typer.Option("--output", "-o", help="Where to save the checkpoint.")
    ] = None,
    steps: Annotated[int | None, typer.Option("--steps", help="Optimiser steps.")] = None,
    batch_size: Annotated[int | None, typer.Option("--batch-size", help="Batch size.")] = None,
    learning_rate: Annotated[
        float | None, typer.Option("--learning-rate", help="Adam learning rate.")
    ] = None,
    device: Annotated[
        str | None, typer.Option("--device", help="Torch device (cpu, cuda, mps).")
    ] = None,
) -> None:
    """Train an ACT policy on a recorded dataset."""
    from samplesort.learning.train import TrainingError
    from samplesort.learning.train import train as run_training

    cfg = _load(config_dir)
    try:
        result = run_training(
            cfg,
            dataset_dir=dataset,
            output_dir=output,
            steps=steps,
            batch_size=batch_size,
            learning_rate=learning_rate,
            device=device,
        )
    except TrainingError as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2) from exc

    typer.echo(f"Trained {result.steps} step(s) on {result.device}")
    typer.echo(f"Frames: {result.num_frames} from {result.num_episodes} episode(s)")
    typer.echo(f"Final loss: {result.final_loss:.4f}")
    typer.echo(f"Checkpoint: {result.checkpoint_dir}")


@app.command("evaluate")
def evaluate(
    config_dir: ConfigDirOpt = None,
    seed: SeedOpt = None,
    policy: Annotated[
        Path | None, typer.Option("--policy", "-p", help="ACT checkpoint to evaluate.")
    ] = None,
    episodes: Annotated[int, typer.Option("--episodes", "-e", help="Rollouts to run.")] = 10,
    num_tubes: Annotated[
        int, typer.Option("--num-tubes", "-n", help="Tubes spawned per rollout.")
    ] = 1,
) -> None:
    """Roll out a trained policy in simulation and report its success rate."""
    from samplesort.learning.evaluate import evaluate as run_evaluation
    from samplesort.learning.train import TrainingError

    cfg = _load(config_dir, seed)
    checkpoint = policy or cfg.pipeline.policy_path or cfg.learning.checkpoint_dir
    try:
        report = run_evaluation(
            cfg,
            checkpoint,
            episodes=episodes,
            num_tubes=num_tubes,
            seed=seed if seed is not None else cfg.seed,
        )
    except TrainingError as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2) from exc

    for line in report.summary_lines():
        typer.echo(line)


@app.command("benchmark")
def benchmark(
    config_dir: ConfigDirOpt = None,
    seed: SeedOpt = None,
    trials: Annotated[int, typer.Option("--trials", "-t", help="Seeded trials to run.")] = 20,
    num_tubes: Annotated[
        int, typer.Option("--num-tubes", "-n", help="Tubes spawned per trial.")
    ] = 6,
    output_dir: Annotated[
        Path | None, typer.Option("--output-dir", "-o", help="Where to write the artefacts.")
    ] = None,
    policy: Annotated[
        Path | None,
        typer.Option("--policy", help="Also benchmark this ACT checkpoint and compare."),
    ] = None,
    perception_noise_mm: Annotated[
        float,
        typer.Option(
            "--perception-noise",
            help="Std-dev of localisation noise to inject, in millimetres.",
        ),
    ] = 0.0,
) -> None:
    """Run seeded simulation trials and write the metrics JSON and Markdown table."""
    from samplesort.benchmark import run_benchmark, write_report

    cfg = _load(config_dir, seed)
    cfg.pipeline.perception_noise_m = perception_noise_mm / 1000.0
    destination = output_dir or cfg.output_dir

    typer.echo(f"Running {trials} scripted trial(s) of {num_tubes} tubes each...")
    scripted = run_benchmark(cfg, trials=trials, num_tubes=num_tubes, start_seed=cfg.seed)

    comparison = None
    if policy is not None:
        typer.echo(f"Running {trials} learned trial(s) with {policy}...")
        learned_cfg = cfg.model_copy(deep=True)
        learned_cfg.pipeline.control_mode = "learned"
        learned_cfg.pipeline.policy_path = policy

        def make_learned(config: SampleSortConfig, backend: Backend) -> ScriptedController:
            return _build_controller(config, backend)

        comparison = run_benchmark(
            learned_cfg,
            trials=trials,
            num_tubes=num_tubes,
            start_seed=cfg.seed,
            controller_factory=make_learned,
        )

    json_path, markdown_path = write_report(scripted, destination, comparison=comparison)
    typer.echo("")
    typer.echo(scripted.to_markdown(comparison=comparison))
    typer.echo(f"Wrote {json_path}")
    typer.echo(f"Wrote {markdown_path}")


@app.command("dashboard")
def dashboard(
    config_dir: ConfigDirOpt = None,
    port: Annotated[int, typer.Option("--port", "-p", help="Port to serve on.")] = 8501,
    host: Annotated[str, typer.Option("--host", help="Address to bind.")] = "localhost",
    headless: Annotated[
        bool, typer.Option("--headless", help="Do not try to open a browser.")
    ] = True,
) -> None:
    """Launch the Streamlit dashboard over the sort log."""
    import subprocess

    cfg = _load(config_dir)
    app_path = Path(__file__).resolve().parent.parent / "dashboard" / "app.py"
    if not app_path.is_file():
        typer.secho(f"Dashboard app not found at {app_path}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app_path),
        "--server.port",
        str(port),
        "--server.address",
        host,
        "--server.headless",
        "true" if headless else "false",
        "--browser.gatherUsageStats",
        "false",
    ]
    typer.echo(f"Serving the SampleSort dashboard on http://{host}:{port}")
    typer.echo(f"Reading the sort log at {cfg.database_path}")

    environment = dict(os.environ)
    environment["SAMPLESORT_CONFIG_DIR"] = str(cfg.config_dir)
    raise typer.Exit(code=subprocess.call(command, env=environment))


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
