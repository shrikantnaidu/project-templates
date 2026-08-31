"""
Command Line Interface for ML Project (FTI Architecture).

Commands for Feature, Training, and Inference pipelines.
"""

import typer
from rich.console import Console
from rich.table import Table

from ml_project import __version__
from ml_project.config import settings

app = typer.Typer(
    name="ml-project",
    help="Modern ML project CLI - FTI Architecture",
    add_completion=False,
)
console = Console()


@app.command()
def feature(
    data_path: str = typer.Option(
        "raw/train.csv",
        "--data",
        "-d",
        help="Path to raw training data",
    ),
    output: str = typer.Option(
        "features.parquet",
        "--output",
        "-o",
        help="Path for processed features",
    ),
    target: str | None = typer.Option(
        None,
        "--target",
        "-t",
        help="Target column to preserve without transforming",
    ),
) -> None:
    """Run the Feature Pipeline."""
    from ml_project.features import NullFiller, StandardScaler
    from ml_project.pipelines import FeaturePipeline

    console.print("[bold blue]Running Feature Pipeline...[/]")

    # Example configuration
    pipeline = FeaturePipeline(
        processors=[NullFiller(strategy="mean"), StandardScaler()]
    )

    path = pipeline.run(data_path, output_name=output, target_column=target)
    console.print(f"[green]✓[/] Features saved to: {path}")


@app.command()
def train(
    data_path: str = typer.Option(
        "raw/train.csv",
        "--data",
        "-d",
        help="Path to raw training data",
    ),
    target: str = typer.Option(
        ...,
        "--target",
        "-t",
        help="Target column name",
    ),
    experiment: str = typer.Option(
        "default",
        "--experiment",
        "-e",
        help="MLflow experiment name",
    ),
) -> None:
    """Run the Training Pipeline."""
    from sklearn.ensemble import RandomForestClassifier

    from ml_project.data import load_csv
    from ml_project.features import NullFiller, StandardScaler
    from ml_project.pipelines import FeaturePipeline, TrainingPipeline

    console.print("[bold blue]Running Training Pipeline...[/]")

    data = load_csv(data_path)
    pipeline = TrainingPipeline(experiment_name=experiment)
    feature_pipeline = FeaturePipeline(
        processors=[NullFiller(strategy="mean"), StandardScaler()]
    )

    run_id = pipeline.run(
        data=data,
        target_column=target,
        model=RandomForestClassifier(n_estimators=100, random_state=42),
        params={"n_estimators": 100, "random_state": 42},
        feature_pipeline=feature_pipeline,
    )

    console.print(f"[green]✓[/] Training complete. Run ID: [bold]{run_id}[/]")


@app.command()
def predict(
    data_path: str = typer.Option(
        "raw/test.csv",
        "--data",
        "-d",
        help="Path to prediction data",
    ),
    model_uri: str = typer.Option(
        "",
        "--model",
        "-m",
        help="MLflow model URI; defaults to the configured model and alias",
    ),
) -> None:
    """Generate predictions from a registered model."""
    from ml_project.data import load_csv
    from ml_project.pipelines import InferencePipeline

    console.print("[bold blue]Running Inference Pipeline...[/]")
    data = load_csv(data_path)
    predictions = InferencePipeline(model_uri=model_uri or None).predict(data)
    for prediction in predictions:
        console.print(prediction)


@app.command()
def serve(
    host: str = typer.Option(settings.api_host, "--host", "-h"),
    port: int = typer.Option(settings.api_port, "--port", "-p"),
    reload: bool = typer.Option(False, "--reload/--no-reload"),
) -> None:
    """Start the Inference Layer (FastAPI)."""
    import uvicorn

    console.print(f"[bold blue]Starting Inference API on {host}:{port}[/]")
    uvicorn.run("ml_project.api:app", host=host, port=port, reload=reload)


@app.command()
def info() -> None:
    """Show project configuration."""
    table = Table(title="FTI Project Configuration")
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Version", __version__)
    table.add_row("Environment", settings.environment)
    table.add_row("MLflow URI", settings.mlflow_tracking_uri or "./mlruns")
    table.add_row("Model", f"{settings.model_name} ({settings.model_alias})")
    table.add_row("Data Directory", str(settings.data_dir))

    console.print(table)


@app.command()
def init() -> None:
    """Initialize project directories."""
    settings.ensure_directories()
    console.print("[green]✓[/] Project directories created")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
