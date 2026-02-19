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
        "--data", "-d",
        help="Path to raw training data",
    ),
    output: str = typer.Option(
        "features.parquet",
        "--output", "-o",
        help="Path for processed features",
    ),
) -> None:
    """Run the Feature Pipeline."""
    from ml_project.pipelines import FeaturePipeline
    from ml_project.features import NullFiller, StandardScaler

    console.print(f"[bold blue]Running Feature Pipeline...[/]")
    
    # Example configuration
    pipeline = FeaturePipeline(processors=[
        NullFiller(strategy="mean"),
        StandardScaler()
    ])
    
    path = pipeline.run(data_path, output_name=output)
    console.print(f"[green]✓[/] Features saved to: {path}")


@app.command()
def train(
    data_path: str = typer.Option(
        "processed/features.parquet",
        "--data", "-d",
        help="Path to processed features",
    ),
    target: str = typer.Option(
        ...,
        "--target", "-t",
        help="Target column name",
    ),
    experiment: str = typer.Option(
        "default",
        "--experiment", "-e",
        help="MLflow experiment name",
    ),
) -> None:
    """Run the Training Pipeline."""
    from sklearn.ensemble import RandomForestClassifier
    from ml_project.data import load_parquet
    from ml_project.pipelines import TrainingPipeline

    console.print(f"[bold blue]Running Training Pipeline...[/]")
    
    data = load_parquet(data_path)
    pipeline = TrainingPipeline(experiment_name=experiment)
    
    run_id = pipeline.run(
        data=data,
        target_column=target,
        model=RandomForestClassifier(n_estimators=100),
        params={"n_estimators": 100, "random_state": 42}
    )
    
    console.print(f"[green]✓[/] Training complete. Run ID: [bold]{run_id}[/]")


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", "--host", "-h"),
    port: int = typer.Option(8000, "--port", "-p"),
) -> None:
    """Start the Inference Layer (FastAPI)."""
    import uvicorn
    console.print(f"[bold blue]Starting Inference API on {host}:{port}[/]")
    uvicorn.run("ml_project.api:app", host=host, port=port, reload=True)


@app.command()
def info() -> None:
    """Show project configuration."""
    table = Table(title="FTI Project Configuration")
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Version", __version__)
    table.add_row("Environment", settings.environment)
    table.add_row("MLflow URI", settings.mlflow_tracking_uri or "Default")
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
