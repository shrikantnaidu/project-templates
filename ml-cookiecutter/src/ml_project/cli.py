"""
Command Line Interface for ML Project (FTI Architecture).

Commands for Feature, Training, and Inference pipelines.
"""

import json
from pathlib import Path

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
    from ml_project.features import NullFiller, StandardScaler
    from ml_project.pipelines import FeaturePipeline

    console.print("[bold blue]Running Feature Pipeline...[/]")

    pipeline = FeaturePipeline(processors=[NullFiller(strategy="mean"), StandardScaler()])

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

    console.print("[bold blue]Running Training Pipeline...[/]")

    data = load_parquet(data_path)
    pipeline = TrainingPipeline(experiment_name=experiment)

    run_id = pipeline.run(
        data=data,
        target_column=target,
        model=RandomForestClassifier(n_estimators=200, random_state=42),
        params={"n_estimators": 200, "random_state": 42},
    )

    console.print(f"[green]✓[/] Training complete. Run ID: [bold]{run_id}[/]")
    console.print(f"[green]✓[/] Model URI: [bold]runs:/{run_id}/model[/]")


@app.command()
def predict(
    data_path: str = typer.Option(
        "processed/churn_features.parquet",
        "--data",
        "-d",
        help="Path to feature data for inference",
    ),
    model_uri: str = typer.Option(
        ...,
        "--model-uri",
        "-m",
        help="MLflow model URI (e.g. runs:/<run_id>/model)",
    ),
    output: str = typer.Option(
        "predictions.json",
        "--output",
        "-o",
        help="Path to write predictions JSON",
    ),
) -> None:
    """Generate batch predictions from a parquet feature file."""
    from ml_project.data import TARGET_COLUMN, load_parquet
    from ml_project.pipelines import InferencePipeline

    console.print("[bold blue]Running Batch Prediction...[/]")

    df = load_parquet(data_path)
    if TARGET_COLUMN in df.columns:
        df = df.drop(columns=[TARGET_COLUMN])

    pipeline = InferencePipeline(model_uri=model_uri)
    predictions = pipeline.predict(df)

    output_path = Path(output)
    if not output_path.is_absolute():
        output_path = Path.cwd() / output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps({"predictions": predictions.tolist()}, indent=2))
    console.print(f"[green]✓[/] Predictions saved to: {output_path}")


@app.command("prepare-churn-demo")
def prepare_churn_demo() -> None:
    """Download and prepare the Telco churn dataset for the showcase demo."""
    from ml_project.data import download_telco_dataset, prepare_churn_features

    console.print("[bold blue]Preparing churn showcase dataset...[/]")
    raw_path = download_telco_dataset()
    features_path = prepare_churn_features()
    console.print(f"[green]✓[/] Raw data: {raw_path}")
    console.print(f"[green]✓[/] Feature dataset: {features_path}")


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
