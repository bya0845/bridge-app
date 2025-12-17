"""
Generic training script that supports YAML configs and command-line overrides.

Usage:
    # T5 training with default config
    python train.py --config config/t5_training.yaml --solver t5

    # With command-line overrides
    python train.py --config config/t5_training.yaml --solver t5 --epochs 20 --batch_size 16

    # Test mode
    python train.py --config config/t5_training.yaml --solver t5 --test --checkpoint results/checkpoints/t5-small_best.pth
"""

import yaml
import os
import sys
import argparse
import tempfile
from pathlib import Path


def load_solver_class(solver_type: str):
    """Dynamically load the appropriate solver class and config class."""
    if solver_type == "t5":
        from src.parser.t5_solver import T5Solver
        from src.parser.config_schema import T5TrainingConfig

        return T5Solver, T5TrainingConfig
    else:
        raise ValueError(f"Unknown solver type: {solver_type}. Available: t5")


def merge_config_with_args(config_path: str, cli_args: dict) -> str:
    """
    Merge YAML config with command-line arguments.
    CLI arguments override config values.
    Returns path to temporary merged config file.
    """
    # Load original config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    arg_mapping = {
        "lr": "learning_rate",
        "reg": "weight_decay",
        "warmup": "warmup_steps",
    }

    modified = False
    for arg_name, arg_value in cli_args.items():
        if arg_value is not None:
            config_name = arg_mapping.get(arg_name, arg_name)
            if config.get(config_name) != arg_value:
                if int(os.environ.get("LOCAL_RANK", 0)) == 0:
                    print(f"CLI override: {config_name} = {arg_value} (was {config.get(config_name)})")
                config[config_name] = arg_value
                modified = True

    if not modified:
        return config_path

    temp_config = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
    yaml.dump(config, temp_config, default_flow_style=False)
    temp_config.close()
    return temp_config.name


def main():
    parser = argparse.ArgumentParser(
        description="Train models using YAML configs with optional CLI overrides",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train T5 with default config
  python train.py --config config/t5_training.yaml --solver t5

  # Override hyperparameters
  python train.py --config config/t5_training.yaml --solver t5 --epochs 20 --lr 3e-5

  # Test mode
  python train.py --config config/t5_training.yaml --solver t5 --test --checkpoint results/checkpoints/t5-small_best.pth
        """,
    )

    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--solver", type=str, required=True, choices=["t5"], help="Solver type (t5 for seq2seq)")

    parser.add_argument("--test", action="store_true", help="Run in test mode instead of training")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint (required for test mode)")

    parser.add_argument("--batch_size", type=int, help="Batch size")
    parser.add_argument("--lr", type=float, help="Learning rate")
    parser.add_argument("--epochs", type=int, help="Number of epochs")
    parser.add_argument("--warmup", type=int, help="Warmup steps")
    parser.add_argument("--weight_decay", type=float, help="Weight decay")
    parser.add_argument("--optimizer", type=str, help="Optimizer (adamw, adam, sgd)")
    parser.add_argument("--scheduler", type=str, help="LR scheduler")
    parser.add_argument("--mixed_precision", type=str, help="Mixed precision (no, fp16, bf16)")

    parser.add_argument("--model_name", type=str, help="Model name or path")
    parser.add_argument("--data_path", type=str, help="Path to training data")
    parser.add_argument("--val_split", type=float, help="Validation split ratio")

    parser.add_argument("--experiment_name", type=str, help="MLflow experiment name")
    parser.add_argument("--run_name", type=str, help="MLflow run name")
    parser.add_argument("--no_mlflow", action="store_true", help="Disable MLflow")

    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)

    if args.test and not args.checkpoint:
        print("Error: --checkpoint required when using --test mode")
        sys.exit(1)

    cli_overrides = {
        "batch_size": args.batch_size,
        "lr": args.lr,
        "epochs": args.epochs,
        "warmup": args.warmup,
        "weight_decay": args.weight_decay,
        "optimizer": args.optimizer,
        "scheduler": args.scheduler,
        "mixed_precision": args.mixed_precision,
        "model_name": args.model_name,
        "data_path": args.data_path,
        "val_split": args.val_split,
        "experiment_name": args.experiment_name,
        "run_name": args.run_name,
    }

    if args.no_mlflow:
        cli_overrides["mlflow"] = False

    cli_overrides = {k: v for k, v in cli_overrides.items() if v is not None}

    final_config_path = merge_config_with_args(str(config_path), cli_overrides)

    # Load solver class and config class
    SolverClass, ConfigClass = load_solver_class(args.solver)
    
    # Load and validate config with type checking
    config = ConfigClass.from_yaml(final_config_path)
    
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print("\n" + "=" * 60)
        print(f"SOLVER: {args.solver}")
        print(f"CONFIG: {args.config}")
        print(f"MODE: {'TEST' if args.test else 'TRAIN'}")
        if args.test:
            print(f"CHECKPOINT: {args.checkpoint}")
        print("=" * 60 + "\n")

    # Initialize solver with validated config object
    solver = SolverClass(config)
    mode = "test" if args.test else "train"
    solver.run(checkpoint_path=args.checkpoint, mode=mode)

    # Cleanup temporary config if created
    if final_config_path != str(config_path):
        try:
            os.unlink(final_config_path)
        except:
            pass


if __name__ == "__main__":
    main()
