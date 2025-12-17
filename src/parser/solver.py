"""
Generic Solver for training deep learning models with MLflow tracking and distributed training support.

Supports different model types (seq2seq, vision, etc.) through inheritance.

All hyperparameters are configured via YAML files with Pydantic validation.
"""

import time
import copy
import pathlib
import logging
import torch
import mlflow
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple
from datetime import datetime
from accelerate import Accelerator
from torch.utils.data import DataLoader
from config.logging_config import configure_logger
from config_schema import BaseTrainingConfig

logger = logging.getLogger(__name__)
configure_logger(log_level="INFO", logger=logger)


class AverageMeter:
    """Computes and stores the average and current value."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class BaseSolver(ABC):
    """
    Abstract base class for training models with MLflow and Accelerate support.

    Subclasses must implement:
        - _load_model(): Load and return the model
        - _load_dataset(): Create train/val dataloaders
        - _train_step(epoch): Execute one training epoch
        - _evaluate(epoch): Evaluate the model
        - _compute_metrics(outputs, targets): Compute task-specific metrics
    """

    def __init__(self, config: BaseTrainingConfig):
        """
        Initialize solver from validated config.
        
        Args:
            config: Validated configuration object (BaseTrainingConfig or subclass)
        """
        self.config = config
        
        self.accelerator = Accelerator(
            mixed_precision=config.mixed_precision,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            log_with=config.log_with if config.mlflow else None,
            project_dir=config.output_dir,
        )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            self.accelerator.print(
                f"Available GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB"
            )

        self.batch_size = config.batch_size
        self.epochs = config.epochs
        self.lr = config.learning_rate
        self.weight_decay = config.weight_decay
        self.warmup_steps = config.warmup_steps
        self.max_grad_norm = config.max_grad_norm

        self.eval_steps = config.eval_steps
        self.save_steps = config.save_steps
        self.save_total_limit = config.save_total_limit
        self.save_best = config.save_best
        self.load_best_model_at_end = config.load_best_model_at_end

        self.output_dir = pathlib.Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.checkpoint_dir.mkdir(exist_ok=True)

        self.model_type = config.model_name
        self.dataset_name = config.dataset

        self.best_metric = 0.0
        self.best_epoch = 0
        self.best_model = None
        self.epoch_times = []
        self.training_start_time = None
        self.global_step = 0

        self.accelerator.print("Loading model...")
        self.model = self._load_model()

        self.accelerator.print("Loading dataset...")
        self.train_loader, self.val_loader = self._load_dataset()

        self.accelerator.print("Setting up optimizer and scheduler...")
        self.optimizer = self._load_optimizer()
        self.scheduler = self._load_scheduler()

        self.param_count = self._get_params()
        if self.param_count is not None:
            self.accelerator.print(f"Trainable parameters: {self.param_count:,}")

        self.model, self.optimizer, self.scheduler, self.train_loader, self.val_loader = self.accelerator.prepare(
            self.model, self.optimizer, self.scheduler, self.train_loader, self.val_loader
        )

        self.mlflow_enabled = config.mlflow
        if self.mlflow_enabled:
            self._start_mlflow()

    def _get_params(self) -> Optional[int]:
        """Count trainable parameters."""
        if self.accelerator.is_main_process:
            try:
                unwrapped_model = self.accelerator.unwrap_model(self.model)
            except (KeyError, AttributeError):
                unwrapped_model = self.model
            return sum(p.numel() for p in unwrapped_model.parameters() if p.requires_grad)
        return None

    def _start_mlflow(self):
        """Start MLflow run (only on main process)."""
        if not self.accelerator.is_main_process:
            return

        try:
            if mlflow.active_run():
                mlflow.end_run()

            mlflow.set_tracking_uri(self.config.mlflow_tracking_uri)
            self.accelerator.print(f"MLflow tracking URI: {self.config.mlflow_tracking_uri}")

            mlflow.set_experiment(self.config.experiment_name)
            self.accelerator.print(f"MLflow experiment: {self.config.experiment_name}")

            # Enable autologging
            mlflow.pytorch.autolog(
                log_models=True,
                log_every_n_epoch=1,
                log_every_n_step=None,
                disable=False,
                exclusive=False,
                disable_for_unsupported_versions=False,
                silent=False,
            )

            run_name = self.config.run_name or f"{self.model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self.accelerator.print("Starting MLflow run...")
            mlflow.start_run(run_name=run_name)

            active_run = mlflow.active_run()
            if not active_run:
                self.accelerator.print("Failed to start MLflow run")
                self.mlflow_enabled = False
                return

            run_id = active_run.info.run_id
            experiment_id = active_run.info.experiment_id
            self.accelerator.print(f"Run ID: {run_id}")
            self.accelerator.print(f"Experiment ID: {experiment_id}")
            
            if self.config.mlflow_tracking_uri.startswith("./") or self.config.mlflow_tracking_uri.startswith("file:"):
                self.accelerator.print("")
                self.accelerator.print("To view training metrics in real-time:")
                self.accelerator.print("  1. Open a NEW terminal")
                self.accelerator.print("  2. Run: mlflow ui")
                self.accelerator.print("  3. Open: http://localhost:5000")
                self.accelerator.print("")

            mlflow.set_tag("model_type", self.model_type)
            mlflow.set_tag("dataset", self.dataset_name)
            mlflow.set_tag("mlflow.runName", run_name)

            if self.config.training_info:
                mlflow.set_tag("training_info", self.config.training_info)

            mlflow.set_tag("num_processes", str(self.accelerator.num_processes))
            mlflow.set_tag("distributed_type", str(self.accelerator.distributed_type))
            mlflow.set_tag("mixed_precision", str(self.accelerator.mixed_precision))

            mlflow.log_params(
                {
                    "learning_rate": self.lr,
                    "batch_size": self.batch_size,
                    "epochs": self.epochs,
                    "weight_decay": self.weight_decay,
                    "warmup_steps": self.warmup_steps,
                    "model_name": self.model_type,
                    "dataset": self.dataset_name,
                }
            )

            if self.param_count:
                mlflow.log_metrics({"trainable_parameters": self.param_count})

        except Exception as e:
            self.accelerator.print(f"MLflow setup failed: {e}")
            self.mlflow_enabled = False

    def _load_optimizer(self) -> torch.optim.Optimizer:
        """Load optimizer based on config (all types pre-validated)."""
        optimizer_type = self.config.optimizer.lower()
        
        if optimizer_type == "adamw":
            return torch.optim.AdamW(
                self.model.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay,
                betas=(self.config.beta1, self.config.beta2),
                eps=self.config.eps,
            )
        elif optimizer_type == "sgd":
            return torch.optim.SGD(
                self.model.parameters(),
                lr=self.lr,
                momentum=self.config.momentum,
                weight_decay=self.weight_decay,
                nesterov=self.config.nesterov,
            )
        elif optimizer_type == "adam":
            return torch.optim.Adam(
                self.model.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay,
                betas=(self.config.beta1, self.config.beta2),
                eps=self.config.eps,
            )
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_type}")

    def _load_scheduler(self) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        """Load learning rate scheduler based on config (all types pre-validated)."""
        scheduler_type = self.config.scheduler.lower()

        if scheduler_type == "none":
            return None

        total_steps = len(self.train_loader) * self.epochs

        if scheduler_type == "linear":
            from transformers import get_linear_schedule_with_warmup

            return get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=self.warmup_steps,
                num_training_steps=total_steps,
            )
        elif scheduler_type == "cosine":
            from transformers import get_cosine_schedule_with_warmup

            return get_cosine_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=self.warmup_steps,
                num_training_steps=total_steps,
            )
        elif scheduler_type == "cosine_annealing":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.epochs,
                eta_min=self.config.min_lr,
            )
        elif scheduler_type == "step":
            return torch.optim.lr_scheduler.MultiStepLR(
                self.optimizer,
                milestones=self.config.scheduler_steps,
                gamma=self.config.scheduler_gamma,
            )
        else:
            raise ValueError(f"Unsupported scheduler: {scheduler_type}")

    def run(self, checkpoint_path: Optional[str] = None, mode: str = "train"):
        """
        Main entry point for training or evaluation.

        Args:
            checkpoint_path: Path to checkpoint for resuming/testing
            mode: "train" or "test"
        """
        try:
            if mode == "test":
                if not checkpoint_path:
                    raise ValueError("checkpoint_path required for testing")
                self._test_model(checkpoint_path)
            else:
                self._train()

            self._save_and_log()

        finally:
            self.accelerator.wait_for_everyone()
            self.accelerator.end_training()
            if self.accelerator.is_main_process and self.mlflow_enabled:
                mlflow.end_run()

    def _train(self):
        """Main training loop."""
        self.training_start_time = time.time()
        self.accelerator.print("\n" + "=" * 60)
        self.accelerator.print("STARTING TRAINING")
        self.accelerator.print("=" * 60)

        for epoch in range(self.epochs):
            epoch_start = time.time()
            train_metrics = self._train_step(epoch)
            val_metrics = self._evaluate(epoch)

            if self.scheduler:
                self.scheduler.step()

            if self.accelerator.is_main_process:
                current_metric = val_metrics.get(self.config.metric_for_best_model, 0.0)

                if current_metric > self.best_metric:
                    self.best_metric = current_metric
                    self.best_epoch = epoch

                    try:
                        unwrapped_model = self.accelerator.unwrap_model(self.model)
                    except (KeyError, AttributeError):
                        unwrapped_model = self.model

                    self.best_model = copy.deepcopy(unwrapped_model.state_dict())

                epoch_time = time.time() - epoch_start
                self.epoch_times.append(epoch_time)
                total_elapsed = time.time() - self.training_start_time
                avg_epoch_time = sum(self.epoch_times) / len(self.epoch_times)
                estimated_remaining = avg_epoch_time * (self.epochs - epoch - 1)

                if self.mlflow_enabled:
                    mlflow_metrics = {
                        **{f"train_{k}": v for k, v in train_metrics.items()},
                        **{f"val_{k}": v for k, v in val_metrics.items()},
                        "best_metric": self.best_metric,
                        "learning_rate": self.optimizer.param_groups[0]["lr"],
                        "epoch_time": epoch_time,
                    }
                    mlflow.log_metrics(mlflow_metrics, step=epoch)

                self.accelerator.print(f"\nEpoch {epoch + 1}/{self.epochs} completed in {epoch_time:.2f}s")
                self.accelerator.print(f"Train metrics: {self._format_metrics(train_metrics)}")
                self.accelerator.print(f"Val metrics: {self._format_metrics(val_metrics)}")
                self.accelerator.print(f"Best metric: {self.best_metric:.4f} (epoch {self.best_epoch + 1})")
                self.accelerator.print(
                    f"Avg epoch: {avg_epoch_time:.2f}s | "
                    f"Elapsed: {total_elapsed/60:.1f}min | "
                    f"ETA: {estimated_remaining/60:.1f}min"
                )
                self.accelerator.print("-" * 60)

        if self.accelerator.is_main_process:
            total_training_time = time.time() - self.training_start_time
            final_avg_epoch_time = sum(self.epoch_times) / len(self.epoch_times)

            self.accelerator.print("\n" + "=" * 60)
            self.accelerator.print("TRAINING COMPLETE")
            self.accelerator.print("=" * 60)
            self.accelerator.print(f"Total time: {total_training_time/60:.1f} min ({total_training_time/3600:.2f} hrs)")
            self.accelerator.print(f"Avg epoch time: {final_avg_epoch_time:.2f}s")
            self.accelerator.print(f"Best metric: {self.best_metric:.4f} (epoch {self.best_epoch + 1})")
            
            # Remind about MLflow UI
            if self.mlflow_enabled and (self.config.mlflow_tracking_uri.startswith("./") or self.config.mlflow_tracking_uri.startswith("file:")):
                self.accelerator.print("")
                self.accelerator.print("To view training results:")
                self.accelerator.print("  1. Open a terminal and run: mlflow ui")
                self.accelerator.print("  2. Open browser to: http://localhost:5000")
                self.accelerator.print("")
            
            self.accelerator.print("=" * 60 + "\n")

            if self.mlflow_enabled:
                mlflow.log_metrics(
                    {
                        "best_metric_final": self.best_metric,
                        "best_epoch": self.best_epoch,
                        "avg_epoch_time_sec": final_avg_epoch_time,
                        "total_training_time_min": total_training_time / 60,
                    }
                )

    def _save_and_log(self):
        """Save best model and log to MLflow."""
        if not self.accelerator.is_main_process:
            return

        if self.save_best and self.best_model:
            save_path = self.checkpoint_dir / f"{self.model_type}_best.pth"
            self.accelerator.save(self.best_model, save_path)
            self.accelerator.print(f"Best model saved to {save_path}")

            if self.mlflow_enabled:
                mlflow.log_artifact(str(save_path), "models")

    def _format_metrics(self, metrics: Dict[str, float]) -> str:
        """Format metrics for display."""
        return " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])

    # ========== ABSTRACT METHODS (must be implemented by subclasses) ==========

    @abstractmethod
    def _load_model(self):
        """
        Load and return the model.

        Returns:
            model: PyTorch model
        """
        pass

    @abstractmethod
    def _load_dataset(self) -> Tuple[DataLoader, DataLoader]:
        """
        Load and return train/val dataloaders.

        Returns:
            train_loader: Training dataloader
            val_loader: Validation dataloader
        """
        pass

    @abstractmethod
    def _train_step(self, epoch: int) -> Dict[str, float]:
        """
        Execute one training epoch.

        Args:
            epoch: Current epoch number

        Returns:
            metrics: Dictionary of training metrics (e.g., {"loss": 0.5, "accuracy": 0.9})
        """
        pass

    @abstractmethod
    def _evaluate(self, epoch: int) -> Dict[str, float]:
        """
        Evaluate the model on validation set.

        Args:
            epoch: Current epoch number

        Returns:
            metrics: Dictionary of validation metrics
        """
        pass

    def _test_model(self, checkpoint_path: str):
        """
        Test model on test set (optional, can override in subclass).

        Args:
            checkpoint_path: Path to checkpoint file
        """
        raise NotImplementedError("Testing not implemented for this solver")
