"""
Helper functions for model training with MLflow tracking and Accelerate

Example usage:
    # Initialize accelerator
    accelerator = setup_accelerator(mixed_precision="fp16")
    
    # Get training parameters
    params = get_training_params(
        num_epochs=10,
        batch_size=8,
        learning_rate=5e-5,
        model_name="t5-small",
        train_size=1200,
        val_size=300,
        accelerator=accelerator
    )
    
    # Start MLflow tracking
    run_id, exp_id = start_mlflow_run(
        experiment_name="bridge_query_parser",
        run_name="t5_training_v1",
        training_params=params,
        accelerator=accelerator
    )
    
    # During training, log metrics
    log_metrics(
        {"train_loss": 0.5, "val_loss": 0.45},
        step=100,
        accelerator=accelerator
    )
    
    # End run
    end_mlflow_run(accelerator=accelerator)
"""
import logging
import mlflow
import mlflow.pytorch
from typing import Optional, Dict, Any
from accelerate import Accelerator
from config.logging_config import configure_logger

logger = logging.getLogger(__name__)
configure_logger(log_level="INFO", logger=logger)


class AverageMeter(object):
    """Computes and stores the average and current value"""
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


def setup_accelerator(
    mixed_precision: str = "fp16",
    gradient_accumulation_steps: int = 1,
    log_with: str = "mlflow",
    project_dir: str = "./results",
) -> Accelerator:
    """
    Initialize Hugging Face Accelerator for distributed training.
    
    Args:
        mixed_precision: Mixed precision mode ('no', 'fp16', 'bf16')
        gradient_accumulation_steps: Number of gradient accumulation steps
        log_with: Logging framework to use ('mlflow', 'tensorboard', 'wandb')
        project_dir: Directory for saving results
        
    Returns:
        Initialized Accelerator instance
    """
    try:
        accelerator = Accelerator(
            mixed_precision=mixed_precision,
            gradient_accumulation_steps=gradient_accumulation_steps,
            log_with=log_with,
            project_dir=project_dir,
        )
        
        accelerator.print(f"Accelerator initialized:")
        accelerator.print(f"  Device: {accelerator.device}")
        accelerator.print(f"  Distributed type: {accelerator.distributed_type}")
        accelerator.print(f"  Mixed precision: {mixed_precision}")
        accelerator.print(f"  Num processes: {accelerator.num_processes}")
        accelerator.print(f"  Process index: {accelerator.process_index}")
        
        return accelerator
        
    except Exception as e:
        logger.error(f"Failed to initialize Accelerator: {e}")
        logger.warning("Falling back to CPU training without Accelerator")
        return None


def start_mlflow_run(
    experiment_name: str = "bridge_query_parser",
    run_name: Optional[str] = None,
    training_params: Optional[Dict[str, Any]] = None,
    accelerator: Optional[Accelerator] = None,
) -> tuple[str, str]:
    """
    Start an MLflow run for model training.
    
    Args:
        experiment_name: Name of the MLflow experiment
        run_name: Optional name for this specific run
        training_params: Dictionary of training parameters to log
        accelerator: Optional Accelerator instance (only main process logs)
        
    Returns:
        Tuple of (run_id, experiment_id)
    """
    if accelerator and not accelerator.is_main_process:
        return None, None
    
    def _print(msg):
        if accelerator:
            accelerator.print(msg)
        else:
            logger.info(msg)
    
    try:
        if mlflow.active_run():
            _print("Ending existing MLflow run")
            mlflow.end_run()
        
        mlflow.set_tracking_uri("./results")
        _print(f"MLflow tracking URI: ./results")
        
        mlflow.set_experiment(experiment_name)
        _print(f"MLflow experiment: {experiment_name}")
        
        mlflow.pytorch.autolog(
            log_models=True,
            log_every_n_epoch=1,
            log_every_n_step=None,
            disable=False,
            exclusive=False,
            disable_for_unsupported_versions=False,
            silent=False,
        )
        
        _print("Starting MLflow run...")
        mlflow.start_run(run_name=run_name)
        
        active_run = mlflow.active_run()
        if not active_run:
            _print("Failed to start MLflow run")
            return None, None
        
        run_id = active_run.info.run_id
        experiment_id = active_run.info.experiment_id
        _print(f"MLflow Run ID: {run_id}")
        _print(f"MLflow Experiment ID: {experiment_id}")
        
        if run_name:
            mlflow.set_tag("mlflow.runName", run_name)
        mlflow.set_tag("model_type", "T5")
        mlflow.set_tag("task", "seq2seq_query_translation")
        
        if accelerator:
            mlflow.set_tag("num_processes", str(accelerator.num_processes))
            mlflow.set_tag("distributed_type", str(accelerator.distributed_type))
            mlflow.set_tag("mixed_precision", str(accelerator.mixed_precision))
        
        if training_params:
            mlflow.log_params(training_params)
            _print(f"Logged {len(training_params)} parameters to MLflow")
        
        return run_id, experiment_id
        
    except Exception as e:
        if accelerator:
            accelerator.print(f"MLflow setup failed: {e}")
        else:
            logger.error(f"MLflow setup failed: {e}")
        return None, None


def get_training_params(
    num_epochs: int,
    batch_size: int,
    learning_rate: float,
    model_name: str = "t5-small",
    warmup_steps: int = 200,
    weight_decay: float = 0.01,
    train_size: int = 0,
    val_size: int = 0,
    accelerator: Optional[Accelerator] = None,
) -> Dict[str, Any]:
    """
    Get training parameters for MLflow logging (matching train_model.py).
    
    Args:
        num_epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Learning rate
        model_name: Pretrained model name
        warmup_steps: Number of warmup steps
        weight_decay: Weight decay for regularization
        train_size: Training dataset size
        val_size: Validation dataset size
        accelerator: Optional Accelerator instance for distributed training info
        
    Returns:
        Dictionary of parameters for MLflow
    """
    params = {
        "model_name": model_name,
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "warmup_steps": warmup_steps,
        "weight_decay": weight_decay,
        "train_size": train_size,
        "val_size": val_size,
        "fp16": True,  # Mixed precision training
        "save_strategy": "steps",
        "save_steps": 100,
        "eval_strategy": "steps",
        "eval_steps": 50,
        "logging_steps": 25,
    }
    
    if accelerator:
        params.update({
            "num_processes": accelerator.num_processes,
            "distributed_type": str(accelerator.distributed_type),
            "mixed_precision": str(accelerator.mixed_precision),
            "device": str(accelerator.device),
        })
    
    return params


def log_metrics(
    metrics: Dict[str, float], 
    step: Optional[int] = None,
    accelerator: Optional[Accelerator] = None
):
    """
    Log metrics to MLflow (only from main process in distributed training).
    
    Args:
        metrics: Dictionary of metric names and values
        step: Optional step number
        accelerator: Optional Accelerator instance
    """
    if accelerator and not accelerator.is_main_process:
        return
    
    try:
        if mlflow.active_run():
            mlflow.log_metrics(metrics, step=step)
            logger.debug(f"Logged metrics: {metrics}")
        else:
            msg = "No active MLflow run, skipping metric logging"
            if accelerator:
                accelerator.print(msg)
            else:
                logger.warning(msg)
    except Exception as e:
        msg = f"Failed to log metrics: {e}"
        if accelerator:
            accelerator.print(msg)
        else:
            logger.error(msg)


def end_mlflow_run(accelerator: Optional[Accelerator] = None):
    """
    End the current MLflow run (only main process in distributed training).
    
    Args:
        accelerator: Optional Accelerator instance
    """
    if accelerator and not accelerator.is_main_process:
        return
    
    try:
        if mlflow.active_run():
            mlflow.end_run()
            msg = "MLflow run ended"
            if accelerator:
                accelerator.print(msg)
            else:
                logger.info(msg)
        else:
            logger.debug("No active MLflow run to end")
    except Exception as e:
        msg = f"Failed to end MLflow run: {e}"
        if accelerator:
            accelerator.print(msg)
        else:
            logger.error(msg)
