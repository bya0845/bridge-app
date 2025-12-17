"""
Configuration schemas with type validation using Pydantic.

All YAML configs are validated once at load time.
"""

from typing import List, Optional, Literal
from pathlib import Path
from pydantic import BaseModel, Field, validator
import yaml


class BaseTrainingConfig(BaseModel):
    """Base configuration schema for all training solvers."""
    
    # Model
    model_name: str = Field(default="t5-small", description="Model name or path")
    
    # Data
    dataset: str = Field(default="unknown", description="Dataset name")
    data_path: str = Field(default="src/data/training_data.json", description="Path to training data")
    val_split: float = Field(default=0.2, ge=0.0, le=1.0, description="Validation split ratio")
    seed: int = Field(default=42, description="Random seed")
    
    # Training hyperparameters
    epochs: int = Field(default=10, ge=1, description="Number of training epochs")
    batch_size: int = Field(default=8, ge=1, description="Batch size")
    learning_rate: float = Field(default=5e-5, gt=0.0, description="Learning rate")
    weight_decay: float = Field(default=0.01, ge=0.0, description="Weight decay")
    warmup_steps: int = Field(default=0, ge=0, description="Warmup steps")
    max_grad_norm: float = Field(default=1.0, ge=0.0, description="Max gradient norm for clipping")
    
    # Optimizer
    optimizer: Literal["adamw", "adam", "sgd"] = Field(default="adamw", description="Optimizer type")
    beta1: float = Field(default=0.9, ge=0.0, le=1.0, description="Adam beta1")
    beta2: float = Field(default=0.999, ge=0.0, le=1.0, description="Adam beta2")
    eps: float = Field(default=1e-8, gt=0.0, description="Optimizer epsilon")
    momentum: float = Field(default=0.9, ge=0.0, le=1.0, description="SGD momentum")
    nesterov: bool = Field(default=True, description="Use Nesterov momentum")
    
    # Scheduler
    scheduler: Literal["linear", "cosine", "cosine_annealing", "step", "none"] = Field(
        default="linear", description="Learning rate scheduler"
    )
    min_lr: float = Field(default=1e-6, ge=0.0, description="Minimum learning rate")
    scheduler_gamma: float = Field(default=0.1, gt=0.0, description="Step scheduler gamma")
    scheduler_steps: List[int] = Field(default_factory=list, description="Step scheduler milestones")
    
    # Evaluation & saving
    eval_steps: int = Field(default=100, ge=1, description="Evaluate every N steps")
    save_steps: int = Field(default=100, ge=1, description="Save checkpoint every N steps")
    save_total_limit: int = Field(default=3, ge=1, description="Max checkpoints to keep")
    save_best: bool = Field(default=True, description="Save best model")
    load_best_model_at_end: bool = Field(default=True, description="Load best model at end")
    metric_for_best_model: str = Field(default="accuracy", description="Metric to track for best model")
    
    # Output
    output_dir: str = Field(default="./results", description="Output directory")
    
    # Distributed training
    mixed_precision: Literal["no", "fp16", "bf16"] = Field(default="no", description="Mixed precision mode")
    gradient_accumulation_steps: int = Field(default=1, ge=1, description="Gradient accumulation steps")
    num_workers: int = Field(default=0, ge=0, description="DataLoader workers")
    
    # MLflow
    mlflow: bool = Field(default=True, description="Enable MLflow tracking")
    mlflow_tracking_uri: str = Field(default="./mlruns", description="MLflow tracking URI")
    experiment_name: str = Field(default="model_training", description="MLflow experiment name")
    run_name: Optional[str] = Field(default=None, description="MLflow run name")
    training_info: Optional[str] = Field(default=None, description="Training info tag")
    log_with: Optional[str] = Field(default="mlflow", description="Accelerate logging framework")
    
    # Logging
    log_interval: int = Field(default=10, ge=1, description="Log every N batches")
    
    class Config:
        extra = "allow"  # Allow extra fields for forward compatibility
    
    @validator("data_path")
    def validate_data_path(cls, v):
        """Validate data path exists (skip validation if file will be created later)."""
        # Note: Validation disabled to allow configs before data generation
        # path = Path(v)
        # if not path.exists():
        #     raise ValueError(f"Data path not found: {v}")
        return v
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> "BaseTrainingConfig":
        """Load config from YAML file with type validation."""
        yaml_path = Path(yaml_path)
        if not yaml_path.exists():
            raise FileNotFoundError(f"Config file not found: {yaml_path}")
        
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
        
        return cls(**data)


class T5TrainingConfig(BaseTrainingConfig):
    """Configuration schema for T5 seq2seq training."""
    
    # T5-specific parameters
    max_source_length: int = Field(default=128, ge=1, description="Max tokens for input")
    max_target_length: int = Field(default=64, ge=1, description="Max tokens for output")
    num_beams: int = Field(default=4, ge=1, description="Beam search width")
    
    # Test queries
    test_queries: List[str] = Field(default_factory=list, description="Test queries for evaluation")
    
    @validator("max_target_length")
    def validate_target_length(cls, v, values):
        """Ensure target length is reasonable."""
        if v > values.get("max_source_length", 128) * 2:
            raise ValueError("max_target_length should not be more than 2x max_source_length")
        return v


# Config type registry
CONFIG_REGISTRY = {
    "t5": T5TrainingConfig,
    "base": BaseTrainingConfig,
}


def load_config(yaml_path: str, config_type: str = "base") -> BaseTrainingConfig:
    """
    Load and validate configuration from YAML.
    
    Args:
        yaml_path: Path to YAML config file
        config_type: Type of config ("t5", "base")
    
    Returns:
        Validated config object with typed attributes
    
    Raises:
        FileNotFoundError: If config file doesn't exist
        ValidationError: If config values are invalid
    """
    if config_type not in CONFIG_REGISTRY:
        raise ValueError(f"Unknown config type: {config_type}. Available: {list(CONFIG_REGISTRY.keys())}")
    
    ConfigClass = CONFIG_REGISTRY[config_type]
    return ConfigClass.from_yaml(yaml_path)

