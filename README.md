# Bridge Inspection Management Application

A Flask-based web application for managing bridge inspections with an AI-powered natural language query interface using a fine-tuned T5 model.

## Live Application

- [Bridge Inspection Management Application](https://sdcc-bridge-app-fbbbeuamd2cycsdq.westus2-01.azurewebsites.net/)

$env:Path += ";C:\Users\Bo\.fly\bin"

## Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Training the T5 Model](#training-the-t5-model)
- [Testing the Model](#testing-the-model)
- [MLflow Experiment Tracking](#mlflow-experiment-tracking)
- [Running the Application](#running-the-application)
- [Production Deployment](#production-deployment)
- [Troubleshooting](#troubleshooting)

## Features

- Bridge inspection data management
- Natural language query interface powered by fine-tuned T5 model
- RESTful API for bridge data access
- Photo upload and management
- Statistics and aggregation views
- Experiment tracking with MLflow
- Multi-GPU training support with Accelerate
- Type-safe configuration with Pydantic

## Project Structure

```
bridge-app/
├── app.py                      # Main Flask application
├── src/
│   ├── data/
│   │   ├── bridge_data.csv     # Bridge data
│   │   └── training_data.json  # T5 training data
│   └── parser/
│       ├── config_schema.py    # Pydantic config schemas
│       ├── solver.py           # Base training solver
│       ├── t5_solver.py        # T5-specific solver
│       ├── train.py            # Training script
│       ├── inference.py        # Model inference
│       ├── generate_training_set.py  # Generate training data
│       └── extract_features.py       # Extract bridge features
├── config/
│   ├── t5_training.yaml        # Basic training config
│   └── t5_training_advanced.yaml  # Advanced config
├── test/
│   ├── test_local_app.py       # Local testing
│   └── test_model_interactive.py  # Interactive model testing
├── templates/                  # HTML templates
├── static/                     # CSS, JS, images
├── results/                    # Training outputs
│   └── checkpoints/            # Model checkpoints
├── mlruns/                     # MLflow tracking data
└── requirements.txt            # Python dependencies
```

## Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (optional, for faster training)
- 8GB+ RAM (16GB+ recommended for training)

### Setup

```bash
# Clone the repository
git clone <repository-url>
cd bridge-app

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install package in editable mode
pip install -e .
```

## Training the T5 Model

The application uses a fine-tuned T5 model to translate natural language queries into API endpoints.

### Quick Start

```bash
# Basic training with default config
python src/parser/train.py --config config/t5_training.yaml --solver t5
```

### Generate Training Data

Before training, generate or update the training data:

```bash
# Extract features from bridge data
python src/parser/extract_features.py

# Generate training examples
python src/parser/generate_training_set.py
```

### Training Options

#### Override Hyperparameters via CLI

```bash
python src/parser/train.py \
    --config config/t5_training.yaml \
    --solver t5 \
    --epochs 20 \
    --lr 3e-5 \
    --batch_size 16
```

#### Multi-GPU Training with FP16

```bash
accelerate launch src/parser/train.py \
    --config config/t5_training_advanced.yaml \
    --solver t5
```

### Configuration

Training parameters are defined in YAML files. Example `config/t5_training.yaml`:

```yaml
# Model
model_name: "t5-small"

# Data
data_path: "src/data/training_data.json"
val_split: 0.2

# Training
epochs: 10
batch_size: 8
learning_rate: 0.00005
weight_decay: 0.01
optimizer: "adamw"
scheduler: "linear"
warmup_steps: 200

# Evaluation
eval_steps: 50
save_steps: 100
metric_for_best_model: "accuracy"

# MLflow
mlflow: true
experiment_name: "bridge_query_parser"
```

### Command-Line Arguments

| Argument | Type | Description |
|----------|------|-------------|
| `--config` | str | Required. Path to YAML config |
| `--solver` | str | Required. Solver type (t5) |
| `--test` | flag | Run in test mode |
| `--checkpoint` | str | Checkpoint path (required for test) |
| `--batch_size` | int | Override batch size |
| `--lr` | float | Override learning rate |
| `--epochs` | int | Override number of epochs |
| `--warmup` | int | Override warmup steps |
| `--weight_decay` | float | Override weight decay |
| `--optimizer` | str | Override optimizer (adamw, adam, sgd) |
| `--scheduler` | str | Override LR scheduler |
| `--mixed_precision` | str | Override mixed precision (no, fp16, bf16) |
| `--model_name` | str | Override model name |
| `--data_path` | str | Override data path |
| `--experiment_name` | str | Override MLflow experiment name |
| `--no_mlflow` | flag | Disable MLflow tracking |

### Performance Tips

#### Multi-GPU Training
```bash
accelerate launch src/parser/train.py --config config/t5_training.yaml --solver t5
```

#### Mixed Precision (FP16)
- 2x speedup
- 2x memory reduction
```yaml
mixed_precision: "fp16"
```

#### Gradient Accumulation
```yaml
batch_size: 8
gradient_accumulation_steps: 4  # Effective batch = 32
```

#### DataLoader Workers
```yaml
num_workers: 4  # Parallel data loading
```

## Testing the Model

### Built-in Test Mode

Test with queries from config:

```bash
python src/parser/train.py \
    --config config/t5_training.yaml \
    --solver t5 \
    --test \
    --checkpoint results/checkpoints/t5-small_best.pth
```

### Interactive Testing

```bash
python test/test_model_interactive.py
```

Example session:
```
Query: How many bridges are there?
→ /api/bridges/count

Query: Show me bridges in Orange county
→ /api/bridges/search?county=Orange

Query: Find bridges with more than 3 spans
→ /api/bridges/search?min_spans=4

Query: quit
```

### Quick Test

```bash
python src/parser/inference.py results/checkpoints/t5-small_best.pth
```

## MLflow Experiment Tracking

MLflow tracks all training experiments, logging hyperparameters, metrics, and model checkpoints.

### What is MLflow?

MLflow is an experiment tracking tool that logs:
- Training hyperparameters
- Metrics per epoch (loss, accuracy, etc.)
- Model checkpoints
- Training duration and system info

### How It Works

1. Training writes data to `./mlruns/` directory
2. You start a UI server to view the data
3. Browser shows all experiments and metrics

### Usage

#### Step 1: Start Training

```bash
python src/parser/train.py --config config/t5_training.yaml --solver t5
```

This creates/updates files in `./mlruns/` but does NOT start a web server.

#### Step 2: Start MLflow UI (In Another Terminal)

```bash
# Open a NEW terminal window
cd bridge-app
mlflow ui
```

You should see:
```
[INFO] Starting gunicorn...
[INFO] Listening at: http://127.0.0.1:5000
```

#### Step 3: Open Browser

Navigate to: http://localhost:5000

### What You'll See

#### Experiments View
- List of all experiments (e.g., "bridge_query_parser")
- Click on experiment to see all runs

#### Runs View
- Each training session is a "run"
- Columns show: start time, duration, metrics, parameters

#### Run Details
Click on a run to see:
- Parameters: learning_rate, batch_size, epochs, etc.
- Metrics: Charts showing train_loss, val_loss, accuracy over epochs
- Artifacts: Saved model checkpoints
- Tags: model_type, dataset, system info

### Comparing Runs

1. Start multiple training runs with different configs:
```bash
python src/parser/train.py --config config/t5_training.yaml --solver t5 --lr 5e-5
python src/parser/train.py --config config/t5_training.yaml --solver t5 --lr 3e-5
python src/parser/train.py --config config/t5_training.yaml --solver t5 --batch_size 16
```

2. In MLflow UI:
   - Select multiple runs (checkboxes)
   - Click "Compare"
   - See side-by-side metrics, parameters, and charts

### MLflow Tips

1. Keep MLflow UI running in a separate terminal during training
2. Refresh browser to see latest metrics (auto-updates every few seconds)
3. Use tags to organize experiments (model_type, dataset, etc.)
4. Download artifacts (model checkpoints) from the UI
5. Export data to CSV for analysis: UI > Run > Export > CSV

### Stopping MLflow UI

In the terminal running `mlflow ui`:
- Press `Ctrl+C` to stop the server
- The data in `./mlruns/` persists - you can restart the UI anytime

### Data Location

All MLflow data is stored in:
```
bridge-app/
└── mlruns/
    ├── 0/                    # Default experiment
    ├── 641163323674926417/   # Your experiment ID
    │   ├── meta.yaml
    │   └── 4a9db524baa.../   # Run ID
    │       ├── metrics/
    │       ├── params/
    │       ├── tags/
    │       └── artifacts/
    └── .trash/
```

## Running the Application

### Local Development

```bash
# Run the Flask app
python app.py

# Or use the test app
python test/test_local_app.py
```

Access the application at: http://localhost:5000

### Update the Model

After training, update the Flask app to use the new model:

```python
# In app.py or test_local_app.py
from src.parser.inference import BridgeQueryParser

# Load the trained model
model_path = "results/checkpoints/t5-small_best.pth"
parser = BridgeQueryParser(model_path)

# Use in endpoint
@app.route('/api/agent/query', methods=['POST'])
def agent_query():
    data = request.get_json()
    query = data.get('query', '').strip()

    # Parse with trained model
    endpoint = parser.parse_query(query)
    # ... handle response
```

## Production Deployment

### Azure App Service Deployment

Listed below are the commands used to deploy the application to Azure App Service.

```bash
# Create deployment package
rm -f deploy.zip
zip -r deploy.zip \
  app.py \
  requirements.txt \
  src/ \
  static/ \
  templates/ \
  results/ \
  -x "*.pyc" "*__pycache__*" "*venv*" "*.env" "*/.git*" "*.zip"

# Deploy to Azure
az webapp deploy \
  --resource-group apps-project \
  --name sdcc-bridge-app \
  --src-path deploy.zip \
  --type zip

# Configure app settings
az webapp config appsettings set \
  --resource-group apps-project \
  --name sdcc-bridge-app \
  --settings AZURE_OPENAI_API_KEY=<your-key> \
             AZURE_OPENAI_ENDPOINT=<your-endpoint> \
             FLASK_BASE_URL=<your-url>

# Restart the app
az webapp restart \
  --name sdcc-bridge-app \
  --resource-group apps-project

# View logs
az webapp log tail --name sdcc-bridge-app --resource-group apps-project
```

## Troubleshooting

### Import Errors

```bash
# Reinstall package in editable mode
pip install -e .
```

### CUDA Out of Memory

```bash
# Reduce batch size
python src/parser/train.py --config config/t5_training.yaml --solver t5 --batch_size 4

# Or use gradient accumulation
# Edit YAML: gradient_accumulation_steps: 4
```

### MLflow Not Starting

```bash
# Check MLflow directory
ls -la mlruns/

# Start MLflow on different port
mlflow ui --port 5001
```

### MLflow "No experiments found"

Make sure you're in the correct directory (`bridge-app`) when running `mlflow ui`. It looks for `./mlruns/` in the current directory.

### Model Not Learning Numbers Correctly

If the model doesn't generalize to different numbers (e.g., "more than 50 spans" returns wrong value):

1. Regenerate training data with diverse numbers:
```bash
python src/parser/generate_training_set.py
```

2. Retrain the model:
```bash
python src/parser/train.py --config config/t5_training.yaml --solver t5
```

The training data now includes examples with numbers ranging from 1 to 100 to help the model generalize.

## Adding a New Model Type

To support a new model type (e.g., GPT, BERT):

### 1. Create a solver class

```python
# src/parser/my_solver.py
from solver import BaseSolver
from typing import Dict, Tuple
from torch.utils.data import DataLoader

class MySolver(BaseSolver):
    def _load_model(self):
        # Load your model
        pass

    def _load_dataset(self) -> Tuple[DataLoader, DataLoader]:
        # Create dataloaders
        pass

    def _train_step(self, epoch: int) -> Dict[str, float]:
        # Training loop
        pass

    def _evaluate(self, epoch: int) -> Dict[str, float]:
        # Validation loop
        pass
```

### 2. Register in train.py

```python
def load_solver_class(solver_type: str):
    if solver_type == "t5":
        from src.parser.t5_solver import T5Solver
        return T5Solver
    elif solver_type == "my_model":
        from src.parser.my_solver import MySolver
        return MySolver
```

### 3. Create config

```yaml
# config/my_model_training.yaml
model_name: "my-model"
# ... other configs
```

### 4. Train

```bash
python src/parser/train.py --config config/my_model_training.yaml --solver my_model
```

## Further Reading

- [Accelerate Documentation](https://huggingface.co/docs/accelerate)
- [MLflow Tracking](https://www.mlflow.org/docs/latest/tracking.html)
- [T5 Paper](https://arxiv.org/abs/1910.10683)
- [Pydantic Documentation](https://docs.pydantic.dev/)
