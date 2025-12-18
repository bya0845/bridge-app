# Bridge Inspection and Database Management Application

This is a flask-based web application for managing bridge inspections and querying a bridge database with both a natural language interface and traditional SQL search. The app is hosted on fly.io: https://bridge-app.fly.dev/.

## About

The basis of this application is a spreadsheet of bridge data from my job as a structural/bridge inspection engineer. The spreadsheet was imported into an SQLite database via pandas, and the application's search and query features are built on top. The natural language search allows users to query the database with plain English questions to fetch data. A list of examples is provided on the AI search page.

The "agent" underneath this NLP feature is the [Google T5-small](https://huggingface.co/google-t5/t5-small) model, which was one of the earliest text-to-text encoder/decoder transformers designed specifically for NLP tasks. More can be found here: [Exploring Transfer Learning with T5, the Text-to-Text Transformer](https://research.google/blog/exploring-transfer-learning-with-t5-the-text-to-text-transfer-transformer/).

The base T5 model was finetuned on a generated dataset of queries based on the spreadsheet data. The model has only 60 million parameters and was trained in a few minutes over 10 epochs on my RTX 4070. The total checkpoint size is roughly 250 MB and is deployed via Github LFS. The solver code was adapted from a neural network pruning project in my [Deep Learning](https://omscs.gatech.edu/cs-7643-deep-learning) course, found here: https://github.com/bya0845/CS7643-DL-Project. All training code is in the [parser module](src/parser/). It includes the complete pipeline for preprocessing data, training, evaluating, and deploying this model. The solver also supports multi-GPU training via Hugging Face's Accelerate library, as well as experiment tracking via MLflow.

All individal bridge search results can be displayed via Leaflet/OpenStreetMap.

The main goal of the project is to demonstrate the use of a lightweight, open source, offline language model that can accomplish most NLP tasks without the need for enterprise foundation LLMs.

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

* Python 3.9+
* CUDA-capable GPU (optional)

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

Training parameters are defined in YAML files. Example `config/t5_training.yaml`

### Command-Line Arguments

| Argument | Type | Description |
| --- | --- | --- |
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

## Further Reading

* [Accelerate Documentation](https://huggingface.co/docs/accelerate)
* [MLflow Tracking](https://www.mlflow.org/docs/latest/tracking.html)
* [T5 Paper](https://arxiv.org/abs/1910.10683)
* [Pydantic Documentation](https://docs.pydantic.dev/)
