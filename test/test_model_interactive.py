"""
Interactive testing script for the trained T5 bridge query parser.

Usage:
    python test_model_interactive.py
    
    Or specify a custom model path:
    python test_model_interactive.py --model results/checkpoints/t5-small_best.pth
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.parser.inference import BridgeQueryParser


def main():
    parser = argparse.ArgumentParser(description="Interactive T5 model testing")
    parser.add_argument(
        "--model",
        type=str,
        default="results/checkpoints/t5-small_best.pth",
        help="Path to trained model checkpoint"
    )
    args = parser.parse_args()
    
    # Check if model exists
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        print("\nAvailable checkpoints:")
        checkpoint_dir = Path("results/checkpoints")
        if checkpoint_dir.exists():
            for ckpt in checkpoint_dir.glob("*.pth"):
                print(f"  - {ckpt}")
        else:
            print("  No checkpoints found. Train a model first:")
            print("  python src/parser/train.py --config config/t5_training.yaml --solver t5")
        return
    
    # Load the trained model
    print("Loading model...")
    try:
        query_parser = BridgeQueryParser(str(model_path))
        print(f"Model loaded from: {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    print("\n" + "="*60)
    print("Bridge Query Parser - Interactive Testing")
    print("="*60)
    print("\nCommands:")
    print("  - Type your natural language query")
    print("  - Type 'examples' to see sample queries")
    print("  - Type 'quit' or 'exit' to quit")
    print("="*60 + "\n")
    
    # Example queries for reference
    examples = [
        "How many bridges are there?",
        "Show me bridges in Orange county",
        "Find bridges with more than 3 spans",
        "List bridges carrying highway 84i",
        "What are the bridge statistics by county?",
        "Show me the top 5 bridges by span count",
        "Orange county bridges carrying 84i with more than 4 spans",
        "Westchester bridges crossing river with at least 3 spans",
        "Bridges crossing saw mill river",
        "Find top 10 bridges by span count",
    ]
    
    while True:
        try:
            query = input("Query: ").strip()
            
            if not query:
                continue
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("\nGoodbye!")
                break
            
            if query.lower() == 'examples':
                print("\nExample queries:")
                for i, example in enumerate(examples, 1):
                    print(f"  {i}. {example}")
                print()
                continue
            
            # Parse the query
            endpoint = query_parser.parse_query(query)
            print(f"→ {endpoint}\n")
            
        except KeyboardInterrupt:
            print("\n\nInterrupted. Goodbye!")
            break
        except Exception as e:
            print(f"Error: {e}\n")


if __name__ == "__main__":
    main()

