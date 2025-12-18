"""
Inference module for the trained T5 bridge query parser.

This module loads a trained T5 model and provides a simple interface
for translating natural language queries to API endpoints.
"""
import os
import logging
import torch
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)
try:
    import torch
    from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config
    AI_AVAILABLE = True
except ImportError:
    AI_AVAILABLE = False
    logger.warning("'transformers' or 'torch' not found. AI features will be disabled.")


class BridgeQueryParser:
    """Parser for translating natural language queries to API endpoints."""

    def __init__(
        self,
        model_path: str,
        model_name: str = "t5-small",
        max_input_length: int = 128,
        max_output_length: int = 64,
        num_beams: int = 4,
        device: Optional[str] = None,
    ):
        """
        Initialize the parser with a trained model.

        Args:
            model_path: Path to trained model checkpoint (.pth file)
            model_name: Base model name (for tokenizer fallback)
            max_input_length: Maximum input sequence length
            max_output_length: Maximum output sequence length
            num_beams: Number of beams for beam search
            device: Device to run on ('cuda', 'cpu', or None for auto)
        """

        if not AI_AVAILABLE:
            self.model = None
            self.tokenizer = None
            return

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        if not model_path:
            raise ValueError("model_path is required and cannot be None")

        current_dir = os.path.dirname(os.path.abspath(__file__))
        local_base_path = os.path.join(current_dir, "t5_base")

        if os.path.exists(local_base_path):
            logger.info(f"Found local base model config at: {local_base_path}")
            model_source = local_base_path
        else:
            logger.warning(
                f"Local base model not found at {local_base_path}. Defaulting to '{model_name}' (Internet Download)."
            )
            model_source = model_name

        self.model_name = model_name
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length
        self.num_beams = num_beams
        self._load_tokenizer(model_source)


        logger.info(f"Initializing model structure from {model_source}")
        config = T5Config.from_pretrained(model_source)
        self.model = T5ForConditionalGeneration(config)

        self._load_model(model_path)
        self.model.to(self.device)
        self.model.eval()

    def _load_model(self, model_path):
        """Load model weights"""
        try:
            mp = Path(model_path)
            if mp.exists():
                logger.info(f"Loading fine-tuned weights from {model_path}")
                state_dict = torch.load(model_path, map_location=self.device)

                # Handle cases where state_dict might be nested
                if "model_state_dict" in state_dict:
                    state_dict = state_dict["model_state_dict"]

                self.model.load_state_dict(state_dict)
                logger.info("Model weights loaded successfully")
            else:
                logger.warning(f"Checkpoint not found at {model_path}, using initialized base weights")
        except Exception as e:
            raise ValueError(f"Failed to load checkpoint '{model_path}': {e}") from e

    def _load_tokenizer(self, model_source):
        """Load the T5 tokenizer."""
        try:
            logger.info(f"Loading tokenizer from {model_source}")
            self.tokenizer = T5Tokenizer.from_pretrained(model_source)
            logger.info("Tokenizer loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load tokenizer: {e}")
            # Fallback to model_name if local load fails
            if model_source != self.model_name:
                logger.info(f"Retrying tokenizer load with {self.model_name}")
                self.tokenizer = T5Tokenizer.from_pretrained(self.model_name)
            else:
                raise ValueError(f"Could not load tokenizer. Error: {e}") from e

    def parse_query(self, query: str) -> str:
        """
        Parse a natural language query to an API endpoint.

        Args:
            query: Natural language query (e.g., "Show me bridges in Orange county")

        Returns:
            API endpoint string (e.g., "/api/bridges/search?county=Orange")
        """
        if not AI_AVAILABLE or self.model is None:
            logger.warning("AI query ignored: Transformers not installed or model failed to load.")
            return ""

        if not query or not query.strip():
            logger.warning("Empty query provided")
            return ""

        input_text = f"translate to api: {query.strip()}"

        try:
            inputs = self.tokenizer(
                input_text,
                return_tensors="pt",
                max_length=self.max_input_length,
                truncation=True,
                padding=False,
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=self.max_output_length,
                    num_beams=self.num_beams,
                    early_stopping=True,
                )

            result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            logger.debug(f"Query: {query} -> Endpoint: {result}")
            return result.strip()

        except Exception as e:
            logger.error(f"Error parsing query '{query}': {e}")
            return ""

    def parse_batch(self, queries: list[str]) -> list[str]:
        """
        Parse multiple queries in a batch.

        Args:
            queries: List of natural language queries

        Returns:
            List of API endpoints
        """
        if not queries:
            return []

        input_texts = [f"translate to api: {q.strip()}" for q in queries]

        try:
            inputs = self.tokenizer(
                input_texts,
                return_tensors="pt",
                max_length=self.max_input_length,
                truncation=True,
                padding=True,
            ).to(self.device)

            # Generate batch
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=self.max_output_length,
                    num_beams=self.num_beams,
                    early_stopping=True,
                )

            results = [self.tokenizer.decode(output, skip_special_tokens=True).strip() for output in outputs]

            return results

        except Exception as e:
            logger.error(f"Error parsing batch: {e}")
            return [""] * len(queries)


def test_parser():
    """Test the parser with sample queries."""
    import sys

    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    else:
        model_path = "results/checkpoints/t5-small_best.pth"

    print(f"Loading model from: {model_path}\n")

    try:
        parser = BridgeQueryParser(model_path)
    except Exception as e:
        print(f"Error: {e}")
        return

    test_queries = [
        "How many bridges are there?",
        "Show me bridges in Orange county",
        "Find bridges with more than 3 spans",
        "List bridges carrying highway 84i",
        "What are the bridge statistics by county?",
        "Show me the top 5 bridges by span count",
        "Orange county bridges carrying 84i with more than 4 spans",
        "Westchester bridges crossing river with at least 3 spans",
    ]

    print("Testing parser:")
    print("=" * 60)

    for query in test_queries:
        endpoint = parser.parse_query(query)
        print(f"Q: {query}")
        print(f"A: {endpoint}\n")

    print("=" * 60)


if __name__ == "__main__":
    test_parser()
