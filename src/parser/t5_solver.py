"""
T5 Solver for sequence-to-sequence training (e.g., natural language to API endpoint translation).

Implements the abstract methods from BaseSolver for T5 models.
"""

import logging
import torch
import json
from typing import Dict, List, Tuple
from pathlib import Path
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import T5Tokenizer, T5ForConditionalGeneration
from config.logging_config import configure_logger
from config_schema import T5TrainingConfig
from solver import BaseSolver

logger = logging.getLogger(__name__)
configure_logger(log_level="INFO", logger=logger)


class T5Dataset(Dataset):
    """Dataset for T5 seq2seq training."""

    def __init__(
        self,
        data: List[Dict[str, str]],
        tokenizer: T5Tokenizer,
        max_source_length: int = 128,
        max_target_length: int = 64,
    ):
        """
        Args:
            data: List of dicts with 'input' and 'output' keys
            tokenizer: T5 tokenizer
            max_source_length: Max length for source sequences
            max_target_length: Max length for target sequences
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]

        source = self.tokenizer(
            item["input"],
            max_length=self.max_source_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        target = self.tokenizer(
            item["output"],
            max_length=self.max_target_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        labels = target["input_ids"].squeeze()
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": source["input_ids"].squeeze(),
            "attention_mask": source["attention_mask"].squeeze(),
            "labels": labels,
        }


class T5Solver(BaseSolver):
    """Solver for training T5 models on seq2seq tasks."""

    def __init__(self, config: T5TrainingConfig):
        """
        Initialize T5 solver from validated config.
        
        Args:
            config: Validated T5 configuration object
        """
        # Load tokenizer BEFORE calling super().__init__()
        # because _load_dataset() needs it
        self.tokenizer = T5Tokenizer.from_pretrained(config.model_name, legacy=False)
        
        super().__init__(config)

    def _load_model(self):
        """Load T5 model from Hugging Face or local checkpoint."""
        self.accelerator.print(f"Loading T5 model: {self.config.model_name}")
        
        try:
            model = T5ForConditionalGeneration.from_pretrained(self.config.model_name)
            self.accelerator.print(f"Loaded model from {self.config.model_name}")
            return model
        except Exception as e:
            raise ValueError(f"Failed to load model {self.config.model_name}: {e}")

    def _load_dataset(self) -> Tuple[DataLoader, DataLoader]:
        """Load training data and create dataloaders (all types pre-validated)."""
        data_path = Path(self.config.data_path)
        if not data_path.exists():
            raise FileNotFoundError(f"Training data not found: {data_path}")

        with open(data_path, "r") as f:
            data = json.load(f)

        self.accelerator.print(f"Loaded {len(data)} training examples from {data_path}")

        # Tokenizer already loaded in __init__
        train_size = int(len(data) * (1 - self.config.val_split))
        val_size = len(data) - train_size

        torch.manual_seed(self.config.seed)
        train_data, val_data = random_split(data, [train_size, val_size])

        self.accelerator.print(f"Train: {len(train_data)} | Val: {len(val_data)}")

        train_dataset = T5Dataset(
            [data[i] for i in train_data.indices],
            self.tokenizer,
            self.config.max_source_length,
            self.config.max_target_length,
        )

        val_dataset = T5Dataset(
            [data[i] for i in val_data.indices],
            self.tokenizer,
            self.config.max_source_length,
            self.config.max_target_length,
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=True,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=True,
        )

        return train_loader, val_loader

    def _train_step(self, epoch: int) -> Dict[str, float]:
        """Execute one training epoch."""
        from solver import AverageMeter

        losses = AverageMeter()
        self.model.train()

        for batch_idx, batch in enumerate(self.train_loader):
            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )

            loss = outputs.loss

            # Backward pass
            self.accelerator.backward(loss)

            # Gradient clipping
            if self.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.max_grad_norm,
                )

            self.optimizer.step()
            if self.scheduler:
                self.scheduler.step()

            # Track loss
            losses.update(loss.item(), batch["input_ids"].size(0))

            # Log progress
            if batch_idx % self.config.log_interval == 0:
                self.accelerator.print(
                    f"Epoch [{epoch+1}/{self.epochs}] "
                    f"Batch [{batch_idx}/{len(self.train_loader)}] "
                    f"Loss: {losses.val:.4f} (avg: {losses.avg:.4f})"
                )

            self.global_step += 1

        # Gather losses across processes
        loss_tensor = torch.tensor(losses.avg, device=self.accelerator.device)
        gathered_loss = self.accelerator.gather(loss_tensor).mean().item()

        return {"loss": gathered_loss}

    def _evaluate(self, epoch: int) -> Dict[str, float]:
        """Evaluate on validation set."""
        from solver import AverageMeter

        losses = AverageMeter()
        exact_matches = 0
        total = 0

        self.model.eval()

        with torch.no_grad():
            for batch_idx, batch in enumerate(self.val_loader):
                # Forward pass
                outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                )

                loss = outputs.loss
                losses.update(loss.item(), batch["input_ids"].size(0))

                # Generate predictions for exact match
                generated = self.model.generate(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    max_length=self.config.max_target_length,
                    num_beams=self.config.num_beams,
                )

                # Decode and compare
                for gen, label in zip(generated, batch["labels"]):
                    # Filter out -100 (padding) from labels
                    label = label[label != -100]

                    pred_text = self.tokenizer.decode(gen, skip_special_tokens=True)
                    true_text = self.tokenizer.decode(label, skip_special_tokens=True)

                    if pred_text.strip() == true_text.strip():
                        exact_matches += 1
                    total += 1

                if batch_idx % self.config.log_interval == 0:
                    self.accelerator.print(
                        f"Val Epoch [{epoch+1}] "
                        f"Batch [{batch_idx}/{len(self.val_loader)}] "
                        f"Loss: {losses.val:.4f}"
                    )

        # Gather metrics across processes
        loss_tensor = torch.tensor(losses.avg, device=self.accelerator.device)
        exact_tensor = torch.tensor(exact_matches, device=self.accelerator.device)
        total_tensor = torch.tensor(total, device=self.accelerator.device)

        gathered_loss = self.accelerator.gather(loss_tensor).mean().item()
        gathered_exact = self.accelerator.gather(exact_tensor).sum().item()
        gathered_total = self.accelerator.gather(total_tensor).sum().item()

        accuracy = gathered_exact / gathered_total if gathered_total > 0 else 0.0

        return {
            "loss": gathered_loss,
            "accuracy": accuracy,
            "exact_match": accuracy,  # same as accuracy for seq2seq
        }

    def _test_model(self, checkpoint_path: str):
        """Test model on test queries."""
        # Load checkpoint
        state_dict = torch.load(checkpoint_path, map_location=self.accelerator.device)
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        unwrapped_model.load_state_dict(state_dict)

        # Test queries from config
        if not self.config.test_queries:
            self.accelerator.print("No test queries provided in config")
            return

        self.accelerator.print("\n" + "=" * 60)
        self.accelerator.print("TESTING MODEL")
        self.accelerator.print("=" * 60)

        self.model.eval()
        
        for query in self.config.test_queries:
            input_text = f"translate to api: {query}"
            inputs = self.tokenizer(
                input_text,
                return_tensors="pt",
                max_length=self.config.max_source_length,
                truncation=True,
            ).to(self.accelerator.device)

            with torch.no_grad():
                generated = unwrapped_model.generate(
                    **inputs,
                    max_length=self.config.max_target_length,
                    num_beams=self.config.num_beams,
                )

            output = self.tokenizer.decode(generated[0], skip_special_tokens=True)
            self.accelerator.print(f"\nQuery: {query}")
            self.accelerator.print(f"Output: {output}")

        self.accelerator.print("=" * 60 + "\n")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python t5_solver.py <config_path> [checkpoint_path] [--test]")
        sys.exit(1)

    config_path = sys.argv[1]
    checkpoint_path = sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].startswith("--") else None
    mode = "test" if "--test" in sys.argv else "train"

    solver = T5Solver(config_path)
    solver.run(checkpoint_path=checkpoint_path, mode=mode)
