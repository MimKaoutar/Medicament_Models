"""
Transformer-based classifier training and evaluation script.

This script loads and tokenizes datasets, prepares a custom Torch dataset,
sets up training arguments, and trains/evaluates a transformer model using either
the default Trainer or a custom Trainer (with class imbalance handling).

Usage:
    python encoder_classifier.py --variation all --trainfile txt_presc_t3.csv --devfile txt_presc_t1.csv
    --testfile1 txt_presc_t1.csv --testfile2 txt_presc_t3.csv --variation all --model answerdotai/ModernBERT-base
    --lr 1e-5 --train 1 --context 8192 --epochs 10 --batch 32 --trainer_class default
"""

import os
import argparse
import json
import pickle
from pathlib import Path
from typing import List, Tuple, Dict, Union, Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset as TorchDataset
from datasets import Dataset, concatenate_datasets
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    IntervalStrategy,
)
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler

from utils import load_dataset, compute_metrics, check_device, default_converter

# Disable parallelism in tokenizers to avoid deadlocks
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Set random seed for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_STATE)

# Check available device
device = check_device()


def tokenize_dataset(
    dataset: Dataset,
    tokenizer: Any,
    max_length: int = 512,
    batch_size: int = 1000,
) -> Dataset:
    """
    Tokenizes text data in a Hugging Face Dataset with batching.

    Args:
        dataset: The dataset containing text data.
        tokenizer: The tokenizer to use.
        max_length: Maximum token length.
        batch_size: Batch size for tokenization.

    Returns:
        Dataset: The tokenized dataset.
    """
    def tokenize_function(examples: Dict[str, List[str]]) -> Dict[str, Any]:
        return tokenizer(
            examples["txt"],
            truncation=True,
            padding="max_length",  # Use max_length padding for consistent tensors
            max_length=max_length,
            return_tensors="pt",
        )

    tokenizer.model_max_length = max_length

    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        batch_size=batch_size,
        desc="Tokenizing dataset",
        num_proc=1, 
        remove_columns=["txt"],
    )

    return tokenized_dataset


class EfemerisDataset(TorchDataset):
    """
    Custom Torch dataset class for efficient data handling.
    """
    def __init__(self, tokenized_data: Dataset, labels: List[int]) -> None:
        self.tokenized_data = tokenized_data
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.tokenized_data[idx]
        item["labels"] = self.labels[idx]
        return item


def prepare_dataset(
    X_data: pd.DataFrame,
    y_data: pd.DataFrame,
    tokenizer: Any,
    label_column: str = "MALFO_MAJ",
    max_length: int = 512,
) -> EfemerisDataset:
    """
    Converts pandas DataFrames to a tokenized Torch dataset.

    Args:
        X_data: DataFrame of input features.
        y_data: DataFrame of labels.
        tokenizer: Tokenizer to use.
        label_column: Column name containing labels.
        max_length: Maximum token length.

    Returns:
        EfemerisDataset: Prepared dataset with tokenized inputs and labels.
    """
    print("Converting dataframes to Hugging Face Dataset...")
    tok_dataset = Dataset.from_pandas(X_data)
    tokenized_ds = tokenize_dataset(tok_dataset, tokenizer, max_length)

    print("Creating final dataset with labels...")
    dataset = EfemerisDataset(tokenized_ds, y_data[label_column].tolist())

    # Print a sample of model input for inspection
    print("\n----- SAMPLE MODEL INPUTS (first 2 examples) -----")
    for i in range(2):
        sample = tokenized_ds[i]
        decoded = tokenizer.decode(sample["input_ids"])[:200]
        print(f"Example {i+1}: {decoded}")
    print("----- END OF SAMPLE -----\n")
    return dataset


def setup_training_args(
        output_dir: str,
        evaluation_steps: int,
        batch_size: int,
        epochs: int,
        learning_rate: float,
        output_name: str,
        gradient_accumulation_steps: int = 1,
        fp16: bool = True,
        warmup_ratio: float = 0.1
) -> TrainingArguments:
    """
    Sets up training arguments for the transformer model with optimized settings.

    Args:
        output_dir: Directory to save model checkpoints.
        evaluation_steps: Number of steps between evaluations.
        batch_size: Batch size for training and evaluation.
        epochs: Number of training epochs.
        learning_rate: Learning rate.
        output_name: Name for the output logs.
        gradient_accumulation_steps: Number of steps to accumulate gradients.
        fp16: Whether to use mixed precision training.
        warmup_ratio: Ratio of steps for learning rate warmup.

    Returns:
        TrainingArguments object.
    """
    return TrainingArguments(
        output_dir=output_dir,
        eval_strategy=IntervalStrategy.STEPS,
        eval_steps=evaluation_steps,
        save_total_limit=2,
        save_steps=evaluation_steps,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,  # Larger batch size for evaluation
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=epochs,
        weight_decay=0.01,
        logging_dir=f'/deepia/inutero/efemeris/expes/logs/{output_name}_log',
        logging_steps=100,  # Log more frequently
        learning_rate=learning_rate,
        load_best_model_at_end=True,
        metric_for_best_model="auc-roc",
        greater_is_better=True,
        fp16=fp16,  # Enable mixed precision training
        warmup_ratio=warmup_ratio,  # Add warmup for better convergence
        report_to=["tensorboard"],
        dataloader_num_workers=4,  # Parallel data loading
        dataloader_pin_memory=True,  # Speeds up data transfer to GPU
    )


def create_compute_metrics_function():
    """
    Creates a metrics computation function for Trainer evaluation.

    Returns:
        A function that computes metrics from (logits, labels).
    """
    def compute_metrics_function(eval_pred: Tuple[np.ndarray, np.ndarray]) -> Dict[str, Any]:
        logits, labels = eval_pred
        logits = torch.tensor(logits)
        probabilities = torch.softmax(logits, dim=1)[:, 1].numpy()  # Positive class probability

        accuracy, f1, auc_roc, auc_prc, conf_matrix = compute_metrics(probabilities, labels)
        return {
            "auc-roc": auc_roc,
            "f1": f1,
            "auc-prc": auc_prc,
            #"confusion": np.array(conf_matrix).tolist(),
        }
    return compute_metrics_function


class CustomTrainer(Trainer):
    """
    Custom trainer that handles class imbalance by dynamically computing class weights
    and applying them to the CrossEntropyLoss function.
    """

    def __init__(self, model, train_dataset, **kwargs):
        super().__init__(model=model, train_dataset=train_dataset, **kwargs)
        # Compute class weights dynamically
        labels = train_dataset.labels
        unique_labels, counts = torch.unique(labels, return_counts=True)
        total_samples = len(labels)
        class_weights = total_samples / (len(unique_labels) * counts)
        # Convert to tensor and store for later use
        self.class_weights_tensor = torch.tensor(
            class_weights.tolist(),
            dtype=torch.float32,
            device=model.device
        )

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        # Forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")
        # Compute custom loss with class weights
        loss_fct = nn.CrossEntropyLoss(weight=self.class_weights_tensor)
        loss = loss_fct(
            logits.view(-1, self.model.config.num_labels),
            labels.view(-1)
        )
        return (loss, outputs) if return_outputs else loss


def transformer_classifier(
    data_dict: Dict[str, Any],
    output_name: str,
    model_name: str = "bert-base-uncased",
    context_length: int = 512,
    lr: float = 5e-5,
    trainer_class: str = "default",
    epochs: int = 8,
    batch_size: int = 32,
    gradient_accumulation_steps: int = 1,
    train_option: int = 1,
    resume: bool = False,
    fp16: bool = True,
) -> Dict[Any, Any]:
    """
    Trains and evaluates a transformer classifier.

    Args:
        data_dict: Dictionary with keys 'train', 'dev', and 'test'. Each key maps to a tuple of (X, y).
        output_name: Name used for output directories and logs.
        model_name: Pre-trained model identifier.
        context_length: Maximum sequence length.
        lr: Learning rate.
        trainer_class: Either 'default' or 'custom'.
        epochs: Number of epochs.
        batch_size: Batch size.
        gradient_accumulation_steps: Gradient accumulation steps.
        train_option: 1 to train, 0 to only evaluate.
        resume: Whether to resume from a checkpoint.
        fp16: Whether to use mixed precision.

    Returns:
        Dictionary of test metrics.
    """
    print(f"Loading tokenizer for model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    (X_train, y_train), (X_dev, y_dev) = data_dict["train"], data_dict["dev"]
    train_dataset = prepare_dataset(X_train, y_train, tokenizer, max_length=context_length)
    dev_dataset = prepare_dataset(X_dev, y_dev, tokenizer, max_length=context_length)

    model_config = {"num_labels": 2}
    model_dir = Path(f"/deepia/inutero/efemeris/expes/models/{output_name}")
    model_dir.mkdir(parents=True, exist_ok=True)

    # If not training, load saved best model
    if train_option != 1:
        print(f"Loading pre-trained model from {model_dir}")
        model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
    else:
        model = AutoModelForSequenceClassification.from_pretrained(model_name, **model_config)

    # Calculate training parameters
    total_samples = len(train_dataset)
    steps_per_epoch = total_samples // (batch_size * gradient_accumulation_steps)
    steps_for_eval = max(steps_per_epoch // 1, 1)  # Evaluate 1 times per epoch

    print(f"Training samples: {total_samples}, Steps per epoch: {steps_per_epoch}, Evaluation every {steps_for_eval} steps")

    # Setup training arguments with optimized settings
    training_args = setup_training_args(
        output_dir=str(model_dir),
        evaluation_steps=steps_for_eval,
        batch_size=batch_size,
        epochs=epochs,
        learning_rate=lr,
        output_name=output_name,
        gradient_accumulation_steps=gradient_accumulation_steps,
        fp16=fp16,
        warmup_ratio=0.1  # Warm up learning rate for 10% of training
    )

    # Create compute metrics function
    compute_metrics_fn = create_compute_metrics_function()

    # Initialize appropriate Trainer
    if trainer_class == 'default':
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=dev_dataset,
            compute_metrics=compute_metrics_fn,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        )
    else:
        # Use CustomTrainer for handling class imbalance
        trainer = CustomTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=dev_dataset,
            compute_metrics=compute_metrics_fn,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        )

    # Train if specified
    if train_option == 1:
        print("Starting training...")
        if torch.cuda.is_available():
            print(f"GPU memory before training: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB")
        trainer.train(resume_from_checkpoint=resume)
        if torch.cuda.is_available():
            print(f"GPU memory after training: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB")
        metrics = trainer.evaluate()
        print("Dev set evaluation metrics:", metrics)
        trainer.save_model(str(model_dir))
        print("Training finished")

    print("Starting evaluation on test sets...")
    test_metrics = {}

    for trim_id, X_test, y_test in zip([1,3], data_dict['test'][0], data_dict['test'][1]):
        test_dataset = prepare_dataset(X_test, y_test, tokenizer, max_length=context_length)
        print(f"Evaluating on test set {trim_id}...")
        test_results = trainer.evaluate(test_dataset)
        test_metrics[trim_id] = test_results
        print(f"Test set {trim_id} metrics: {test_results}")

    return test_metrics


def parse_arguments() -> argparse.Namespace:
    """
    Parses command line arguments.

    Returns:
        Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Train and evaluate a transformer classifier")
    parser.add_argument("--root_data", type=str, default="/deepia/inutero/efemeris/data/efemeris_txt_v022025",
                        help="Main data directory")
    parser.add_argument("--variation", type=str, default="all", choices=["only_hosp", "all"])
    parser.add_argument("--trainfile", type=str, default=None, help="Input train file")
    parser.add_argument("--devfile", type=str, default=None, help="Input dev file")
    parser.add_argument("--testfile1", type=str, default=None, help="Input test file 1")
    parser.add_argument("--testfile2", type=str, default=None, help="Input test file 2")
    parser.add_argument("--model", type=str, default="bert-base-uncased", help="Pre-trained model identifier")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--gradient_accumulation", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--fp16", type=int, default=1, choices=[0, 1], help="Use mixed precision (1) or not (0)")
    parser.add_argument("--lang", type=str, default="en", choices=["en", "fr"], help="Language version: en or fr")
    parser.add_argument("--train", type=int, default=1, choices=[0, 1], help="Train (1) or only evaluate (0)")
    parser.add_argument("--context", type=int, default=512, help="Max context length for model")
    parser.add_argument("--resume", type=int, default=0, choices=[0, 1], help="Resume previous training (1) or not (0)")
    parser.add_argument("--epochs", type=int, default=8, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=32, help="Batch size")
    parser.add_argument("--trainer_class", type=str, default="default", choices=["default", "custom"],
                        help="Trainer type")
    return parser.parse_args()


def main():
    """
    Main function to run the script.
    """
    # Parse command line arguments
    args = parse_arguments()
    print("Running with arguments:")
    print(args)

    # Convert resume argument to boolean
    is_resume = args.resume == 1
    use_fp16 = args.fp16 == 1

    # Log system information
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Total GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.2f} GB")

    # Load datasets
    print("Loading datasets...")
    [X_train, X_dev], [y_train, y_dev], X_all_test, Y_all_test = load_dataset(
        args.trainfile, args.devfile, [args.testfile1, args.testfile2], args.root_data,
        args.variation, lang=args.lang, encoder=True)

    print("Data loaded")
    print(f"Training set dimensions: {X_train.shape}, {y_train.shape}")
    print(f"Development set dimensions: {X_dev.shape}, {y_dev.shape}")
    for i, (X_test, y_test) in enumerate(zip(X_all_test, Y_all_test)):
        print(f"Test set {i+1} dimensions: {X_test.shape}, {y_test.shape}")

    # Setup output paths
    output_root = '/deepia/inutero/efemeris/expes/results'
    Path(output_root).mkdir(parents=True, exist_ok=True)

    # Create a descriptive output name
    model_safe_name = args.model.replace("/", "_")
    train_file_name = os.path.basename(args.trainfile).replace(".csv", "")
    output_name = (
        f'{args.lang}_split_rand_var_{args.variation}_{model_safe_name}_'
        f'trainer_{args.trainer_class}_lr_{args.lr}_{train_file_name}_'
        f'grad_accum_{args.gradient_accumulation}_fp16_{args.fp16}'
    )
    print(f"Output will be saved as: {output_name}")

    data_dict = {
        'train': (X_train, y_train),
        'dev': (X_dev, y_dev),
        'test': [X_all_test, Y_all_test]
    }

    # Run classifier
    test_metrics = transformer_classifier(
        data_dict=data_dict,
        output_name=output_name,
        model_name=args.model,
        context_length=args.context,
        lr=args.lr,
        epochs=args.epochs,
        batch_size=args.batch,
        gradient_accumulation_steps=args.gradient_accumulation,
        train_option=args.train,
        resume=is_resume,
        trainer_class=args.trainer_class,
        fp16=use_fp16
    )

    # Save results
    output_filename = os.path.join(output_root, output_name)
    print(f"Writing results to: {output_filename}")
    print("Test metrics:")
    print(test_metrics)

    with open(f"{output_filename}.json", "w") as f:
        json.dump(test_metrics, f, indent=4, default=default_converter)
    #with open(output_filename, 'wb') as handle:
    #    pickle.dump(test_metrics, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("Script completed successfully")


if __name__ == '__main__':
    main()

