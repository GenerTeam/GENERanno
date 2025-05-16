import argparse
import os
import time
from typing import Dict, Tuple, Union, Optional, Callable

import numpy as np
import torch
import torch.distributed as dist
import transformers
import yaml
from datasets import (
    Dataset,
    load_dataset,
    DatasetDict,
    IterableDatasetDict,
    IterableDataset,
)
from sklearn.metrics import f1_score, matthews_corrcoef
from sklearn.model_selection import KFold
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    PreTrainedTokenizer,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    DataCollatorWithPadding,
    PreTrainedModel,
)

# Set logging level for transformers
transformers.logging.set_verbosity_info()

# Define optimization direction for each metric (whether higher or lower is better)
METRICS_DIRECTION: Dict[str, str] = {
    "accuracy": "max",
    "f1_score": "max",
    "mcc": "max",
    "f1_max": "max",
    "auprc_micro": "max",
    "mse": "min",
    "mae": "min",
    "r2": "max",
    "pearson": "max",
}


def is_main_process() -> bool:
    """
    Check if current process is the main process (rank 0) in distributed training.

    Returns:
        bool: True if this is the main process, False otherwise
    """
    if dist.is_initialized():
        return dist.get_rank() == 0
    return True


def dist_print(*args, **kwargs) -> None:
    """
    Print only from the main process (rank 0) in distributed training.
    Prevents duplicate outputs in multi-GPU settings.

    Args:
        *args: Arguments to pass to print function
        **kwargs: Keyword arguments to pass to print function
    """
    if is_main_process():
        print(*args, **kwargs)


def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments for sequence understanding fine-tuning.

    Returns:
        argparse.Namespace: Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description="Fine-tune a model for sequence understanding"
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        required=True,
        help="Name of the dataset on HuggingFace Hub",
    )
    parser.add_argument(
        "--subset_name",
        type=str,
        default=None,
        help="Name of the subset of the dataset (if applicable)",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="GenerTeam/GENERator-eukaryote-1.2b-base",
        help="HuggingFace model path or name",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size per GPU for training and evaluation",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=16384,
        help="Maximum sequence length for tokenization",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of steps to accumulate gradients before updating model",
    )
    parser.add_argument(
        "--padding_and_truncation_side",
        type=str,
        default="right",
        choices=["right", "left"],
        help="Which side to apply padding and truncation",
    )
    parser.add_argument(
        "--learning_rate", type=float, default=1e-5, help="Learning rate for training"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/sequence_understanding",
        help="Path to save the fine-tuned model",
    )
    parser.add_argument(
        "--num_folds",
        type=int,
        default=10,
        help="Number of folds for cross-validation (if splitting locally)",
    )
    parser.add_argument(
        "--fold_id",
        type=int,
        default=0,
        help="Fold ID for cross-validation (if splitting locally)",
    )
    parser.add_argument(
        "--main_metrics",
        type=str,
        default="mcc",
        help="Main metric for early stopping and model selection",
    )
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=5,
        help="Number of evaluations with no improvement after which training will be stopped",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--problem_type",
        type=str,
        default="single_label_classification",
        choices=[
            "single_label_classification",
            "multi_label_classification",
            "regression",
        ],
        help="Problem type for the task",
    )
    parser.add_argument(
        "--hf_config_path",
        type=str,
        default="configs/hf_configs/sequence_understanding.yaml",
        help="Path to the YAML configuration file for HuggingFace Trainer",
    )
    parser.add_argument(
        "--distributed_type",
        type=str,
        default="ddp",
        choices=["ddp", "deepspeed", "fsdp"],
        help="Type of distributed training to use",
    )
    return parser.parse_args()


def setup_dataset(
    dataset_name: str,
    subset_name: Optional[str] = None,
    tokenizer: Optional[PreTrainedTokenizer] = None,
    max_length: int = 16384,
    problem_type: str = "single_label_classification",
    seed: int = 42,
    num_folds: int = 0,
    fold_id: int = -1,
) -> Tuple[Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset], int]:
    """
    Load and prepare dataset for sequence understanding.

    Args:
        dataset_name: Name of the dataset on HuggingFace Hub
        subset_name: Name of the dataset subset (if applicable)
        tokenizer: Tokenizer for the model
        max_length: Maximum sequence length for tokenization
        problem_type: Type of problem (classification or regression)
        seed: Random seed for reproducibility
        num_folds: Number of folds for cross-validation (0 to use existing splits)
        fold_id: Current fold ID when using cross-validation

    Returns:
        Tuple of (preprocessed dataset, number of labels)
    """
    dist_print(f"📚 Loading dataset {dataset_name}...")
    start_time = time.time()

    # Load dataset from HuggingFace
    if subset_name is None:
        dataset = load_dataset(dataset_name, trust_remote_code=True)
    else:
        dataset = load_dataset(dataset_name, subset_name, trust_remote_code=True)
    dist_print(f"⚡ Dataset loaded in {time.time() - start_time:.2f} seconds")

    # Determine number of labels based on problem type
    if problem_type == "single_label_classification":
        assert isinstance(
            dataset["train"]["label"][0], int
        ), "Label must be an integer for single-label classification"
        max_label = max(dataset["train"]["label"])
        num_labels = max_label + 1
    elif problem_type == "multi_label_classification":
        assert isinstance(
            dataset["train"]["label"][0], list
        ), "Label must be a list for multi-label classification"
        assert isinstance(
            dataset["train"]["label"][0][0], float
        ), "Label values must be floats for multi-label classification"
        num_labels = len(dataset["train"]["label"][0])
    elif problem_type == "regression":
        if isinstance(dataset["train"]["label"][0], list):
            num_labels = len(dataset["train"]["label"][0])
        elif isinstance(dataset["train"]["label"][0], float):
            num_labels = 1
        else:
            raise NotImplementedError(
                "Regression with non-float labels is not supported yet."
            )
    else:
        raise ValueError(f"Unknown problem type: {problem_type}")

    assert num_labels is not None, "Number of labels could not be determined."

    # Create validation split if not present in the dataset
    if not any(x in dataset for x in ["validation", "valid", "val"]):
        # No validation set, split train into train and validation
        assert (
            num_folds > 0 and fold_id >= 0
        ), "No validation set found. Please provide a valid setting for cross-validation."

        dist_print(
            f"Performing {num_folds}-fold cross-validation (using fold {fold_id})"
        )

        kfold = KFold(
            n_splits=num_folds,
            shuffle=True,
            random_state=seed,
        )
        train_data_list = list(dataset["train"])
        splits = list(kfold.split(train_data_list))
        train_idx, valid_idx = splits[fold_id]
        dataset["validation"] = dataset["train"].select(valid_idx)
        dataset["train"] = dataset["train"].select(train_idx)

    # Process dataset with tokenizer
    def _process_function(examples):
        # Find the correct field containing the sequence
        if "sequence" in examples:
            sequences = examples["sequence"]
        elif "seq" in examples:
            sequences = examples["seq"]
        elif "dna_sequence" in examples:
            sequences = examples["dna_sequence"]
        elif "dna_seq" in examples:
            sequences = examples["dna_seq"]
        elif "text" in examples:
            sequences = examples["text"]
        else:
            raise ValueError(
                "No sequence column found in dataset. Expected 'sequence', 'seq', 'dna_sequence', 'dna_seq', or 'text'."
            )

        # Tokenize sequences
        tokenized = tokenizer(
            sequences,
            truncation=True,
            max_length=max_length,
            add_special_tokens=True,
            padding=False,
        )

        # Create attention masks manually
        tokenized["attention_mask"] = [
            [1] * len(input_id) for input_id in tokenized["input_ids"]
        ]
        tokenized["label"] = examples["label"]
        return tokenized

    # Apply tokenization to dataset
    dataset = dataset.map(
        _process_function,
        batched=True,
        remove_columns=[
            col
            for col in dataset["train"].column_names
            if col not in ["input_ids", "label"]
        ],
    )

    return dataset, num_labels


def setup_tokenizer(
    model_name: str, padding_and_truncation_side: str
) -> PreTrainedTokenizer:
    """
    Load and configure tokenizer for sequence understanding.

    Args:
        model_name: Name or path of the HuggingFace model
        padding_and_truncation_side: Side for padding and truncation (left or right)

    Returns:
        PreTrainedTokenizer: Configured tokenizer for the model
    """
    dist_print(f"🔤 Loading tokenizer from: {model_name}")
    start_time = time.time()

    # Load tokenizer with trust_remote_code to support custom models
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Configure padding and truncation settings
    tokenizer.padding_side = padding_and_truncation_side
    tokenizer.truncation_side = padding_and_truncation_side

    # Set pad_token to eos_token if not defined
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dist_print(
        f"⏱️ Tokenizer loading completed in {time.time() - start_time:.2f} seconds"
    )

    return tokenizer


def setup_model(model_name: str, problem_type: str, num_labels: int) -> PreTrainedModel:
    """
    Load and configure model for sequence understanding.

    Args:
        model_name: Name or path of the HuggingFace model
        problem_type: Type of problem
        num_labels: Number of labels for the task

    Returns:
        PreTrainedModel: Configured pre-trained model for sequence classification
    """
    dist_print(
        f"🤗 Loading AutoModelForSequenceClassification from: {model_name} with {num_labels} labels"
    )
    start_time = time.time()

    # Load model with appropriate problem type and label configuration
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        problem_type=problem_type,
        trust_remote_code=True,
    )

    # Ensure pad_token_id is set
    if model.config.pad_token_id is None:
        model.config.pad_token_id = model.config.eos_token_id

    # Report model size for reference
    total_params = sum(p.numel() for p in model.parameters())
    dist_print(f"📊 Model size: {total_params / 1e6:.1f}M parameters")
    dist_print(f"⏱️ Model loading completed in {time.time() - start_time:.2f} seconds")

    return model


def get_compute_metrics_func(problem_type: str, num_labels: int) -> Callable:
    """
    Get the appropriate compute_metrics function based on problem type.

    Args:
        problem_type: Type of problem (classification or regression)
        num_labels: Number of labels for the task

    Returns:
        Callable: A function to compute metrics for the given problem type
    """

    def _compute_metrics_single_label_classification(eval_pred):
        """
        Compute metrics for single-label classification.

        Args:
            eval_pred: Tuple of (logits, labels)

        Returns:
            Dict of metrics: accuracy, F1 score, and Matthews correlation coefficient
        """
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)

        accuracy = (predictions == labels).mean()
        f1 = f1_score(labels, predictions, average="weighted")
        mcc = matthews_corrcoef(labels, predictions)

        return {"accuracy": accuracy, "f1_score": f1, "mcc": mcc}

    def _compute_metrics_multi_label_classification(eval_pred):
        """
        Compute metrics for multi-label classification.

        Args:
            eval_pred: Tuple of (predictions, labels)

        Returns:
            Dict of metrics: F1 max and area under precision-recall curve
        """
        predictions, labels = eval_pred

        return {
            "f1_max": f1_max(torch.tensor(predictions), torch.tensor(labels)),
            "auprc_micro": area_under_prc(
                torch.tensor(predictions).flatten(),
                torch.tensor(labels).long().flatten(),
            ),
        }

    def _compute_metrics_regression(eval_pred):
        """
        Compute metrics for regression tasks.

        Args:
            eval_pred: Tuple of (predictions, labels)

        Returns:
            Dict of metrics: MSE, MAE, R², Pearson correlation, both per dimension and overall
        """
        logits, labels = eval_pred
        predictions = logits.squeeze()
        labels = labels.squeeze()

        # Reshape if needed
        if predictions.ndim == 1:
            predictions = predictions.reshape(-1, num_labels)
        if labels.ndim == 1:
            labels = labels.reshape(-1, num_labels)

        results = {}

        # Calculate metrics per dimension if multi-dimensional
        if num_labels > 1:
            label_names = [f"label_{i}" for i in range(num_labels)]

            for idx, label in enumerate(label_names):
                pred = predictions[:, idx]
                true = labels[:, idx]

                # MSE
                mse = np.mean((pred - true) ** 2)
                results[f"mse_{label}"] = mse

                # MAE
                mae = np.mean(np.abs(pred - true))
                results[f"mae_{label}"] = mae

                # R²
                y_mean = np.mean(true)
                ss_tot = np.sum((true - y_mean) ** 2)
                ss_res = np.sum((true - pred) ** 2)
                r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else float("nan")
                results[f"r2_{label}"] = r2

                # Pearson
                x_mean = np.mean(pred)
                numerator = np.sum((pred - x_mean) * (true - y_mean))
                denominator = np.sqrt(
                    np.sum((pred - x_mean) ** 2) * np.sum((true - y_mean) ** 2)
                )
                pearson = numerator / denominator if denominator != 0 else float("nan")
                results[f"pearson_{label}"] = pearson

        # Calculate overall metrics across all dimensions
        total_mse = np.mean((predictions - labels) ** 2)
        total_mae = np.mean(np.abs(predictions - labels))
        total_y_mean = np.mean(labels)
        total_ss_tot = np.sum((labels - total_y_mean) ** 2)
        total_ss_res = np.sum((labels - predictions) ** 2)
        total_r2 = (
            1 - (total_ss_res / total_ss_tot) if total_ss_tot != 0 else float("nan")
        )
        total_pred_mean = np.mean(predictions)
        total_numerator = np.sum(
            (predictions - total_pred_mean) * (labels - total_y_mean)
        )
        total_denominator = np.sqrt(
            np.sum((predictions - total_pred_mean) ** 2)
            * np.sum((labels - total_y_mean) ** 2)
        )
        total_pearson = (
            total_numerator / total_denominator
            if total_denominator != 0
            else float("nan")
        )

        results["mse"] = total_mse
        results["mae"] = total_mae
        results["r2"] = total_r2
        results["pearson"] = total_pearson

        return results

    def area_under_prc(pred, target):
        """
        Calculate area under precision-recall curve (PRC).

        Args:
            pred (Tensor): predictions of shape (n,)
            target (Tensor): binary targets of shape (n,)

        Returns:
            float: Area under the precision-recall curve
        """
        order = pred.argsort(descending=True)
        target = target[order]
        precision = target.cumsum(0) / torch.arange(
            1, len(target) + 1, device=target.device
        )
        auprc = precision[target == 1].sum() / ((target == 1).sum() + 1e-10)
        return auprc

    def f1_max(pred, target):
        """
        Calculate F1 score with the optimal threshold.

        This function enumerates all possible thresholds for deciding positive and negative
        samples, and picks the threshold with the maximal F1 score.

        Args:
            pred (Tensor): predictions of shape (B, N)
            target (Tensor): binary targets of shape (B, N)

        Returns:
            float: Maximum achievable F1 score across all thresholds
        """
        order = pred.argsort(descending=True, dim=1)
        target = target.gather(1, order)
        precision = target.cumsum(1) / torch.ones_like(target).cumsum(1)
        recall = target.cumsum(1) / (target.sum(1, keepdim=True) + 1e-10)
        is_start = torch.zeros_like(target).bool()
        is_start[:, 0] = 1
        is_start = torch.scatter(is_start, 1, order, is_start)

        all_order = pred.flatten().argsort(descending=True)
        order = (
            order
            + torch.arange(order.shape[0], device=order.device).unsqueeze(1)
            * order.shape[1]
        )
        order = order.flatten()
        inv_order = torch.zeros_like(order)
        inv_order[order] = torch.arange(order.shape[0], device=order.device)
        is_start = is_start.flatten()[all_order]
        all_order = inv_order[all_order]
        precision = precision.flatten()
        recall = recall.flatten()
        all_precision = precision[all_order] - torch.where(
            is_start, torch.zeros_like(precision), precision[all_order - 1]
        )
        all_precision = all_precision.cumsum(0) / is_start.cumsum(0)
        all_recall = recall[all_order] - torch.where(
            is_start, torch.zeros_like(recall), recall[all_order - 1]
        )
        all_recall = all_recall.cumsum(0) / pred.shape[0]
        all_f1 = 2 * all_precision * all_recall / (all_precision + all_recall + 1e-10)
        return all_f1.max()

    # Return the appropriate metrics function based on problem type
    if problem_type == "single_label_classification":
        return _compute_metrics_single_label_classification
    elif problem_type == "multi_label_classification":
        return _compute_metrics_multi_label_classification
    elif problem_type == "regression":
        return _compute_metrics_regression
    else:
        raise ValueError(f"Unknown problem type: {problem_type}")


def setup_training_args(yaml_path=None, cli_args=None, **kwargs):
    """
    Create a TrainingArguments instance from YAML, CLI arguments, and code arguments.
    Priority: code kwargs > CLI args > YAML config

    Args:
        yaml_path: Path to YAML configuration file
        cli_args: Parsed command line arguments
        **kwargs: Direct arguments that take highest priority

    Returns:
        TrainingArguments: Configured training arguments
    """
    # Start with yaml configuration if provided
    yaml_kwargs = {}
    if yaml_path and os.path.exists(yaml_path):
        with open(yaml_path, "r") as f:
            yaml_kwargs = yaml.safe_load(f)

    # Create a dictionary from CLI arguments
    cli_kwargs = {}
    if cli_args is not None:
        # Add basic training parameters
        if hasattr(cli_args, "output_dir"):
            cli_kwargs["output_dir"] = cli_args.output_dir
        if hasattr(cli_args, "batch_size"):
            cli_kwargs["per_device_train_batch_size"] = cli_args.batch_size
            cli_kwargs["per_device_eval_batch_size"] = cli_args.batch_size
        if hasattr(cli_args, "learning_rate"):
            cli_kwargs["learning_rate"] = cli_args.learning_rate
        if hasattr(cli_args, "gradient_accumulation_steps"):
            cli_kwargs["gradient_accumulation_steps"] = (
                cli_args.gradient_accumulation_steps
            )
        if hasattr(cli_args, "seed"):
            cli_kwargs["seed"] = cli_args.seed
            cli_kwargs["data_seed"] = cli_args.seed

        # Handle distributed training configurations
        if hasattr(cli_args, "distributed_type"):
            if cli_args.distributed_type == "deepspeed":
                cli_kwargs["deepspeed"] = "configs/ds_configs/zero1.json"
            elif cli_args.distributed_type == "fsdp":
                cli_kwargs["fsdp"] = "shard_grad_op auto_wrap"
                cli_kwargs["fsdp_config"] = "configs/ds_configs/fsdp.json"

    # Handle BF16 precision based on GPU capability
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
        cli_kwargs["bf16"] = True

    # Check the main metrics
    if cli_args.problem_type == "regression":
        if cli_args.main_metrics not in ["mse", "mae", "r2", "pearson"]:
            dist_print(
                f"⚠️ Warning: {cli_args.main_metrics} is not a valid metric for regression. Defaulting to 'mse'."
            )
            cli_args.main_metrics = "mse"
    elif cli_args.problem_type == "single_label_classification":
        if cli_args.main_metrics not in ["accuracy", "f1_score", "mcc"]:
            dist_print(
                f"⚠️ Warning: {cli_kwargs['main_metrics']} is not a valid metric for single-label classification. Defaulting to 'mcc'."
            )
            cli_args.main_metrics = "mcc"
    elif cli_args.problem_type == "multi_label_classification":
        if cli_args.main_metrics not in ["f1_max", "auprc_micro"]:
            dist_print(
                f"⚠️ Warning: {cli_kwargs['main_metrics']} is not a valid metric for multi-label classification. Defaulting to 'f1_max'."
            )
            cli_args.main_metrics = "f1_max"
    else:
        raise ValueError(
            f"Unknown problem type: {cli_args.problem_type}. Cannot determine main metrics."
        )

    # Handle metrics direction
    if hasattr(cli_args, "main_metrics") and "METRICS_DIRECTION" in globals():
        cli_kwargs["greater_is_better"] = (
            METRICS_DIRECTION[cli_args.main_metrics] == "max"
        )

    # Update lr_scheduler_kwargs if needed
    if (
        "lr_scheduler_kwargs" in yaml_kwargs
        and hasattr(cli_args, "main_metrics")
        and "METRICS_DIRECTION" in globals()
    ):
        if (
            isinstance(yaml_kwargs["lr_scheduler_kwargs"], dict)
            and "mode" in yaml_kwargs["lr_scheduler_kwargs"]
        ):
            yaml_kwargs["lr_scheduler_kwargs"]["mode"] = METRICS_DIRECTION[
                cli_args.main_metrics
            ]

    # Merge all configurations, with priority: kwargs > cli_kwargs > yaml_kwargs
    final_kwargs = {**yaml_kwargs, **cli_kwargs, **kwargs}

    # Create and return the TrainingArguments instance
    return TrainingArguments(**final_kwargs)


def train_model(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    datasets: Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset],
    args: argparse.Namespace,
) -> Trainer:
    """
    Train the model for sequence understanding.

    Args:
        model: Pre-trained language model
        tokenizer: Tokenizer for the model
        datasets: Dictionary containing train, validation, and test datasets
        args: Command line arguments

    Returns:
        Trainer: Trained model trainer
    """
    dist_print("🚀 Setting up training...")
    start_time = time.time()

    # Configure training arguments
    training_args = setup_training_args(yaml_path=args.hf_config_path, cli_args=args)

    # Setup early stopping callback
    early_stopping_callback = EarlyStoppingCallback(
        early_stopping_patience=args.early_stopping_patience
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["validation"],
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=get_compute_metrics_func(
            args.problem_type, model.config.num_labels
        ),
        callbacks=[early_stopping_callback],
    )

    dist_print(f"⏱️ Training setup completed in {time.time() - start_time:.2f} seconds")
    dist_print("🏋️ Starting model training...")
    training_start_time = time.time()

    # Train the model
    trainer.train()

    dist_print(
        f"✅ Training completed in {(time.time() - training_start_time) / 60:.2f} minutes"
    )
    return trainer


def evaluate_model(trainer: Trainer, test_dataset: Dataset) -> Dict[str, float]:
    """
    Evaluate the fine-tuned model on the test dataset.

    Args:
        trainer: Trained model trainer
        test_dataset: Test dataset

    Returns:
        Dict[str, float]: Dictionary of evaluation metrics
    """
    dist_print("📊 Evaluating model on test dataset...")
    start_time = time.time()

    # Run evaluation
    test_results = trainer.evaluate(test_dataset)

    dist_print(f"⏱️ Evaluation completed in {time.time() - start_time:.2f} seconds")

    # Prepare readable results
    metrics = {k.replace("eval_", "test_"): v for k, v in test_results.items()}

    return metrics


def save_model(
    trainer: Trainer, tokenizer: PreTrainedTokenizer, output_dir: str
) -> None:
    """
    Save the fine-tuned model and tokenizer.

    Args:
        trainer: Trained model trainer
        tokenizer: Tokenizer for the model
        output_dir: Directory to save the model
    """
    dist_print(f"💾 Saving fine-tuned model to {output_dir}")
    start_time = time.time()

    # Save the model
    trainer.save_model(output_dir)

    # Save the tokenizer
    tokenizer.save_pretrained(output_dir)

    dist_print(f"✅ Model saved in {time.time() - start_time:.2f} seconds")


def display_progress_header() -> None:
    """
    Display a stylized header for the sequence understanding fine-tuning.
    """
    dist_print("\n" + "=" * 80)
    dist_print("🔥  SEQUENCE UNDERSTANDING FINE-TUNING PIPELINE  🔥")
    dist_print("=" * 80 + "\n")


def main() -> None:
    """
    Main function to run the sequence fine-tuning pipeline.
    """
    # Display header
    display_progress_header()

    # Start timer for total execution
    total_start_time = time.time()

    # Parse command line arguments
    args = parse_arguments()

    # Set seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Setup tokenizer first
    tokenizer = setup_tokenizer(args.model_name, args.padding_and_truncation_side)

    # Load and prepare data
    datasets, num_labels = setup_dataset(
        args.dataset_name,
        args.subset_name,
        tokenizer,
        args.max_length,
        args.problem_type,
        args.seed,
        args.num_folds,
        args.fold_id,
    )

    # Now initialize model with correct number of labels
    model = setup_model(args.model_name, args.problem_type, num_labels)

    # Train model
    trainer = train_model(model, tokenizer, datasets, args)

    # Evaluate on test set
    test_metrics = evaluate_model(trainer, datasets["test"])

    # Print results
    dist_print("\n" + "=" * 80)
    dist_print(f"🏆 EVALUATION RESULTS FOR {args.model_name} 🏆")
    dist_print("=" * 80)
    for metric, value in test_metrics.items():
        if metric.startswith("test_"):
            dist_print(f"📊 {metric[5:].upper()}: {value:.4f}")
    dist_print("=" * 80)

    # Save fine-tuned model
    save_model(trainer, tokenizer, os.path.join(args.output_dir, "best_model"))

    # Print total execution time
    total_time = time.time() - total_start_time
    minutes, seconds = divmod(total_time, 60)
    dist_print(f"\n⏱️ Total execution time: {int(minutes)}m {seconds:.2f}s")
    dist_print("✨ Fine-tuning completed successfully! ✨\n")


if __name__ == "__main__":
    main()
