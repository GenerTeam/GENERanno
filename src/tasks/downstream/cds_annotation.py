import argparse
import datetime
import os
import time
from typing import List, Tuple, Union, IO

import pandas as pd
import torch
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    PreTrainedTokenizer,
    PreTrainedModel,
    AutoModelForTokenClassification,
)

# Constants for label mapping
LABEL2CHAR = {
    "CDS": "+",
    "NON_CODING": "-",
}

# Mapping for numeric representation in parquet
CHAR2NUM = {"+": 1, "-": 0}


def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments for coding DNA sequence (CDS) annotation.

    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description="Downstream Task: Coding DNA Sequence (CDS) Annotation."
    )
    parser.add_argument(
        "--input_fasta",
        type=str,
        default="hf://datasets/GenerTeam/cds-annotation/examples/Bacillus_anthracis_plasmid.fasta",
        help="Path to input FASTA to be annotated.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="GenerTeam/GENERanno-prokaryote-0.5b-cds-annotator-preview",
        help="HuggingFace model path or name",
    )
    parser.add_argument(
        "--batch_size", type=int, default=4, help="Batch size for model inference"
    )
    parser.add_argument(
        "--dp_size",
        type=int,
        default=1,
        help="Number of GPUs to use for DataParallel (1 for single GPU)",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="./result",
        help="Directory to save the output predictions",
    )
    parser.add_argument(
        "--context_length",
        type=int,
        default=8192,
        help="Context length in base pairs (bp) for sequence extraction",
    )
    return parser.parse_args()


def setup_model(
    model_name: str, dp_size: int = 1
) -> Tuple[
    Union[PreTrainedModel, torch.nn.DataParallel], PreTrainedTokenizer, torch.device
]:
    """
    Load and setup the model with optional multi-GPU support.

    Args:
        model_name: Name or path of the HuggingFace model
        dp_size: Number of GPUs to use for DataParallel (1 for single GPU)

    Returns:
        tuple of (model, tokenizer, device)
    """
    print(f"🤗 Loading model from Hugging Face: {model_name}")
    start_time = time.time()

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    dtype_str = "bfloat16" if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else "float32"
    model = AutoModelForTokenClassification.from_pretrained(
        model_name, torch_dtype=getattr(torch, dtype_str), trust_remote_code=True
    )

    # Check available GPUs
    device = torch.device("cpu")
    if torch.cuda.is_available():
        available_gpus = torch.cuda.device_count()
        if dp_size > 1 and available_gpus >= dp_size:
            print(f"🚀 Using DataParallel with {dp_size} GPUs")
            model = torch.nn.DataParallel(model, device_ids=list(range(dp_size)))
            device = torch.device("cuda")
        elif dp_size > 1:
            print(f"⚠️ Requested {dp_size} GPUs but only {available_gpus} available")
            print(f"🔄 Using single GPU: {torch.cuda.get_device_name(0)}")
            device = torch.device("cuda:0")
        else:
            print(f"💻 Using single GPU: {torch.cuda.get_device_name(0)}")
            device = torch.device("cuda:0")
    else:
        print("⚠️ No GPU available, using CPU")

    model.to(device)
    model.eval()
    print(f"⏱️ Model loading completed in {time.time() - start_time:.2f} seconds")

    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    print(f"📊 Model size: {total_params / 1e6:.1f}M parameters")

    return model, tokenizer, device


def _parse_fasta_from_stream(stream: IO[str]) -> List[Tuple[str, str]]:
    """
    Parse FASTA data from an open text stream.

    Args:
        stream: An open text stream (file-like object).

    Returns:
        A list of (header, sequence) tuples.
    """
    records, header, seq_lines = [], None, []
    for line in stream:
        line = line.strip()
        if line.startswith(">"):
            if header is not None:
                records.append((header, "".join(seq_lines).upper()))
            header, seq_lines = line, []
        else:
            seq_lines.append(line)
    # Process the last sequence in the file
    if header is not None:
        records.append((header, "".join(seq_lines).upper()))
    return records


def read_fasta(path: str) -> List[Tuple[str, str]]:
    """
    Read sequences from a FASTA file, supporting local paths and hf:// URLs.

    Args:
        path: Path to the FASTA file (e.g., "local/file.fasta" or
              "hf://datasets/username/my_dataset/my_file.fasta").

    Returns:
        A list of (header, sequence) tuples.
    """
    records: List[Tuple[str, str]]

    if path.startswith("hf://"):
        try:
            import fsspec

            with fsspec.open(path, mode="rt", encoding="utf-8") as f:
                records = _parse_fasta_from_stream(f)
            print(f"✅ Read {len(records)} sequences from Hugging Face Hub: {path}")
        except ImportError:
            error_msg = (
                "⚠️ The library 'fsspec' is required for reading hf:// paths but it's not installed.\n"
                "Please install both 'fsspec' and 'huggingface_hub': pip install fsspec huggingface_hub"
            )
            print(error_msg)
            raise ImportError(error_msg)
        except Exception as e:
            # Catch other errors that might occur during fsspec.open or parsing
            print(f"❌ Error reading from Hugging Face Hub path {path}: {e}")
            raise  # Re-throw the exception to indicate failure
    else:  # Local file path
        try:
            with open(path, "r", encoding="utf-8") as f:
                records = _parse_fasta_from_stream(f)
            print(f"✅ Read {len(records)} sequences from local file: {path}")
        except FileNotFoundError:
            print(f"❌ Error: Local file not found at {path}")
            raise
        except Exception as e:
            print(f"❌ Error reading from local file {path}: {e}")
            raise

    return records


def write_fasta(records: List[Tuple[str, str]], path: str, width: int = 60) -> None:
    """
    Write sequences to a FASTA file with specified line width.

    Args:
        records: List of (header, sequence) tuples
        path: Output file path
        width: Line width for sequences (default: 60)
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for header, seq in records:
            f.write(header + "\n")
            for i in range(0, len(seq), width):
                f.write(seq[i : i + width] + "\n")
    print(f"✅ Annotated FASTA written to {path}")


def write_parquet(
    all_head_records: List[List[Tuple[str, str]]], head_names: List[str], path: str
) -> None:
    """
    Write annotations to a parquet file with record name and numeric predictions.

    Args:
        all_head_records: A list where each item is the list of records for a head.
        head_names: A list of names for each prediction head.
        path: Output parquet file path.
    """
    if not all_head_records:
        print("⚠️ No records to write to Parquet.")
        return

    data = []
    num_records = len(all_head_records[0])

    # Iterate through each sequence record
    for i in range(num_records):
        # Extract record name from the first head's record (it's the same for all)
        header, _ = all_head_records[0][i]
        record_name = (
            header[1:].split()[0] if header.startswith(">") else header.split()[0]
        )
        row_data = {"record_name": record_name}

        # Add predictions from each head as a new column
        for h, head_name in enumerate(head_names):
            _, annotation = all_head_records[h][i]
            # Use .get() for safety, defaulting to 0 for unknown characters
            numeric_labels = [CHAR2NUM.get(char, 0) for char in annotation]
            row_data[f"pred_{head_name}"] = numeric_labels

        data.append(row_data)

    # Create DataFrame and save as a single Parquet file
    df = pd.DataFrame(data)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    df.to_parquet(path, index=False)
    print(f"✅ Parquet file written to {path}")


@torch.no_grad()
def annotate_fasta(
    records: List[Tuple[str, str]],
    model: Union[PreTrainedModel, torch.nn.DataParallel],
    tokenizer: PreTrainedTokenizer,
    device: torch.device,
    max_length: int,
    micro_batch_size: int,
) -> List[List[Tuple[str, str]]]:
    """
    Annotate sequences in FASTA format with gene predictions.

    Batched version -- all sequences are chunked first, then inferred in a
    single loop of micro-batches. This mirrors the logic used in
    NucleotideClassificationInferenceTask while avoiding per-record overhead.

    Args:
        records: List of (header, sequence) tuples
        model: The pre-trained model for sequence annotation
        tokenizer: Tokenizer for the model
        device: Computation device (CPU or GPU)
        max_length: Maximum sequence length for chunking
        micro_batch_size: Batch size for model inference

    Returns:
        A list of lists of (header, annotation) tuples. The outer list
        corresponds to prediction heads, the inner list to records.
    """
    pad_id = tokenizer.pad_token_id
    model_module = model.module if isinstance(model, torch.nn.DataParallel) else model
    id2label = model_module.config.id2label
    num_heads = getattr(model_module, "num_prediction_heads", 1)
    print(f"🧬 Model has {num_heads} prediction head(s). Preparing sequences...")

    # Step 1: Chunk all sequences
    all_chunks, all_masks, token_lens = [], [], []
    for _, seq in records:
        ids = tokenizer(seq, add_special_tokens=False)["input_ids"]
        token_lens.append(len(ids))
        while ids:
            chunk = ids[:max_length]
            ids = ids[max_length:]
            pad_len = max_length - len(chunk)
            all_chunks.append(chunk + [pad_id] * pad_len)
            all_masks.append([1] * len(chunk) + [0] * pad_len)

    print(f"📏 Created {len(all_chunks)} chunks from {len(records)} sequences")

    # Step 2: Run inference and separate predictions by head
    print("🧠 Running model inference...")
    preds_per_head_flat = [[] for _ in range(num_heads)]

    for start in tqdm(
        range(0, len(all_chunks), micro_batch_size),
        desc="Model inference",
        unit="batch",
    ):
        end = min(start + micro_batch_size, len(all_chunks))
        inp = torch.tensor(all_chunks[start:end], dtype=torch.long).to(device)
        att = torch.tensor(all_masks[start:end], dtype=torch.long).to(device)

        logits = model(input_ids=inp, attention_mask=att).logits
        preds = logits.argmax(dim=-1).cpu()

        for i in range(preds.shape[0]):
            mask = att[i].cpu()
            valid_len = int(mask.sum())

            # The model concatenates head predictions, so we split them
            total_pred_len = valid_len * num_heads
            chunk_preds_all_heads = preds[i, :total_pred_len].tolist()

            for h in range(num_heads):
                start_idx = h * valid_len
                end_idx = (h + 1) * valid_len
                preds_per_head_flat[h].extend(chunk_preds_all_heads[start_idx:end_idx])

    # Step 3: Reconstruct full sequence annotations for each head
    print("🔍 Generating annotations from predictions...")
    all_annotated_records = []
    for h in range(num_heads):
        head_annotations = []
        cursor = 0
        for (header, seq), tok_len in zip(records, token_lens):
            seq_preds = preds_per_head_flat[h][cursor : cursor + tok_len]
            cursor += tok_len

            labels = [id2label[i] for i in seq_preds]
            annot = "".join(LABEL2CHAR[l] for l in labels)

            assert len(annot) == len(seq)

            head_annotations.append((header, annot))
        all_annotated_records.append(head_annotations)

    print(f"✅ Successfully generated annotations for {num_heads} head(s).")
    return all_annotated_records


def display_progress_header() -> None:
    """
    Display a stylized header for the CDS annotation pipeline.
    """
    print("\n" + "=" * 80)
    print("🧬  CODING DNA SEQUENCE (CDS) ANNOTATION PIPELINE  🧬")
    print("=" * 80 + "\n")


def main() -> None:
    """
    Main function to run the CDS annotation pipeline.
    """
    # Display header
    display_progress_header()

    # Start timer for total execution
    total_start_time = time.time()

    # Parse command line arguments
    args = parse_arguments()

    # Read input FASTA file
    print("🔄 Reading input FASTA file...")
    fasta_records = read_fasta(args.input_fasta)

    # Setup model and tokenizer with specified DP size
    model, tokenizer, device = setup_model(args.model_name, args.dp_size)

    annotated_records_per_head = annotate_fasta(
        fasta_records,
        model,
        tokenizer,
        device,
        max_length=args.context_length,
        micro_batch_size=args.batch_size,
    )

    # Define names for each prediction head for output files
    num_heads = len(annotated_records_per_head)
    if num_heads == 2:
        head_names = ["positive_strand", "negative_strand"]
    else:
        head_names = [f"head_{i}" for i in range(num_heads)]

    # Generate output filenames with timestamp and write files for each head
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    input_filename = os.path.basename(args.input_fasta)
    base_input_name = os.path.splitext(input_filename)[0]

    # Create output directory if it doesn't exist
    os.makedirs(args.output_path, exist_ok=True)

    # --- Write separate FASTA file for each head ---
    for head_name, annotated_records in zip(head_names, annotated_records_per_head):
        print(f"\n--- Writing FASTA output for: {head_name} ---")
        output_suffix = f"{timestamp}_{head_name}"
        fasta_output_path = os.path.join(
            args.output_path, f"{base_input_name}_{output_suffix}.fasta"
        )
        write_fasta(annotated_records, fasta_output_path)

    # --- Write a single Parquet file with all heads as columns ---
    if annotated_records_per_head:
        print("\n--- Writing multi-head Parquet output ---")
        parquet_output_path = os.path.join(
            args.output_path, f"{base_input_name}_{timestamp}.parquet"
        )
        write_parquet(annotated_records_per_head, head_names, parquet_output_path)

    # Print total execution time
    total_time = time.time() - total_start_time
    minutes, seconds = divmod(total_time, 60)
    print(f"\n⏱️ Total execution time: {int(minutes)}m {seconds:.2f}s")
    print("✨ Completed successfully! ✨\n")


if __name__ == "__main__":
    main()
