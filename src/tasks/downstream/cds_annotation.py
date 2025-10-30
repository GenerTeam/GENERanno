import os
# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import argparse
import datetime
import time
from typing import List, Tuple, Union, IO
import math

import pandas as pd
import torch
import torch.multiprocessing as mp
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
        help="Download from https://huggingface.co/datasets/GenerTeam/cds-annotation",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="GenerTeam/GENERanno-prokaryote-0.5b-cds-annotator",
        help="Download from https://huggingface.co/GenerTeam/GENERanno-prokaryote-0.5b-cds-annotator",
    )
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=4, 
        help="Batch size for model inference"
    )
    parser.add_argument(
        "--gpu_count",
        type=int,
        default=-1,
        help="Number of GPUs to use for parallel processing (-1 for all available GPUs)",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="./annotation_results",
        help="Directory to save the output predictions",
    )
    parser.add_argument(
        "--context_length",
        type=int,
        default=8192,
        help="Context length in base pairs (bp) for sequence extraction",
    )
    parser.add_argument(
        "--bf16",
        action="store_true",
        help="Use bfloat16 for faster inference",
    )
    return parser.parse_args()


def setup_model_for_gpu(
    model_name: str, gpu_id: int, dtype_str: str
) -> Tuple[PreTrainedModel, PreTrainedTokenizer, torch.device]:
    """
    Load and setup the model for a specific GPU.

    Args:
        model_name: Name or path of the HuggingFace model
        gpu_id: GPU device ID to use
        dtype_str: Data type string ('float32' or 'bfloat16')

    Returns:
        tuple of (model, tokenizer, device)
    """
    print(f"🤗 Loading model on GPU {gpu_id}: {model_name}")
    start_time = time.time()

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForTokenClassification.from_pretrained(
        model_name, torch_dtype=getattr(torch, dtype_str), trust_remote_code=True
    )

    device = torch.device(f"cuda:{gpu_id}" if gpu_id >= 0 else "cpu")
    model.to(device)
    model.eval()
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    print(f"📊 Device {device} model size: {total_params / 1e6:.1f}M parameters")
    print(f"⏱️ Model loading completed in {time.time() - start_time:.2f} seconds")

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


def distribute_sequences_to_gpus(
    records: List[Tuple[str, str]], gpu_count: int
) -> List[List[Tuple[str, str]]]:
    """
    Distribute sequences evenly across GPUs.

    Args:
        records: List of (header, sequence) tuples
        gpu_count: Number of GPUs to distribute across

    Returns:
        List of record lists, one for each GPU
    """
    sequences_per_gpu = math.ceil(len(records) / gpu_count)
    gpu_sequences = []
    
    for gpu_id in range(gpu_count):
        start_idx = gpu_id * sequences_per_gpu
        end_idx = min(start_idx + sequences_per_gpu, len(records))
        gpu_sequences.append(records[start_idx:end_idx])
        print(f"📋 GPU {gpu_id} assigned {len(gpu_sequences[gpu_id])} sequences")
    
    return gpu_sequences


@torch.no_grad()
def process_sequences_on_gpu(
    sequences: List[Tuple[str, str]],
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    device: torch.device,
    max_length: int,
    micro_batch_size: int,
    progress_bar: tqdm = None
) -> List[List[Tuple[str, str]]]:
    """
    Process sequences on a specific GPU.

    Args:
        sequences: List of (header, sequence) tuples to process
        model: The pre-trained model for sequence annotation
        tokenizer: Tokenizer for the model
        device: Computation device (CPU or GPU)
        max_length: Maximum sequence length for chunking
        micro_batch_size: Batch size for model inference
        progress_bar: Optional progress bar to update

    Returns:
        A list of lists of (header, annotation) tuples for the sequences
    """
    pad_id = tokenizer.pad_token_id
    id2label = model.config.id2label
    num_heads = getattr(model, "num_prediction_heads", 1)

    # Step 1: Prepare all sequences for this GPU
    all_chunks, all_masks, token_lens = [], [], []
    sequence_headers = []
    
    for header, seq in sequences:
        sequence_headers.append(header)
        ids = tokenizer(seq, add_special_tokens=False)["input_ids"]
        token_lens.append(len(ids))
        while ids:
            chunk_seq = ids[:max_length]
            ids = ids[max_length:]
            pad_len = max_length - len(chunk_seq)
            all_chunks.append(chunk_seq + [pad_id] * pad_len)
            all_masks.append([1] * len(chunk_seq) + [0] * pad_len)

    # Step 2: Run inference for sequences on this GPU
    preds_per_head_flat = [[] for _ in range(num_heads)]

    # Calculate how many sequences are processed in each batch
    # This is approximate since batches contain chunks, not whole sequences
    sequences_processed = 0
    total_chunks = len(all_chunks)
    
    for start in range(0, total_chunks, micro_batch_size):
        end = min(start + micro_batch_size, total_chunks)
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

        # Update progress bar if provided
        if progress_bar is not None:
            # Estimate progress based on chunks processed
            # Each batch processes multiple chunks, but we want sequence-level progress
            # We update more frequently for better visual feedback
            chunks_processed = min(end, total_chunks)
            progress_fraction = chunks_processed / total_chunks
            target_sequences = int(progress_fraction * len(sequences))
            if target_sequences > sequences_processed:
                progress_bar.update(target_sequences - sequences_processed)
                sequences_processed = target_sequences

    # Step 3: Reconstruct annotations for sequences on this GPU
    if progress_bar is not None:
        # Ensure progress bar reaches 100%
        remaining = len(sequences) - sequences_processed
        if remaining > 0:
            progress_bar.update(remaining)
    
    gpu_annotated_records = []
    for h in range(num_heads):
        head_annotations = []
        cursor = 0
        for header, (_, seq), tok_len in zip(sequence_headers, sequences, token_lens):
            seq_preds = preds_per_head_flat[h][cursor : cursor + tok_len]
            cursor += tok_len

            labels = [id2label[i] for i in seq_preds]
            annot = "".join(LABEL2CHAR[l] for l in labels)

            assert len(annot) == len(seq)
            head_annotations.append((header, annot))
        gpu_annotated_records.append(head_annotations)

    return gpu_annotated_records


def worker_process(
    gpu_id: int,
    model_name: str,
    dtype_str: str,
    sequences: List[Tuple[str, str]],
    max_length: int,
    micro_batch_size: int,
    result_queue: mp.Queue,
    progress_queue: mp.Queue,
):
    """
    Worker process that runs on a specific GPU.

    Args:
        gpu_id: GPU device ID to use
        model_name: HuggingFace model name
        sequences: List of sequences to process on this GPU
        max_length: Maximum sequence length
        micro_batch_size: Batch size for inference
        result_queue: Queue to put results in
        progress_queue: Queue to report progress
    """
    try:
        # Setup model for this GPU
        model, tokenizer, device = setup_model_for_gpu(model_name, gpu_id, dtype_str)
        
        # Process sequences assigned to this GPU (no progress bar in worker processes)
        gpu_results = process_sequences_on_gpu(
            sequences, model, tokenizer, device, max_length, micro_batch_size, progress_bar=None
        )
        result_queue.put((gpu_id, gpu_results))
        progress_queue.put(len(sequences))  # Report number of processed sequences
            
    except Exception as e:
        print(f"❌ Error in GPU {gpu_id} worker: {e}")
        result_queue.put(("error", gpu_id, str(e)))


def annotate_fasta(
    records: List[Tuple[str, str]],
    model_name: str,
    dtype_str: str,
    gpu_count: int,
    max_length: int,
    micro_batch_size: int,
) -> List[List[Tuple[str, str]]]:
    """
    Annotate sequences using single or multiple GPUs with independent model loading.

    Args:
        records: List of (header, sequence) tuples
        model_name: HuggingFace model name
        dtype_str: Data type string
        gpu_count: Number of GPUs to use (1 for single GPU)
        max_length: Maximum sequence length
        micro_batch_size: Batch size for inference

    Returns:
        A list of lists of (header, annotation) tuples
    """
    # Single GPU case - process directly with sequence-level progress bar
    if gpu_count <= 1:
        if torch.cuda.is_available() and gpu_count == 1:
            print("💻 Using single GPU processing")
            gpu_id = 0
        else:
            print("💻 Using single CPU processing. This could be SLOW!!!")
            gpu_id = -1
        
        model, tokenizer, device = setup_model_for_gpu(model_name, gpu_id, dtype_str)
        
        print(f"🧬 Processing {len(records)} sequences on {device}")
        
        # For single GPU, we use a sequence-level progress bar
        with tqdm(total=len(records), desc="Processing sequences", unit="seq") as pbar:
            # Pass the progress bar to the processing function
            results = process_sequences_on_gpu(
                records, model, tokenizer, device, max_length, micro_batch_size, progress_bar=pbar
            )
        
        return results
    
    # Multi-GPU case - use multiprocessing (保持原有代码不变)
    print(f"🚀 Starting parallel processing with {gpu_count} GPUs")
    print(f"📊 Distributing {len(records)} sequences across {gpu_count} GPUs")

    # Distribute sequences evenly across GPUs
    gpu_sequences = distribute_sequences_to_gpus(records, gpu_count)

    # Create queues for results and progress
    result_queue = mp.Queue()
    progress_queue = mp.Queue()

    # Start worker processes
    processes = []
    for gpu_id in range(gpu_count):
        if gpu_id < len(gpu_sequences) and gpu_sequences[gpu_id]:
            p = mp.Process(
                target=worker_process,
                args=(
                    gpu_id,
                    model_name,
                    dtype_str,
                    gpu_sequences[gpu_id],
                    max_length,
                    micro_batch_size,
                    result_queue,
                    progress_queue,
                ),
            )
            p.start()
            processes.append(p)

    # Collect results with progress bar
    total_sequences = len(records)
    results = [None] * gpu_count  # Store results by GPU ID
    
    with tqdm(total=total_sequences, desc="Processing sequences", unit="seq") as pbar:
        completed = 0
        while completed < total_sequences:
            # Update progress from progress queue
            while not progress_queue.empty():
                processed_count = progress_queue.get()
                pbar.update(processed_count)
                completed += processed_count
            
            # Check for results
            if not result_queue.empty():
                gpu_id, gpu_results = result_queue.get()
                if gpu_id == "error":
                    print(f"❌ Worker error: {gpu_results}")
                    continue
                
                results[gpu_id] = gpu_results
            
            time.sleep(0.1)  # Small delay to prevent busy waiting

    # Wait for all processes to finish
    for p in processes:
        p.join()

    # Combine results from all GPUs in order
    print("🔗 Combining results from all GPUs...")
    
    # Initialize empty lists for each head
    num_heads = len(results[0]) if results[0] is not None else 0
    all_annotated_records = [[] for _ in range(num_heads)]
    
    # Combine results in GPU order (which corresponds to sequence order)
    for gpu_id in range(gpu_count):
        if results[gpu_id] is not None:
            for head_idx in range(num_heads):
                all_annotated_records[head_idx].extend(results[gpu_id][head_idx])

    print(f"✅ Successfully processed {len(records)} sequences across {gpu_count} GPUs")
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

    dtype_str = "bfloat16" if args.bf16 else "float32"
    print(f"📊 Using dtype: {dtype_str}")

    # Read input FASTA file
    print("🔄 Reading input FASTA file...")
    fasta_records = read_fasta(args.input_fasta)

    # Determine GPU count
    available_gpus = torch.cuda.device_count()
    if args.gpu_count == -1:
        gpu_count = available_gpus
    else:
        gpu_count = min(args.gpu_count, available_gpus)
    
    print(f"🎯 Using {gpu_count} GPU(s) out of {available_gpus} available")

    # Use unified annotate function for both single and multi-GPU
    annotated_records_per_head = annotate_fasta(
        fasta_records,
        args.model_name,
        dtype_str,
        gpu_count,
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
    # Set multiprocessing start method
    mp.set_start_method('spawn', force=True)
    main()
