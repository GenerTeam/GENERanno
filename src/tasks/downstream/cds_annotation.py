import os
# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import datetime
import signal
import threading
import time
from typing import Dict, IO, List, Optional, Tuple, Union

import pandas as pd
import torch
import numpy as np
import torch.multiprocessing as mp
from tqdm import tqdm
from numba import njit
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

PRESET_DEFAULTS = {
    "eukaryote": {
        "input": ["hf://datasets/GenerTeam/cds-annotation/examples/fly_GCF_000001215.4.parquet"],
        "model_name": "GenerTeam/GENERanno-eukaryote-1.2b-cds-annotator-preview",
        "output_path": "./eukaryote_annotation_results",
        "context_length": 16384,
        "overlap_length": 1024,
        "postprocess_stair_outward_shift": 64,
        "postprocess_stair_inward_shift": 16,
        "postprocess_min_length": 4,
    },
    "prokaryote": {
        "input": ["hf://datasets/GenerTeam/cds-annotation/examples/Escherichia_coli_genome.fasta"],
        "model_name": "GenerTeam/GENERanno-prokaryote-0.5b-cds-annotator",
        "output_path": "./prokaryote_annotation_results",
        "context_length": 8192,
        "overlap_length": 512,
        "postprocess_stair_outward_shift": 64,
        "postprocess_stair_inward_shift": 16,
        "postprocess_min_length": 4,
    },
}

def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments for CDS annotation.
    All defaults are provided via organism-specific presets.
    """

    # ---- Pass 1: require organism to select preset ----
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--organism", type=str, choices=list(PRESET_DEFAULTS.keys()), required=True, help="Select which preset configuration to use.")
    pre_args, _ = pre_parser.parse_known_args()
    d = PRESET_DEFAULTS[pre_args.organism]

    # ---- Pass 2: full parser ----
    parser = argparse.ArgumentParser(description="Downstream Task: Coding DNA Sequence (CDS) Annotation.", parents=[pre_parser])

    # ===== Inputs / outputs =====
    parser.add_argument("--input", type=str, nargs="+", default=d["input"], help="Input FASTA/Parquet files or directories")
    parser.add_argument("--output_path", type=str, default=d["output_path"], help="Output directory")

    # ===== Model / runtime =====
    parser.add_argument("--model_name", type=str, default=d["model_name"], help="HuggingFace model path or name")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for inference")
    parser.add_argument("--gpu_count", type=int, default=-1, help="Number of GPUs to use (-1 for all)")
    parser.add_argument("--cpu_count", type=int, default=max(1, int((os.cpu_count() or 1) * 0.8)), help="Number of CPUs to use")
    parser.add_argument("--bf16", action="store_true", help="Use bfloat16 for faster inference")

    # ===== Sequence chunking =====
    parser.add_argument("--context_length", type=int, default=d["context_length"], help="Context length in tokens")
    parser.add_argument("--overlap_length", type=int, default=d["overlap_length"], help="Overlap length in tokens")

    # ===== Postprocess =====
    parser.add_argument("--postprocess_stair_outward_shift", type=int, default=d["postprocess_stair_outward_shift"], help="Max outward bp shift")
    parser.add_argument("--postprocess_stair_inward_shift", type=int, default=d["postprocess_stair_inward_shift"], help="Max inward bp shift")
    parser.add_argument("--postprocess_min_length", type=int, default=d["postprocess_min_length"], help="Minimum run length after refinement")

    # ===== Debug =====
    parser.add_argument("--no_postprocess", action="store_true", help="Disable postprocess (debug)")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of sequences (debug)")

    return parser.parse_args()


def calc(pred_classes, valid_labels):
    # Manually compute binary classification metrics
    tp = np.sum((pred_classes != 0) & (valid_labels != 0)).item()
    fp = np.sum((pred_classes != 0) & (valid_labels == 0)).item()
    fn = np.sum((pred_classes == 0) & (valid_labels != 0)).item()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return precision, recall, f1


def calc_acc(pos_pred, neg_pred, pos_true, neg_true):
    assert(pos_pred.shape == neg_pred.shape == pos_true.shape == neg_true.shape and pos_pred.ndim == 1)
    if pos_pred.shape[0] == 0:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "start_precision": 0.0,
            "start_recall": 0.0,
            "start_f1": 0.0,
            "end_precision": 0.0,
            "end_recall": 0.0,
            "end_f1": 0.0,
            "boundary_precision": 0.0,
            "boundary_recall": 0.0,
            "boundary_f1": 0.0,
            "exact_match": 0.0
        }
    
    pred = np.stack([pos_pred, neg_pred], axis=0) != 0
    true = np.stack([pos_true, neg_true], axis=0) != 0

    precision, recall, f1 = calc(pred.ravel(), true.ravel())
    
    pred = np.column_stack((np.zeros((2, 1), dtype=bool), pred, np.zeros((2, 1), dtype=bool)))
    true = np.column_stack((np.zeros((2, 1), dtype=bool), true, np.zeros((2, 1), dtype=bool)))

    dp = np.diff(pred.astype(np.int8).ravel())
    dt = np.diff(true.astype(np.int8).ravel())
    sp = np.cumsum(pred.astype(int), axis=1).ravel()
    st = np.cumsum(true.astype(int), axis=1).ravel()
    
    start_precision, start_recall, start_f1 = calc(dp == 1, dt == 1)
    end_precision, end_recall, end_f1 = calc(dp == -1, dt == -1)

    true_start_indices = np.nonzero(dt == 1)[0]
    true_end_indices = np.nonzero(dt == -1)[0]
    true_rng = np.stack((true_start_indices, true_end_indices), axis=1)
    pred_start_indices = np.nonzero(dp == 1)[0]
    pred_end_indices = np.nonzero(dp == -1)[0]
    pred_rng = np.stack((pred_start_indices, pred_end_indices), axis=1)

    boundary_precision = (dt[pred_rng] == np.array([1, -1])).all(axis=1).mean().item() if len(pred_rng) > 0 else 0.0
    boundary_recall = (dp[true_rng] == np.array([1, -1])).all(axis=1).mean().item() if len(true_rng) > 0 else 0.0
    boundary_f1 = (
        2 * boundary_precision * boundary_recall / (boundary_precision + boundary_recall)
        if (boundary_precision + boundary_recall) > 0 else 0.0
    )

    exact_match = (
        (dp[true_rng] == np.array([1, -1])).all(axis=1) &
        (np.diff(sp[true_rng + 1], axis=1) == np.diff(st[true_rng + 1], axis=1)).flatten()
    ).mean().item() if len(true_rng) > 0 else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "start_precision": start_precision,
        "start_recall": start_recall,
        "start_f1": start_f1,
        "end_precision": end_precision,
        "end_recall": end_recall,
        "end_f1": end_f1,
        "boundary_precision": boundary_precision,
        "boundary_recall": boundary_recall,
        "boundary_f1": boundary_f1,
        "exact_match": exact_match
    }


@njit(cache=True)
def extract_intervals_from_binary(labels: np.ndarray) -> np.ndarray:
    if labels.ndim != 1 or labels.shape[0] == 0:
        return np.empty((0, 2), dtype=np.int32)

    labels_i8 = labels.astype(np.int8)
    padded = np.empty(labels_i8.shape[0] + 2, dtype=np.int8)
    padded[0] = 0
    padded[-1] = 0
    padded[1:-1] = labels_i8
    diff = np.diff(padded)
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0] - 1
    if starts.shape[0] == 0:
        return np.empty((0, 2), dtype=np.int32)
    return np.stack((starts.astype(np.int32), ends.astype(np.int32)), axis=1)


@njit(cache=True)
def cleanup_short_binary_runs(values: np.ndarray, min_length: int) -> np.ndarray:
    """
    1) Fill internal 0-runs shorter than min_length (edge 0-runs kept).
    2) Remove 1-runs shorter than min_length.
    """
    if values.ndim != 1:
        raise ValueError("values must be 1D")
    n = values.shape[0]
    if n == 0:
        return np.empty((0,), dtype=np.int64)

    m = min_length
    out = (values.astype(np.int8) != 0).astype(np.int8)
    if m <= 1:
        return out.astype(np.int64)

    i = 0
    while i < n:
        if out[i] != 0:
            i += 1
            continue
        j = i + 1
        while j < n and out[j] == 0:
            j += 1
        if i > 0 and j < n and (j - i) < m:
            out[i:j] = 1
        i = j

    i = 0
    while i < n:
        if out[i] != 1:
            i += 1
            continue
        j = i + 1
        while j < n and out[j] == 1:
            j += 1
        if (j - i) < m:
            out[i:j] = 0
        i = j

    return out.astype(np.int64)


@njit(cache=True)
def find_largest_downstep_top(
    values: np.ndarray,
    left: int,
    right: int,
    dir_is_right: bool = True,
) -> Tuple[int, float]:
    """
    Find max adjacent down-step in [left, right].
    - dir_is_right=True (right): x[i]-x[i+1]
    - dir_is_right=False (left): x[i]-x[i-1]
    """
    n = values.shape[0]
    if n == 0:
        return 0, 0.0

    l = max(0, left)
    r = min(n - 1, right)
    if l > r:
        return max(0, min(n - 1, l)), 0.0

    if l == r:
        return l, 0.0

    best_top = l if dir_is_right else r
    best_drop_abs = 0.0

    if dir_is_right:
        for top in range(l, r):
            drop_abs = float(values[top] - values[top + 1])
            if drop_abs > best_drop_abs:
                best_top = top
                best_drop_abs = drop_abs
    else:
        for top in range(l + 1, r + 1):
            drop_abs = float(values[top] - values[top - 1])
            if drop_abs > best_drop_abs:
                best_top = top
                best_drop_abs = drop_abs

    return best_top, best_drop_abs


@njit(cache=True)
def postprocess_argmax_stair_refine(
    class1_confidence: np.ndarray,
    argmax_preds: np.ndarray,
    max_shift: int = 64,
    inner_shift: int = 16,
) -> np.ndarray:
    if class1_confidence.ndim != 1:
        raise ValueError("class1_confidence must be 1D")
    if argmax_preds.ndim != 1 or argmax_preds.shape[0] != class1_confidence.shape[0]:
        raise ValueError("argmax_preds must be 1D and aligned with class1_confidence")
    if class1_confidence.shape[0] == 0:
        return np.empty((0,), dtype=np.int64)

    n = class1_confidence.shape[0]
    shift = max_shift
    in_shift = inner_shift

    pred_is_cds = argmax_preds.astype(np.int8) != 0
    intervals = extract_intervals_from_binary(pred_is_cds.astype(np.int8))
    if intervals.shape[0] == 0:
        return pred_is_cds.astype(np.int64)

    out = np.zeros(n, dtype=np.int64)
    for s, e in intervals:
        start = int(s)
        end = int(e)
        new_start = start
        new_end = end

        left_out_l = max(0, start - shift)
        left_out_r = start
        cand_start_out, drop_start_out = find_largest_downstep_top(
            class1_confidence, left_out_l, left_out_r, False
        )
        left_in_l = start
        left_in_r = min(end, start + in_shift)
        cand_start_in, drop_start_in = find_largest_downstep_top(
            class1_confidence, left_in_l, left_in_r, False
        )

        best_start = start
        best_start_drop = 0.0
        for cand_idx, drop_abs in (
            (cand_start_out, drop_start_out),
            (cand_start_in, drop_start_in),
        ):
            if drop_abs > best_start_drop:
                best_start = int(cand_idx)
                best_start_drop = float(drop_abs)
        new_start = best_start

        right_out_l = end
        right_out_r = min(n - 1, end + shift)
        cand_end_out, drop_end_out = find_largest_downstep_top(
            class1_confidence, right_out_l, right_out_r, True
        )
        right_in_l = max(start, end - in_shift)
        right_in_r = end
        cand_end_in, drop_end_in = find_largest_downstep_top(
            class1_confidence, right_in_l, right_in_r, True
        )

        best_end = end
        best_end_drop = 0.0
        for cand_idx, drop_abs in (
            (cand_end_out, drop_end_out),
            (cand_end_in, drop_end_in),
        ):
            if drop_abs > best_end_drop:
                best_end = int(cand_idx)
                best_end_drop = float(drop_abs)
        new_end = best_end

        if new_end < new_start:
            continue
        out[new_start : new_end + 1] = 1
    return out


def postprocess_sequence_predictions(
    seq_idx: int,
    argmax_preds_per_head: List[np.ndarray],
    class1_conf_per_head: List[Optional[np.ndarray]],
    postprocess_stair_outward_shift: int,
    postprocess_stair_inward_shift: int,
    postprocess_min_length: int,
) -> Tuple[int, List[np.ndarray]]:
    refined_preds_per_head: List[np.ndarray] = []
    for argmax_preds_np, class1_conf in zip(argmax_preds_per_head, class1_conf_per_head):
        if class1_conf is None:
            final_seq_preds_np = argmax_preds_np
        else:
            final_seq_preds_np = postprocess_argmax_stair_refine(
                class1_confidence=class1_conf,
                argmax_preds=argmax_preds_np,
                max_shift=postprocess_stair_outward_shift,
                inner_shift=postprocess_stair_inward_shift,
            )
            if postprocess_min_length > 1:
                final_seq_preds_np = cleanup_short_binary_runs(
                    final_seq_preds_np,
                    min_length=postprocess_min_length,
                )
        refined_preds_per_head.append(final_seq_preds_np)
    return seq_idx, refined_preds_per_head


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
        model_name, dtype=getattr(torch, dtype_str), trust_remote_code=True
    )

    device = torch.device(f"cuda:{gpu_id}" if gpu_id >= 0 else "cpu")
    model.to(device)
    model.eval()
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    print(f"📊 Device {device} model size: {total_params / 1e6:.1f}M parameters")
    print(f"⏱️ Model loading completed in {time.time() - start_time:.2f} seconds")

    return model, tokenizer, device


def parse_fasta_from_stream(stream: IO[str]) -> List[Tuple[str, str]]:
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


def read_fasta(path: str) -> Tuple[List[Tuple[str, str]], None]:
    """
    Read sequences from a FASTA file, supporting local paths and hf:// URLs.

    Args:
        path: Path to the FASTA file (e.g., "local/file.fasta" or
              "hf://datasets/username/my_dataset/my_file.fasta").

    Returns:
        A tuple of (records list, None).
    """
    records: List[Tuple[str, str]]

    if path.startswith("hf://"):
        try:
            import fsspec

            with fsspec.open(path, mode="rt", encoding="utf-8") as f:
                records = parse_fasta_from_stream(f)
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
                records = parse_fasta_from_stream(f)
            print(f"✅ Read {len(records)} sequences from local file: {path}")
        except FileNotFoundError:
            print(f"❌ Error: Local file not found at {path}")
            raise
        except Exception as e:
            print(f"❌ Error reading from local file {path}: {e}")
            raise

    return records, None


def sniff_fasta(path: str, lines_to_check: int = 50) -> bool:
    """
    Heuristically detect whether a path looks like a FASTA file by checking
    the first non-empty line for a '>' header.
    """
    if path.startswith("hf://"):
        try:
            import fsspec

            with fsspec.open(path, mode="rt", encoding="utf-8") as f:
                for _ in range(lines_to_check):
                    line = f.readline()
                    if not line:
                        break
                    line = line.strip()
                    if line:
                        return line.startswith(">")
        except ImportError:
            return False
    else:
        with open(path, "rt", encoding="utf-8", errors="ignore") as f:
            for _ in range(lines_to_check):
                line = f.readline()
                if not line:
                    break
                line = line.strip()
                if line:
                    return line.startswith(">")
    return False


def detect_input_format(path: str) -> str:
    """
    Detect supported input formats.

    Supported:
      - FASTA: by extension or content sniffing
      - Parquet: by extension (.parquet/.parq)
    """
    lower_path = path.lower()
    if lower_path.endswith((".parquet", ".parq")):
        return "parquet"
    if lower_path.endswith((".fasta", ".fa", ".fna", ".ffn", ".faa", ".fas")):
        return "fasta"
    if sniff_fasta(path):
        return "fasta"
    raise ValueError(
        f"Unsupported input format for '{path}'. Please provide a FASTA file or a Parquet file."
    )


def resolve_parquet_input_paths(input_items: List[str]) -> List[str]:
    """
    Resolve input sources in user-given order.
    - File paths are kept as-is.
    - Directory paths expand to supported files recursively, sorted per directory.
    """
    supported_suffixes = (
        ".parquet",
        ".parq",
        ".fasta",
        ".fa",
        ".fna",
        ".ffn",
        ".faa",
        ".fas",
    )
    resolved: List[str] = []
    for item in input_items:
        if item.startswith("hf://"):
            resolved.append(item)
            continue

        if os.path.isdir(item):
            files_in_dir: List[str] = []
            for root, _, files in os.walk(item):
                for name in files:
                    if name.lower().endswith(supported_suffixes):
                        files_in_dir.append(os.path.join(root, name))
            for path in sorted(files_in_dir):
                resolved.append(path)
            continue

        resolved.append(item)

    return resolved


def read_sequences_from_parquet(
    path: str, limit: Union[int, None] = None
) -> Tuple[List[Tuple[str, str]], Union[List[str], None]]:
    """
    Read sequences from a Parquet file.

    Expected columns:
      - sequence column: one of ['sequence', 'seq', 'dna', 'text'] (required)
      - header column: one of ['header', 'fasta_header', 'record_name', 'name', 'id'] (optional)
      - label column: 'label_cds' (optional, for accuracy calculation)
    """
    if path.startswith("hf://"):
        try:
            import fsspec

            with fsspec.open(path, mode="rb") as f:
                df = pd.read_parquet(f)
        except ImportError as e:
            raise ImportError(
                "⚠️ The library 'fsspec' is required for reading hf:// paths. "
                "Please install both 'fsspec' and 'huggingface_hub': pip install fsspec huggingface_hub"
            ) from e
    else:
        df = pd.read_parquet(path)

    if limit is not None:
        df = df.head(limit)

    sequence_col = next(
        (col for col in ("sequence", "seq", "dna", "text") if col in df.columns), None
    )
    if sequence_col is None:
        raise ValueError(
            f"Parquet input '{path}' must contain a sequence column "
            f"(one of: sequence/seq/dna/text). Found columns: {list(df.columns)}"
        )

    # Prefer domain-specific identifiers as FASTA headers if present.
    # Priority: record_id > species_name > other common header fields.
    header_col = next(
        (
            col
            for col in (
                "record_id",
                "species_name",
                "header",
                "fasta_header",
                "record_name",
                "name",
                "id",
            )
            if col in df.columns
        ),
        None,
    )

    sequences = df[sequence_col].tolist()
    headers = df[header_col].tolist() if header_col is not None else [None] * len(sequences)
    
    # Check for ground truth labels
    labels = None
    if "label_cds" in df.columns:
        labels = df["label_cds"].tolist()
        print("✅ Found ground truth labels in 'label_cds' column")
    elif "label_plus" in df.columns and "label_minus" in df.columns:
        labels = [np.concatenate([x, y]) for x, y in zip(df["label_plus"].tolist(), df["label_minus"].tolist())]
        print("✅ Found ground truth labels in 'label_plus' and 'label_minus' columns")
    elif "label+" in df.columns and "label-" in df.columns:
        labels = [np.concatenate([x, y]) for x, y in zip(df["label+"].tolist(), df["label-"].tolist())]
        print("✅ Found ground truth labels in 'label+' and 'label-' columns")

    records: List[Tuple[str, str]] = []
    for i, (seq, header_val) in enumerate(zip(sequences, headers)):
        if not isinstance(seq, str):
            raise ValueError(
                f"Invalid sequence value at row {i} in '{path}': expected string, got {type(seq)}"
            )
        seq = seq.strip().upper()
        if not seq:
            raise ValueError(f"Empty sequence at row {i} in '{path}'")

        # Build FASTA header from selected column. Accept non-string IDs (e.g., int) as well.
        if header_val is not None and str(header_val).strip():
            header = str(header_val).strip()
            if not header.startswith(">"):
                header = ">" + header
        else:
            header = f">record_{i}"

        records.append((header, seq))

    print(f"✅ Read {len(records)} sequences from Parquet file: {path}")
    return records, labels


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
    all_head_records: List[List[Tuple[str, str]]],
    head_names: List[str],
    path: str,
    sequences: Union[List[str], List[Tuple[str, str]], None] = None,
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

    # Normalize sequences input (optional): accept list[str] or list[(header, seq)].
    seq_list: Union[List[str], None] = None
    if sequences is not None:
        if len(sequences) != num_records:
            raise ValueError(
                f"sequences length mismatch: expected {num_records}, got {len(sequences)}"
            )
        if sequences and isinstance(sequences[0], tuple):
            seq_list = [s for _, s in sequences]  # type: ignore[misc]
        else:
            seq_list = sequences  # type: ignore[assignment]

    # Iterate through each sequence record
    for i in range(num_records):
        # Extract record name from the first head's record (it's the same for all)
        header, _ = all_head_records[0][i]
        record_name = (
            header[1:].split()[0] if header.startswith(">") else header.split()[0]
        )
        row_data = {"record_name": record_name}

        # Optionally include original sequence in parquet output
        if seq_list is not None:
            row_data["sequence"] = seq_list[i]

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
    total = len(records)
    base = total // gpu_count
    remainder = total % gpu_count

    gpu_sequences: List[List[Tuple[str, str]]] = []
    cursor = 0
    for gpu_id in range(gpu_count):
        size = base + (1 if gpu_id < remainder else 0)
        chunk = records[cursor : cursor + size]
        cursor += size
        gpu_sequences.append(chunk)
        print(f"📋 GPU {gpu_id} assigned {len(chunk)} sequences")

    return gpu_sequences

@torch.no_grad()
def process_sequences_on_gpu(
    sequences: List[Tuple[str, str]],
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    device: torch.device,
    max_length: int,
    overlap_length: int,
    micro_batch_size: int,
    progress_event_queue: mp.Queue,
    postprocess_workers: int = 1,
    enable_postprocess: bool = True,
    postprocess_stair_outward_shift: int = 64,
    postprocess_stair_inward_shift: int = 16,
    postprocess_min_length: int = 4,
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

    Returns:
        A list of lists of (header, annotation) tuples for the sequences
    """
    pad_id = tokenizer.pad_token_id
    id2label = model.config.id2label
    num_heads = getattr(model, "num_prediction_heads", 1)
    tokenizer_k = getattr(tokenizer, "k", 1) or 1

    max_char_length = max_length * tokenizer_k
    overlap_char_length = overlap_length * tokenizer_k
    num_sequences = len(sequences)
    annotations_per_head = [[""] * num_sequences for _ in range(num_heads)]
    postprocess_executor: Optional[ProcessPoolExecutor] = None
    pending_post_futures: set[object] = set()
    if enable_postprocess:
        postprocess_executor = ProcessPoolExecutor(max_workers=postprocess_workers)

    def update_infer_progress() -> None:
        progress_event_queue.put("infer")

    def notify_postprocess_done(fut: object) -> None:
        if fut.cancelled():
            return
        if fut.exception() is None:
            progress_event_queue.put("post")

    def assign_sequence_annotations(seq_idx: int, preds_per_head: List[np.ndarray]) -> None:
        header, seq = sequences[seq_idx]
        for h in range(num_heads):
            labels = [id2label[i] for i in preds_per_head[h].tolist()]
            annot = "".join(LABEL2CHAR[l] for l in labels)
            assert len(annot) == len(seq)
            annotations_per_head[h][seq_idx] = annot

    def collect_postprocess_result(block: bool = False) -> bool:
        if not pending_post_futures:
            return False
        future_list = list(pending_post_futures)
        if block:
            done, _ = wait(future_list, return_when=FIRST_COMPLETED)
        else:
            done, _ = wait(future_list, timeout=0.0, return_when=FIRST_COMPLETED)
        if not done:
            return False
        for fut in done:
            pending_post_futures.discard(fut)
            seq_idx, final_preds_per_head = fut.result()
            assign_sequence_annotations(seq_idx, final_preds_per_head)
        return True

    def pump_postprocess_nonblocking() -> None:
        while collect_postprocess_result(block=False):
            pass

    try:
        for seq_idx, (_, seq) in enumerate(sequences):
            seq_chunks: List[List[int]] = []
            seq_masks: List[List[int]] = []
            seq_chunk_pos: List[Tuple[int, int, int]] = []
            chr_len = 0

            def add_chunk(chunk_seq: List[int]) -> None:
                pad_len = max_length - len(chunk_seq)
                seq_chunks.append(chunk_seq + [pad_id] * pad_len)
                seq_masks.append([1] * len(chunk_seq) + [0] * pad_len)

            seq_work = seq
            if len(seq_work) < max_char_length:
                chrs = seq_work[:len(seq_work) // tokenizer_k * tokenizer_k]
                chunk_seq = tokenizer(chrs, add_special_tokens=False)["input_ids"]
                seq_chunk_pos.append((chr_len, 0, len(chrs)))
                chr_len += len(chrs)
                add_chunk(chunk_seq)
                if len(seq_work) % tokenizer_k != 0:
                    chrs = seq_work[len(seq_work) % tokenizer_k:]
                    chunk_seq = tokenizer(chrs, add_special_tokens=False)["input_ids"]
                    seq_chunk_pos.append((chr_len, len(seq_work) % tokenizer_k, len(chrs)))
                    chr_len += len(chrs)
                    add_chunk(chunk_seq)
            else:
                while True:
                    chrs = seq_work[:max_char_length]
                    chunk_seq = tokenizer(chrs, add_special_tokens=False)["input_ids"]
                    seq_chunk_pos.append((chr_len, len(seq) - len(seq_work), len(chrs)))
                    assert len(chrs) == max_char_length
                    chr_len += len(chrs)
                    add_chunk(chunk_seq)
                    if len(chrs) == len(seq_work):
                        break
                    if len(seq_work) - max_char_length + overlap_char_length < max_char_length:
                        seq_work = seq_work[-max_char_length:]
                    else:
                        seq_work = seq_work[max_char_length - overlap_char_length:]

            probs_per_head_chunks: List[List[torch.Tensor]] = [[] for _ in range(num_heads)]
            total_chunks = len(seq_chunks)
            for start in range(0, total_chunks, micro_batch_size):
                end = min(start + micro_batch_size, total_chunks)
                inp = torch.tensor(seq_chunks[start:end], dtype=torch.long).to(device)
                att = torch.tensor(seq_masks[start:end], dtype=torch.long).to(device)
                logits = model(input_ids=inp, attention_mask=att).logits
                probs = logits.softmax(dim=-1).cpu()

                for i in range(probs.shape[0]):
                    valid_len = int(att[i].sum().item()) * tokenizer_k
                    total_pred_len = valid_len * num_heads
                    chunk_probs_all_heads = probs[i, :total_pred_len]
                    chunk_probs_all_heads = chunk_probs_all_heads.view(num_heads, valid_len, -1)
                    for h in range(num_heads):
                        probs_per_head_chunks[h].append(chunk_probs_all_heads[h])
                if enable_postprocess:
                    pump_postprocess_nonblocking()

            argmax_preds_per_head: List[np.ndarray] = []
            class1_conf_per_head: List[Optional[np.ndarray]] = []
            seq_len = len(seq)
            for h in range(num_heads):
                seq_probs = torch.cat(probs_per_head_chunks[h], dim=0).float()
                final_seq_probs = torch.zeros(seq_len, seq_probs.shape[1], dtype=torch.float32)
                overlap_cnt = torch.zeros(seq_len, dtype=torch.long)
                for orig_start_pos, new_start_pos, char_length in seq_chunk_pos:
                    final_seq_probs[new_start_pos:new_start_pos + char_length] += seq_probs[
                        orig_start_pos:orig_start_pos + char_length
                    ]
                    overlap_cnt[new_start_pos:new_start_pos + char_length] += 1
                final_seq_probs /= overlap_cnt.unsqueeze(-1)

                argmax_preds_np = final_seq_probs.argmax(dim=-1).cpu().numpy().astype(np.int64, copy=False)
                argmax_preds_per_head.append(argmax_preds_np)
                if enable_postprocess and final_seq_probs.ndim == 2 and final_seq_probs.shape[1] > 1:
                    class1_conf_per_head.append(final_seq_probs[:, 1].cpu().numpy())
                else:
                    class1_conf_per_head.append(None)

            update_infer_progress()
            if enable_postprocess:
                assert postprocess_executor is not None
                fut = postprocess_executor.submit(
                    postprocess_sequence_predictions,
                    seq_idx,
                    argmax_preds_per_head,
                    class1_conf_per_head,
                    postprocess_stair_outward_shift,
                    postprocess_stair_inward_shift,
                    postprocess_min_length,
                )
                fut.add_done_callback(notify_postprocess_done)
                pending_post_futures.add(fut)
                pump_postprocess_nonblocking()
            else:
                assign_sequence_annotations(seq_idx, argmax_preds_per_head)

            if enable_postprocess:
                while pending_post_futures:
                    collect_postprocess_result(block=True)
    finally:
        if enable_postprocess:
            assert postprocess_executor is not None
            postprocess_executor.shutdown(wait=True, cancel_futures=False)

    gpu_annotated_records: List[List[Tuple[str, str]]] = [[] for _ in range(num_heads)]
    for h in range(num_heads):
        for seq_idx, (header, _) in enumerate(sequences):
            gpu_annotated_records[h].append((header, annotations_per_head[h][seq_idx]))
    return gpu_annotated_records


def persistent_worker_process(
    gpu_id: int,
    model_name: str,
    dtype_str: str,
    max_length: int,
    overlap_length: int,
    micro_batch_size: int,
    enable_postprocess: bool,
    postprocess_stair_outward_shift: int,
    postprocess_stair_inward_shift: int,
    postprocess_min_length: int,
    postprocess_workers: int,
    task_queue: mp.Queue,
    result_queue: mp.Queue,
    progress_event_queue: mp.Queue,
    ) -> None:
    job_id = -1
    try:
        model = None
        tokenizer = None
        device = None

        while True:
            task_item = task_queue.get()
            if task_item is None:
                break

            job_id, sequences = task_item
            if model is None:
                model, tokenizer, device = setup_model_for_gpu(model_name, gpu_id, dtype_str)
            gpu_results = process_sequences_on_gpu(
                sequences,
                model,
                tokenizer,
                device,
                max_length,
                overlap_length,
                micro_batch_size,
                progress_event_queue=progress_event_queue,
                postprocess_workers=postprocess_workers,
                enable_postprocess=enable_postprocess,
                postprocess_stair_outward_shift=postprocess_stair_outward_shift,
                postprocess_stair_inward_shift=postprocess_stair_inward_shift,
                postprocess_min_length=postprocess_min_length,
            )
            result_queue.put(("ok", job_id, gpu_id, gpu_results))
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"❌ Error in persistent GPU {gpu_id} worker: {e}")
        result_queue.put(("error", job_id, gpu_id, str(e)))


def create_persistent_worker_runtime(
    model_name: str,
    dtype_str: str,
    gpu_count: int,
    max_length: int,
    overlap_length: int,
    micro_batch_size: int,
    enable_postprocess: bool,
    postprocess_stair_outward_shift: int,
    postprocess_stair_inward_shift: int,
    postprocess_min_length: int,
    cpu_count: int,
) -> Dict[str, object]:
    cpu_budget = int(cpu_count)
    post_workers_total = 0
    post_workers_per_gpu = [0] * gpu_count
    if enable_postprocess:
        post_workers_total = max(1, cpu_budget - gpu_count - 1)
        base = post_workers_total // gpu_count
        remainder = post_workers_total % gpu_count
        for gpu_id in range(gpu_count):
            post_workers_per_gpu[gpu_id] = base + (1 if gpu_id < remainder else 0)
        for gpu_id in range(gpu_count):
            post_workers_per_gpu[gpu_id] = max(1, post_workers_per_gpu[gpu_id])
        post_workers_total = sum(post_workers_per_gpu)
        print(
            f"🧠 post_workers_total={post_workers_total} "
            f"(budget={cpu_budget}, gpu_processes={gpu_count}, main=1)"
        )

    task_queues = [mp.Queue() for _ in range(gpu_count)]
    result_queue = mp.Queue()
    progress_event_queue = mp.Queue()

    processes: List[mp.Process] = []
    for gpu_id in range(gpu_count):
        p = mp.Process(
            target=persistent_worker_process,
            args=(
                gpu_id,
                model_name,
                dtype_str,
                max_length,
                overlap_length,
                micro_batch_size,
                enable_postprocess,
                postprocess_stair_outward_shift,
                postprocess_stair_inward_shift,
                postprocess_min_length,
                post_workers_per_gpu[gpu_id],
                task_queues[gpu_id],
                result_queue,
                progress_event_queue,
            ),
        )
        p.start()
        processes.append(p)

    return {
        "gpu_count": gpu_count,
        "task_queues": task_queues,
        "result_queue": result_queue,
        "progress_event_queue": progress_event_queue,
        "processes": processes,
        "next_job_id": 0,
    }


def shutdown_persistent_worker_runtime(runtime: Dict[str, object], interrupted: bool = False) -> None:
    task_queues = runtime["task_queues"]
    result_queue = runtime["result_queue"]
    progress_event_queue = runtime["progress_event_queue"]
    processes = runtime["processes"]

    if interrupted:
        for p in processes:
            if p.is_alive() and p.pid is not None:
                os.kill(p.pid, signal.SIGINT)
    else:
        for task_queue in task_queues:
            task_queue.put(None)

    for p in processes:
        p.join()

    for q in task_queues:
        q.close()
        q.join_thread()
    result_queue.close()
    result_queue.join_thread()
    progress_event_queue.close()
    progress_event_queue.join_thread()


def annotate_fasta(
    records: List[Tuple[str, str]],
    model_name: str,
    dtype_str: str,
    gpu_count: int,
    max_length: int,
    overlap_length: int,
    micro_batch_size: int,
    enable_postprocess: bool = True,
    postprocess_stair_outward_shift: int = 64,
    postprocess_stair_inward_shift: int = 16,
    postprocess_min_length: int = 4,
    cpu_count: int = max(1, int((os.cpu_count() or 1) * 0.8)),
    persistent_runtime: Optional[Dict[str, object]] = None,
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
    owns_runtime = False
    runtime = persistent_runtime
    if runtime is None:
        print(f"🚀 Starting persistent GPU workers ({gpu_count} GPUs)")
        runtime = create_persistent_worker_runtime(
            model_name=model_name,
            dtype_str=dtype_str,
            gpu_count=gpu_count,
            max_length=max_length,
            overlap_length=overlap_length,
            micro_batch_size=micro_batch_size,
            enable_postprocess=enable_postprocess,
            postprocess_stair_outward_shift=postprocess_stair_outward_shift,
            postprocess_stair_inward_shift=postprocess_stair_inward_shift,
            postprocess_min_length=postprocess_min_length,
            cpu_count=cpu_count,
        )
        owns_runtime = True
    else:
        print(f"🚀 Reusing persistent GPU workers ({gpu_count} GPUs)")

    if int(runtime["gpu_count"]) != gpu_count:
        raise ValueError(
            f"persistent runtime gpu_count={runtime['gpu_count']} does not match requested gpu_count={gpu_count}"
        )

    interrupted = False
    try:
        print(f"📊 Distributing {len(records)} sequences across {gpu_count} GPUs")
        gpu_sequences = distribute_sequences_to_gpus(records, gpu_count)
        active_gpu_ids = [gpu_id for gpu_id in range(gpu_count) if gpu_sequences[gpu_id]]

        result_queue = runtime["result_queue"]
        progress_event_queue = runtime["progress_event_queue"]
        task_queues = runtime["task_queues"]

        job_id = runtime["next_job_id"]
        runtime["next_job_id"] = job_id + 1
        for gpu_id in active_gpu_ids:
            task_queues[gpu_id].put((job_id, gpu_sequences[gpu_id]))

        total_sequences = len(records)
        results = [None] * gpu_count
        expected_results = len(active_gpu_ids)
        received_results = 0

        def progress_worker() -> None:
            infer_pbar = tqdm(
                total=total_sequences,
                desc="Inference",
                unit="seq",
                position=0,
                leave=True,
                dynamic_ncols=True,
                miniters=1,
                mininterval=0.0,
            )
            post_pbar: Optional[tqdm] = None
            if enable_postprocess:
                post_pbar = tqdm(
                    total=total_sequences,
                    desc="Postprocessing",
                    unit="seq",
                    position=1,
                    leave=True,
                    dynamic_ncols=True,
                    miniters=1,
                    mininterval=0.0,
                )

            while infer_pbar.n < infer_pbar.total or (
                post_pbar is not None and post_pbar.n < post_pbar.total
            ):
                event_type = progress_event_queue.get()
                if event_type == "stop":
                    break
                if event_type == "infer":
                    infer_pbar.update(1)
                elif event_type == "post":
                    post_pbar.update(1)
            
            infer_pbar.close()
            if post_pbar is not None:
                post_pbar.close()

        progress_thread = threading.Thread(target=progress_worker, daemon=True)
        progress_thread.start()
        try:
            while received_results < expected_results:
                state, result_job_id, result_gpu_id, payload = result_queue.get()
                if state == "error":
                    raise RuntimeError(f"Worker error on GPU {result_gpu_id}: {payload}")
                if result_job_id != job_id:
                    continue
                results[result_gpu_id] = payload
                received_results += 1
        except KeyboardInterrupt:
            interrupted = True
            raise
        finally:
            if interrupted:
                progress_event_queue.put("stop")
            progress_thread.join()

        print("🔗 Combining results from all GPUs...")
        num_heads = len(results[0]) if results and results[0] is not None else 0
        all_annotated_records = [[] for _ in range(num_heads)]
        for gpu_id in range(gpu_count):
            if results[gpu_id] is not None:
                for head_idx in range(num_heads):
                    all_annotated_records[head_idx].extend(results[gpu_id][head_idx])

        print(f"✅ Successfully processed {len(records)} sequences across {gpu_count} GPUs")
        return all_annotated_records
    finally:
        if owns_runtime:
            shutdown_persistent_worker_runtime(runtime, interrupted=interrupted)


def display_progress_header() -> None:
    """
    Display a stylized header for the CDS annotation pipeline.
    """
    print("\n" + "=" * 80)
    print("🧬  CODING DNA SEQUENCE (CDS) ANNOTATION PIPELINE  🧬")
    print("=" * 80 + "\n")


def read_input_records(
    input_path: str, limit: Optional[int]
) -> Tuple[List[Tuple[str, str]], Optional[List[object]]]:
    print("🔄 Reading input file...")
    input_format = detect_input_format(input_path)
    if input_format == "parquet":
        records, labels = read_sequences_from_parquet(input_path, limit=limit)
    else:
        records, labels = read_fasta(input_path)
        if limit is not None:
            records = records[:limit]
    return records, labels


def write_outputs_for_input(
    annotated_records_per_head: List[List[Tuple[str, str]]],
    head_names: List[str],
    base_input_name: str,
    run_timestamp: str,
    output_path: str,
    fasta_records: List[Tuple[str, str]],
) -> None:
    for head_idx, head_name in enumerate(head_names):
        annotated_records = annotated_records_per_head[head_idx]
        print(f"\n--- Writing FASTA output for: {head_name} ---")
        output_suffix = f"{run_timestamp}_{head_name}"
        fasta_output_path = os.path.join(
            output_path, f"{base_input_name}_{output_suffix}.fasta"
        )
        write_fasta(annotated_records, fasta_output_path)

    if annotated_records_per_head:
        print("\n--- Writing multi-head Parquet output ---")
        parquet_output_path = os.path.join(
            output_path, f"{base_input_name}_{run_timestamp}.parquet"
        )
        write_parquet(
            annotated_records_per_head,
            head_names,
            parquet_output_path,
            sequences=fasta_records,
        )


def calculate_metrics_for_input(
    annotated_records_per_head: List[List[Tuple[str, str]]],
    fasta_records: List[Tuple[str, str]],
    ground_truth_labels: Optional[List[object]],
) -> Optional[Dict[str, float]]:
    if ground_truth_labels is None:
        return None

    print("\n--- Calculating Accuracy Metrics ---")

    pos_pred_parts = []
    neg_pred_parts = []
    pos_true_parts = []
    neg_true_parts = []
    sep = np.array([0], dtype=np.int8)

    for i in tqdm(range(len(fasta_records)), desc="Calculating metrics"):
        _, pos_pred_str = annotated_records_per_head[0][i]
        _, neg_pred_str = annotated_records_per_head[1][i]

        pos_pred = np.array([CHAR2NUM.get(c, 0) for c in pos_pred_str], dtype=np.int8)
        neg_pred = np.array([CHAR2NUM.get(c, 0) for c in neg_pred_str], dtype=np.int8)

        label = np.asarray(ground_truth_labels[i])
        seq_len = len(pos_pred)
        pos_true = label[:seq_len].astype(np.int8, copy=False)
        neg_true = label[seq_len:].astype(np.int8, copy=False)

        if pos_pred_parts:
            pos_pred_parts.append(sep)
            neg_pred_parts.append(sep)
            pos_true_parts.append(sep)
            neg_true_parts.append(sep)

        pos_pred_parts.append(pos_pred)
        neg_pred_parts.append(neg_pred)
        pos_true_parts.append(pos_true)
        neg_true_parts.append(neg_true)

    pos_pred_all = np.concatenate(pos_pred_parts)
    neg_pred_all = np.concatenate(neg_pred_parts)
    pos_true_all = np.concatenate(pos_true_parts)
    neg_true_all = np.concatenate(neg_true_parts)

    metrics = calc_acc(pos_pred_all, neg_pred_all, pos_true_all, neg_true_all)
    df_metrics = pd.DataFrame([metrics])
    print("\n📊 Accuracy Report:")
    print(df_metrics.iloc[0])
    return metrics


def flush_metrics_csv(metrics_rows: List[Dict[str, object]], metrics_path: str) -> None:
    if not metrics_rows:
        return
    df_metrics_all = pd.DataFrame(metrics_rows).set_index("input_name")
    df_metrics_all.to_csv(metrics_path, index=True)
    print(
        f"📝 Updated aggregated 2D metrics CSV "
        f"({len(metrics_rows)} file(s) with labels): {metrics_path}"
    )


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

    # Determine GPU count
    available_gpus = torch.cuda.device_count()
    if available_gpus <= 0:
        raise RuntimeError("No CUDA devices available.")
    if args.gpu_count == -1:
        gpu_count = available_gpus
    else:
        gpu_count = min(args.gpu_count, available_gpus)
    
    print(f"🎯 Using {gpu_count} GPU(s) out of {available_gpus} available")

    enable_postprocess = not args.no_postprocess
    postprocess_stair_outward_shift = args.postprocess_stair_outward_shift
    postprocess_stair_inward_shift = args.postprocess_stair_inward_shift
    postprocess_min_length = args.postprocess_min_length
    cpu_count = args.cpu_count

    if enable_postprocess:
        print(
            "🎯 Postprocess enabled: "
            f"outward_shift={postprocess_stair_outward_shift}, "
            f"inward_shift={postprocess_stair_inward_shift}, "
            f"min_length={postprocess_min_length}, "
            f"cpu_count={cpu_count}"
        )
    else:
        print("🎯 Postprocess disabled: using argmax only")

    input_paths = resolve_parquet_input_paths(args.input)
    if not input_paths:
        raise ValueError("No valid input files resolved from --input.")
    print(f"📁 Resolved {len(input_paths)} input file(s)")

    # Create output directory if it doesn't exist
    os.makedirs(args.output_path, exist_ok=True)

    run_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    metrics_rows: List[Dict[str, object]] = []
    metrics_path = os.path.join(args.output_path, f"metrics_{run_timestamp}.csv")

    persistent_runtime: Optional[Dict[str, object]] = None
    interrupted = False
    try:
        if gpu_count > 0:
            print(f"🚀 Initializing persistent GPU workers ({gpu_count} GPUs)")
            persistent_runtime = create_persistent_worker_runtime(
                model_name=args.model_name,
                dtype_str=dtype_str,
                gpu_count=gpu_count,
                max_length=args.context_length,
                overlap_length=args.overlap_length,
                micro_batch_size=args.batch_size,
                enable_postprocess=enable_postprocess,
                postprocess_stair_outward_shift=postprocess_stair_outward_shift,
                postprocess_stair_inward_shift=postprocess_stair_inward_shift,
                postprocess_min_length=postprocess_min_length,
                cpu_count=cpu_count,
            )

        for input_idx, input_path in enumerate(input_paths, start=1):
            print(f"\n{'=' * 80}")
            print(f"📂 Processing input {input_idx}/{len(input_paths)}: {input_path}")
            print(f"{'=' * 80}")

            fasta_records, ground_truth_labels = read_input_records(
                input_path, limit=args.limit
            )

            # Use unified annotate function for both single and multi-GPU
            annotated_records_per_head = annotate_fasta(
                fasta_records,
                args.model_name,
                dtype_str,
                gpu_count,
                max_length=args.context_length,
                overlap_length=args.overlap_length,
                micro_batch_size=args.batch_size,
                enable_postprocess=enable_postprocess,
                postprocess_stair_outward_shift=postprocess_stair_outward_shift,
                postprocess_stair_inward_shift=postprocess_stair_inward_shift,
                postprocess_min_length=postprocess_min_length,
                cpu_count=cpu_count,
                persistent_runtime=persistent_runtime,
            )

            head_names = ["positive_strand", "negative_strand"]

            input_filename = os.path.basename(input_path)
            base_input_name = os.path.splitext(input_filename)[0]

            write_outputs_for_input(
                annotated_records_per_head=annotated_records_per_head,
                head_names=head_names,
                base_input_name=base_input_name,
                run_timestamp=run_timestamp,
                output_path=args.output_path,
                fasta_records=fasta_records,
            )

            metrics = calculate_metrics_for_input(
                annotated_records_per_head=annotated_records_per_head,
                fasta_records=fasta_records,
                ground_truth_labels=ground_truth_labels,
            )
            if metrics is not None:
                metrics_rows.append({"input_name": base_input_name, **metrics})
                flush_metrics_csv(metrics_rows, metrics_path)
    except KeyboardInterrupt:
        interrupted = True
        raise
    finally:
        if persistent_runtime is not None:
            print("🛑 Shutting down persistent GPU workers...")
            shutdown_persistent_worker_runtime(persistent_runtime, interrupted=interrupted)

    # Print total execution time
    total_time = time.time() - total_start_time
    minutes, seconds = divmod(total_time, 60)
    print(f"\n⏱️ Total execution time: {int(minutes)}m {seconds:.2f}s")
    print("✨ Completed successfully! ✨\n")


if __name__ == "__main__":
    # Set multiprocessing start method
    mp.set_start_method('spawn', force=True)
    main()
