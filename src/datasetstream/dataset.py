import math
import os
from dataclasses import dataclass
from typing import Optional, List
import json
from pathlib import Path
import numpy as np
from numpy.random import Generator, PCG64

from datasetstream import nibble_utils, math_utils


@dataclass
class TokenizerConfig:
    """Configuration for the tokenizer and token representation"""
    document_separator_token: int
    vocab_size: int

    def __post_init__(self):
        if self.document_separator_token >= self.vocab_size:
            raise ValueError(
                f"document_separator_token {self.document_separator_token} must be less than vocab_size {self.vocab_size}"
            )


@dataclass
class DatasetConfig:
    """Configuration for the binary token dataset"""
    data_files: List[Path]
    tokenizer_config: TokenizerConfig
    token_size_bits: int

    @classmethod
    def from_json(cls, json_path: Path) -> 'DatasetConfig':
        """Load dataset config from a JSON file"""
        with open(json_path) as f:
            data = json.load(f)

        if 'data_files' not in data:
            raise ValueError("Missing required field: data_files")

        if 'tokenizer_config' not in data:
            raise ValueError("Missing required field: tokenizer_config")

        if 'token_size_bits' not in data:
            raise ValueError("Missing required field: token_size_bits")

        tokenizer_data = data['tokenizer_config']
        required_fields = ['document_separator_token', 'vocab_size']
        missing = [f for f in required_fields if f not in tokenizer_data]
        if missing:
            raise ValueError(f"Missing required fields in tokenizer_config: {', '.join(missing)}")

        vocab_size = tokenizer_data['vocab_size']
        token_size_bits = data['token_size_bits']

        if token_size_bits < 0:
            raise ValueError("token_size_bits must be a positive integer")

        if token_size_bits > 64:
            raise ValueError("token_size_bits must be less than or equal to 64")

        max_value = (1 << token_size_bits) - 1

        if vocab_size > max_value:
            raise ValueError(
                f"vocab_size {vocab_size} is too large for token_size_bits {token_size_bits}. "
                f"Maximum value: {max_value}"
            )

        if vocab_size > 2 ** token_size_bits:
            raise ValueError(
                f"vocab_size {vocab_size} exceeds the maximum representable value for token_size_bits {token_size_bits}"
            )

        tokenizer_config = TokenizerConfig(
            document_separator_token=tokenizer_data['document_separator_token'],
            vocab_size=vocab_size
        )

        return DatasetConfig(
            data_files=[Path(file) for file in data['data_files']],
            tokenizer_config=tokenizer_config,
            token_size_bits=token_size_bits
        )


class TokenDataset:
    """
    Memory mapped access to a token file that encodes tokens bit-by-bit
    (instead of byte-per-token). Supports seeking to document boundaries and truncation.
    """

    def __init__(self, data_file: Path, token_size_bits: int, document_separator_token: int):
        """
        Initialize a TokenDataset with direct parameters.

        Args:
            data_file: Path to the data file
            token_size_bits: Number of bits per token
            document_separator_token: The integer token marking document boundaries
        """
        if not os.path.isfile(data_file):
            raise ValueError(f"File not found: {data_file}")

        self.data_file = data_file
        self.n_bits = token_size_bits
        self.document_separator_token = document_separator_token

        # Memory-map the file as raw bytes:
        self.data = np.memmap(data_file, dtype=np.uint8, mode='r')
        self.file_size_bytes = self.data.shape[0]

        # Number of tokens in the file:
        total_bits = self.file_size_bytes * 8
        self.num_tokens = total_bits // self.n_bits

        self._build_doc_index()

    # --- helpers ------------------------------------------------------------
    def _lcm_bits_with_byte(self) -> int:
        """Return LCM(self.n_bits, 8) in *bits* (period at which token & byte boundaries coincide)."""
        return self.n_bits * 8 // math.gcd(self.n_bits, 8)

    def _align_token_floor_byte_aligned(self, bit_idx: int) -> int:
        """Floor to the closest <= bit index that is both a token boundary and a byte boundary."""
        period_bits = self._lcm_bits_with_byte()
        return (bit_idx // period_bits) * period_bits

    def _build_doc_index(self, scan_block_bytes: int = 4096) -> None:
        """
        Scan the file once and record the byte offset of the first token of each document
        (i.e., the position immediately after each separator token). Offsets are floored to
        bytes; read_sequence will align to the next token boundary when starting from these.
        """
        period_bits = self._lcm_bits_with_byte()
        period_bytes = period_bits // 8
        block_bytes = max(period_bytes, (scan_block_bytes // period_bytes) * period_bytes or period_bytes)

        doc_starts: list[int] = []
        file_bits = self.file_size_bytes * 8
        bit_offset = 0
        byte_pos = 0

        while byte_pos < self.file_size_bytes:
            block_end = min(self.file_size_bytes, byte_pos + block_bytes)
            if block_end <= byte_pos:
                break

            chunk = self.data[byte_pos:block_end]
            tokens, _, _ = nibble_utils.read_nibbles(chunk.tobytes(), self.n_bits, carry=0, carry_bits=0)
            if tokens:
                for i, t in enumerate(tokens):
                    if t == self.document_separator_token:
                        sep_bit = bit_offset + (i + 1) * self.n_bits  # first token AFTER separator
                        if sep_bit < file_bits:  # ignore trailing sep at EOF
                            doc_starts.append(sep_bit // 8)
                bit_offset += len(tokens) * self.n_bits

            byte_pos = block_end

        self.doc_start_bytes = np.array(doc_starts, dtype=np.uint64)
        self.num_docs = int(self.doc_start_bytes.shape[0])

    def read_sequence(self,
                      start_byte_pos: int,
                      seq_len: int,
                      seek_document_start: bool = True,
                      scan_block_bytes: int = 4096) -> Optional[np.ndarray]:
        """
        Reads up to seq_len tokens starting at the first token boundary at or after start_byte_pos.
        If seek_document_start, scans forward in blocks until the next document separator token,
        then begins reading immediately after that separator. Truncates at the next separator after reading.

        IMPORTANT: We align the initial scan to the previous *byte-and-token-aligned* boundary and
        then skip a small number of whole tokens to reach the true start bit. This avoids bit-offset
        desynchronization that would otherwise cause separators to never be detected.

        Args:
            start_byte_pos: byte offset to begin scanning from
            seq_len: maximum number of tokens to return
            seek_document_start: if True, advance to the next separator first
            scan_block_bytes: preferred block size in bytes for scanning (will be rounded down to a
                               multiple of the LCM(n_bits,8)/8 so blocks end on token boundaries)

        Returns:
            A numpy array of tokens (length <= seq_len), or None if no valid start found.
        """
        # Validate position
        if start_byte_pos < 0 or start_byte_pos >= self.file_size_bytes:
            return None

        # The first token boundary at or after start_byte_pos, expressed in bits
        start_bit_token_aligned = ((start_byte_pos * 8 + self.n_bits - 1) // self.n_bits) * self.n_bits

        # Align to a *byte-and-token* boundary at or before start_bit_token_aligned
        period_bits = self._lcm_bits_with_byte()
        period_bytes = period_bits // 8
        floor_aligned_bit = self._align_token_floor_byte_aligned(start_bit_token_aligned)
        byte_pos = floor_aligned_bit // 8

        # Number of whole tokens to skip from the floor-aligned boundary to the exact start bit
        tokens_to_skip = (start_bit_token_aligned - floor_aligned_bit) // self.n_bits

        # Choose a block size that is a multiple of the token/byte period so each chunk ends at a token boundary
        if scan_block_bytes < period_bytes:
            block_bytes = period_bytes
        else:
            block_bytes = (scan_block_bytes // period_bytes) * period_bytes
            if block_bytes == 0:
                block_bytes = period_bytes

        result: list[int] = []
        seeking = seek_document_start
        found_start_sep = not seek_document_start  # if we aren't seeking, we are already at start

        # Stream through the file
        while byte_pos < self.file_size_bytes and len(result) < seq_len:
            block_end = min(self.file_size_bytes, byte_pos + block_bytes)
            if block_end <= byte_pos:
                break

            chunk = self.data[byte_pos:block_end]
            # Because blocks are chosen to be token-boundary aligned, we can decode cleanly each time.
            tokens, _, _ = nibble_utils.read_nibbles(chunk.tobytes(), self.n_bits, carry=0, carry_bits=0)
            if not tokens:
                break

            # On the very first bytes, skip whole tokens to reach the precise start bit
            if tokens_to_skip:
                if len(tokens) <= tokens_to_skip:
                    tokens_to_skip -= len(tokens)
                    byte_pos = block_end
                    continue
                else:
                    tokens = tokens[tokens_to_skip:]
                    tokens_to_skip = 0

            # If we still need to find the *start* separator, look for it and drop everything up to and including it
            if seeking:
                # Find first separator in this token slice
                try:
                    idx = next(i for i, t in enumerate(tokens) if t == self.document_separator_token)
                except StopIteration:
                    # No separator here; advance
                    byte_pos = block_end
                    continue

                # Move past the separator to the first token of the next document
                tokens = tokens[idx + 1:]
                found_start_sep = True
                seeking = False

            # Now we are in-Document: collect until the next separator or seq_len tokens
            # Check for next separator inside this chunk
            sep_idx = None
            for i, t in enumerate(tokens):
                if t == self.document_separator_token:
                    sep_idx = i
                    break

            if sep_idx is not None:
                tokens = tokens[:sep_idx]

            # Append up to seq_len
            if tokens:
                take = min(seq_len - len(result), len(tokens))
                result.extend(tokens[:take])

            # If we hit a separator, we stop regardless of whether we've filled seq_len
            if sep_idx is not None or len(result) >= seq_len:
                break

            byte_pos = block_end

        if seek_document_start and not found_start_sep:
            # We scanned to EOF without finding the beginning of a document
            return None

        return np.array(result, dtype=np.int64)



class TokenDatasetIterator:
    """
    Iterator for a single TokenDataset. The difference is that read_sequence now
    deals with bits internally. The interface is unchanged.
    """

    def __init__(self, dataset: TokenDataset, batch_size: int, seq_len: int, seed: int, shuffle: bool = False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.shuffle = shuffle
        self.rng = Generator(PCG64(seed))

        # Pre-generate document indices if shuffle=False
        if not shuffle:
            if getattr(self.dataset, 'num_docs', 0) == 0:
                raise ValueError("No documents found in dataset (no separators).")
            # Start with deterministic, evenly spaced doc indices
            self.positions = (np.arange(self.batch_size, dtype=np.int64) % self.dataset.num_docs)

    def __next__(self) -> List[np.ndarray]:
        # Collect a full batch
        current_batch: List[np.ndarray] = []

        # Choose which documents to fetch
        if self.shuffle:
            if getattr(self.dataset, 'num_docs', 0) == 0:
                raise ValueError("No documents found in dataset (no separators).")
            doc_indices = self.rng.integers(0, self.dataset.num_docs, self.batch_size)
        else:
            doc_indices = self.positions

        i = 0
        while i < self.batch_size:
            doc_idx = int(doc_indices[i])
            byte_pos = int(self.dataset.doc_start_bytes[doc_idx])

            # Read starting at the *current* document start; do not skip to the next separator
            seq = self.dataset.read_sequence(byte_pos, self.seq_len, seek_document_start=False)
            if seq is not None and len(seq) > 0:
                current_batch.append(seq)
                if not self.shuffle:
                    # Move to the next document deterministically (with wrap-around)
                    self.positions[i] = (self.positions[i] + 1) % self.dataset.num_docs
                i += 1
            else:
                # If the chosen doc is empty (e.g., trailing separator), pick another
                if self.shuffle:
                    doc_indices[i] = self.rng.integers(0, self.dataset.num_docs)
                else:
                    self.positions[i] = (self.positions[i] + 1) % self.dataset.num_docs

        return current_batch


class CompoundDataset:
    """
    A dataset that combines multiple TokenDatasets and allows sampling from them.
    """
    
    def __init__(self, datasets: List[TokenDataset]):
        """
        Initialize a CompoundDataset with a list of TokenDatasets.
        
        Args:
            datasets: List of TokenDataset objects
        """
        if not datasets:
            raise ValueError("CompoundDataset requires at least one TokenDataset")
        
        self.datasets = datasets
        self.num_datasets = len(datasets)


class CompoundDatasetIterator:
    """
    Iterator for a CompoundDataset that randomly samples from the underlying datasets.
    """
    
    def __init__(self, dataset: CompoundDataset, batch_size: int, seq_len: int, seed: int, shuffle: bool = False):
        """
        Initialize a CompoundDatasetIterator.
        
        Args:
            dataset: The CompoundDataset to iterate over
            batch_size: Number of sequences to return in each batch
            seq_len: Length of each sequence
            seed: Random seed for reproducibility
            shuffle: Whether to shuffle the data
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.shuffle = shuffle
        self.rng = Generator(PCG64(seed))
        
        # Create iterators for each dataset
        self.iterators = [
            TokenDatasetIterator(dataset, 1, seq_len, seed, shuffle)
            for dataset in dataset.datasets
        ]
        
        # Pre-generate dataset indices if shuffle=False
        if not shuffle:
            self.dataset_indices = np.zeros(batch_size, dtype=np.int32)
            for i in range(batch_size):
                self.dataset_indices[i] = i % dataset.num_datasets
    
    def __next__(self) -> List[np.ndarray]:
        # We'll store the results in int64 for safety:
        current_batch = []
        
        # If shuffle, randomize dataset indices each time
        if self.shuffle:
            dataset_indices = self.rng.integers(0, self.dataset.num_datasets, self.batch_size)
        else:
            dataset_indices = self.dataset_indices
        
        # Get sequences from each dataset
        for i in range(self.batch_size):
            dataset_idx = dataset_indices[i]
            next_seq = next(self.iterators[dataset_idx])[0]
            current_batch.append(next_seq)

        return current_batch
