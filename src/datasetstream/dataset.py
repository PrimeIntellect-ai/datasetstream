import math
import os
from dataclasses import dataclass
from typing import Optional, List, Tuple
import json
from pathlib import Path
import numpy as np
from numpy.random import Generator, PCG64

from datasetstream import nibble_utils


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
    Memory mapped access to a token file that encodes tokens bit-by-bit.
    Supports scanning to document boundaries and truncation.
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

        # Number of tokens in the file (floor):
        total_bits = self.file_size_bytes * 8
        self.num_tokens = total_bits // self.n_bits

    # --- helpers ------------------------------------------------------------
    def _lcm_bits_with_byte(self) -> int:
        """Return LCM(self.n_bits, 8) in *bits* (period at which token & byte boundaries coincide)."""
        return self.n_bits * 8 // math.gcd(self.n_bits, 8)

    def _align_token_floor_byte_aligned(self, bit_idx: int) -> int:
        """Floor to the closest <= bit index that is both a token boundary and a byte boundary."""
        period_bits = self._lcm_bits_with_byte()
        return (bit_idx // period_bits) * period_bits

    # --- reading APIs -------------------------------------------------------
    def read_sequence(self,
                      start_byte_pos: int,
                      seq_len: int,
                      seek_document_start: bool = True,
                      stop_at_seq_end: bool = True,
                      scan_block_bytes: int = 4096) -> Optional[np.ndarray]:
        """
        Reads up to seq_len tokens starting at the first token boundary
        at or after start_byte_pos. If seek_document_start, scans forward in blocks until the next
        document separator token, then begins reading immediately after that separator. Truncates
        at the next separator after reading. Returns tokens or None if no start found.
        """
        out, _ = self.read_next_sequence(start_byte_pos, seq_len, seek_document_start, stop_at_seq_end,
                                         scan_block_bytes)
        return out

    def read_next_sequence(self,
                           start_byte_pos: int,
                           seq_len: int,
                           seek_document_start: bool = True,
                           stop_at_document_end: bool = True,
                           scan_block_bytes: int = 4096) -> Tuple[Optional[np.ndarray], int]:
        """
        Streaming-friendly read that also returns the **next cursor** to continue from.

        Returns:
            (tokens or None, next_cursor_byte_pos)
            - next_cursor_byte_pos:
                * If a trailing separator was found while reading, this is the byte immediately
                  after that separator (start of the following document).
                * If no separator was found (EOF or seq_len hit first), this is the byte position
                  where scanning ended; a subsequent call with seek_document_start=True will
                  hop to the next doc start from there.
        """
        # Validate position
        if start_byte_pos < 0 or start_byte_pos >= self.file_size_bytes:
            return None, self.file_size_bytes

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
            block_bytes = (scan_block_bytes // period_bytes) * period_bytes or period_bytes

        result: List[int] = []
        seeking = seek_document_start
        found_start_sep = not seek_document_start  # if we aren't seeking, we are already at start
        file_bits = self.file_size_bytes * 8
        bit_offset_from_floor = 0  # counts bits from floor_aligned_bit
        end_sep_bit_abs: Optional[int] = None

        while byte_pos < self.file_size_bytes and len(result) < seq_len:
            block_end = min(self.file_size_bytes, byte_pos + block_bytes)
            if block_end <= byte_pos:
                break

            chunk = self.data[byte_pos:block_end]
            tokens, _, _ = nibble_utils.read_nibbles(chunk.tobytes(), self.n_bits, carry=0, carry_bits=0)
            if not tokens:
                byte_pos = block_end
                continue

            # On the very first bytes, skip whole tokens to reach the precise start bit
            if tokens_to_skip:
                if len(tokens) <= tokens_to_skip:
                    tokens_to_skip -= len(tokens)
                    bit_offset_from_floor += len(tokens) * self.n_bits
                    byte_pos = block_end
                    continue
                else:
                    tokens = tokens[tokens_to_skip:]
                    bit_offset_from_floor += tokens_to_skip * self.n_bits
                    tokens_to_skip = 0

            if seeking:
                # Find first separator in this token slice
                sep_rel = None
                for i, t in enumerate(tokens):
                    if t == self.document_separator_token:
                        sep_rel = i
                        break

                if sep_rel is None:
                    bit_offset_from_floor += len(tokens) * self.n_bits
                    byte_pos = block_end
                    continue

                # Move past the separator to the first token of the next document
                tokens = tokens[sep_rel + 1:]
                bit_offset_from_floor += (sep_rel + 1) * self.n_bits
                found_start_sep = True
                seeking = False

            # Now we are inside a document: collect until the next separator or seq_len tokens
            sep_rel = None
            for i, t in enumerate(tokens):
                if t == self.document_separator_token and stop_at_document_end:
                    sep_rel = i
                    break

            take_tokens = tokens if sep_rel is None else tokens[:sep_rel]

            if take_tokens:
                take = min(seq_len - len(result), len(take_tokens))
                result.extend(take_tokens[:take])

            # If hit a separator within this chunk, compute the absolute bit of the separator,
            # and set next cursor to right after it.
            if sep_rel is not None:
                # Absolute bit where this block's tokens start:
                block_start_bit_abs = floor_aligned_bit + bit_offset_from_floor
                end_sep_bit_abs = block_start_bit_abs + (sep_rel * self.n_bits)
                # Move reader state as if we consumed up to sep (not really needed further)
                bit_offset_from_floor += (sep_rel + 1) * self.n_bits
                byte_pos = block_end
                break

            # Otherwise, continue to next block
            bit_offset_from_floor += len(tokens) * self.n_bits
            byte_pos = block_end

            if len(result) >= seq_len:
                break

        if seek_document_start and not found_start_sep:
            return None, self.file_size_bytes

        # Compute next cursor:
        if end_sep_bit_abs is not None:
            # byte right after the separator token
            next_cursor = min(file_bits, end_sep_bit_abs + self.n_bits) // 8
        else:
            # No trailing separator during read; continue from where scanning ended
            next_cursor = min(self.file_size_bytes, byte_pos)

        return (np.array(result, dtype=np.int64) if result else None), next_cursor


class TokenDatasetIterator:
    """
    Iterator for a single TokenDataset using on-the-fly scanning.
    Maintains a per-stream byte cursor instead of a prebuilt document index.
    """

    def __init__(self, dataset: TokenDataset, batch_size: int, seq_len: int, seed: int,
                 shuffle: bool = True, seek_document_start: bool = True, stop_at_document_end: bool = True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.shuffle = shuffle
        self.rng = Generator(PCG64(seed))

        self.seek_document_start = seek_document_start
        self.stop_at_document_end = stop_at_document_end

        # Per-slot byte cursors for deterministic streaming
        if not shuffle:
            # Start each slot evenly spaced through the file to reduce correlation.
            # (Even spacing by bytes; exact doc boundaries are discovered on the fly.)
            if self.dataset.file_size_bytes == 0:
                raise ValueError("Empty dataset file.")
            step = max(1, self.dataset.file_size_bytes // self.batch_size)
            self.cursors = np.array([(i * step) % self.dataset.file_size_bytes for i in range(self.batch_size)],
                                    dtype=np.uint64)

    def __next__(self) -> List[np.ndarray]:
        current_batch: List[np.ndarray] = []
        i = 0

        while i < self.batch_size:
            if self.shuffle:
                start_pos = int(self.rng.integers(0, max(1, self.dataset.file_size_bytes)))
            else:
                start_pos = int(self.cursors[i])

            seq, next_cursor = self.dataset.read_next_sequence(
                start_byte_pos=start_pos,
                seq_len=self.seq_len,
                seek_document_start=self.seek_document_start,  # always hop to the next doc boundary
                stop_at_document_end=self.stop_at_document_end,
                scan_block_bytes=4096
            )

            if seq is not None and len(seq) > 0:
                current_batch.append(seq.astype(np.int64, copy=False))
                if not self.shuffle:
                    # Advance cursor for this slot
                    self.cursors[i] = np.uint64(next_cursor % max(1, self.dataset.file_size_bytes))
                i += 1
            else:
                # If we couldn't find a doc from this position, move the cursor forward
                if self.shuffle:
                    # try another random position
                    continue
                else:
                    # advance by one block period to avoid degenerate loops
                    period_bytes = self.dataset._lcm_bits_with_byte() // 8
                    self.cursors[i] = np.uint64((self.cursors[i] + max(period_bytes, 1)) %
                                                max(1, self.dataset.file_size_bytes))

        return current_batch


class CompoundDataset:
    """
    A dataset that combines multiple TokenDatasets and allows sampling from them.
    """

    def __init__(self, datasets: List[TokenDataset]):
        if not datasets:
            raise ValueError("CompoundDataset requires at least one TokenDataset")
        self.datasets = datasets
        self.num_datasets = len(datasets)


class CompoundDatasetIterator:
    """
    Iterator for a CompoundDataset that randomly samples from the underlying datasets.
    Uses on-the-fly scanning; no precomputed indices.
    """

    def __init__(self, dataset: CompoundDataset, batch_size: int, seq_len: int, seed: int, shuffle: bool = True,
                 seek_document_start: bool = True, stop_at_document_end: bool = True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.shuffle = shuffle
        self.rng = Generator(PCG64(seed))

        # Create per-dataset iterators (each yields 1 sequence per __next__)
        self.iterators = [
            TokenDatasetIterator(ds, 1, seq_len, seed, shuffle, seek_document_start, stop_at_document_end)
            for ds in dataset.datasets
        ]

        # Pre-generate dataset indices if shuffle=False
        if not shuffle:
            self.dataset_indices = np.zeros(batch_size, dtype=np.int32)
            for i in range(batch_size):
                self.dataset_indices[i] = i % dataset.num_datasets

    def __next__(self) -> List[np.ndarray]:
        current_batch: List[np.ndarray] = []

        if self.shuffle:
            dataset_indices = self.rng.integers(0, self.dataset.num_datasets, self.batch_size)
        else:
            dataset_indices = self.dataset_indices

        for i in range(self.batch_size):
            dataset_idx = int(dataset_indices[i])
            # Each per-dataset iterator yields a list with one sequence
            next_seq = next(self.iterators[dataset_idx])[0]
            current_batch.append(next_seq.astype(np.int64, copy=False))

        return current_batch
