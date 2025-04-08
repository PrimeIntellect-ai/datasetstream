import os
from dataclasses import dataclass
from typing import Optional
import json
from pathlib import Path
import numpy as np
from numpy.random import Generator, PCG64

from src.datasetstream import nibble_utils, math_utils
from src.datasetstream.utils import UINT_DTYPE_MAP, get_np_dtype


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
    data_file_path: Path
    tokenizer_config: TokenizerConfig
    token_size_bits: int

    @classmethod
    def from_json(cls, json_path: Path) -> 'DatasetConfig':
        """Load dataset config from a JSON file"""
        with open(json_path) as f:
            data = json.load(f)

        if 'data_file_path' not in data:
            raise ValueError("Missing required field: data_file_path")

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
            data_file_path=Path(data['data_file_path']),
            tokenizer_config=tokenizer_config,
            token_size_bits=token_size_bits
        )


class TokenDataset:
    """
    Memory mapped access to a token file that encodes tokens bit-by-bit
    (instead of byte-per-token). This uses a read_n_bits approach
    similar to the 'nibble' example but generalized to arbitrary bit
    widths.
    """

    def __init__(self, config):
        """
        config must have at least:
        - config.data_file_path: str
        - config.token_size_bits: int
        - config.tokenizer_config.document_separator_token: int
        """
        if not os.path.isfile(config.data_file_path):
            raise ValueError(f"File not found: {config.data_file_path}")

        self.config = config
        self.n_bits = config.token_size_bits

        # Memory-map the file as raw bytes:
        self.data = np.memmap(config.data_file_path, dtype=np.uint8, mode='r')
        self.file_size_bytes = self.data.shape[0]

        # Number of tokens in the file:
        total_bits = self.file_size_bytes * 8
        self.num_tokens = total_bits // self.n_bits

    def read_sequence(self, start_byte_pos: int, seq_len: int) -> Optional[np.ndarray]:
        """
        Reads seq_len tokens starting at the first nibble that aligns with a byte boundary after start_byte_pos
        or returns None if it doesn't fit.
        """
        if start_byte_pos < 0 or start_byte_pos >= self.file_size_bytes:
            return None
        # floor to nearest nibble start
        num_nibbles = (start_byte_pos * 8) // self.n_bits
        nibble_start_bit = num_nibbles * self.n_bits

        # find the next bit where a nibble starts that aligns with a byte boundary
        no_carry_start_bit = math_utils.next_common_multiple(self.n_bits, 8, nibble_start_bit)

        chunk_start_pos = no_carry_start_bit // 8

        # compute num bytes to read to be (seq_len + 1) * n_bits ceil-ed to next byte
        chunk_size = ((seq_len + 1) * self.n_bits + 7) // 8

        chunk = self.data[chunk_start_pos: chunk_start_pos + chunk_size]

        tokens, _, _ = nibble_utils.read_nibbles(chunk.tobytes(), self.n_bits,
                                                 carry=0, carry_bits=0)

        if len(tokens) < seq_len:
            return None

        # Slice out exactly seq_len tokens:
        tokens = tokens[:seq_len]
        return np.array(tokens, dtype=np.int64)


class DatasetIterator:
    """
    Same as before. The difference is that read_sequence now
    deals with bits internally. The interface is unchanged.
    """

    def __init__(self, dataset: TokenDataset, batch_size: int, seq_len: int, seed: int, shuffle: bool = True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.shuffle = shuffle
        self.rng = Generator(PCG64(seed))

        # Pre-generate positions if shuffle=False
        if not shuffle:
            self.positions = self.rng.integers(0, self.dataset.file_size_bytes, self.batch_size)

    def __next__(self) -> np.ndarray:
        # We'll store the results in int64 for safety:
        current_batch = np.zeros((self.batch_size, self.seq_len), dtype=np.int64)

        # If shuffle, randomize positions each time
        if self.shuffle:
            self.positions = self.rng.integers(0, self.dataset.file_size_bytes, self.batch_size)

        current_batch_size = 0
        while current_batch_size < self.batch_size:
            pos = int(self.positions[current_batch_size])
            # Move pos to next document boundary:
            seq = self.dataset.read_sequence(pos, self.seq_len)
            if seq is not None:
                current_batch[current_batch_size] = seq
                if not self.shuffle:
                    # Advance position by seq_len tokens next time
                    self.positions[current_batch_size] += self.seq_len
                current_batch_size += 1
            else:
                # If read_sequence fails, pick a new random position
                self.positions[current_batch_size] = self.rng.integers(0, self.dataset.num_tokens)

        return current_batch
