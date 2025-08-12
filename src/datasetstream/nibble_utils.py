import os
from torch.utils.cpp_extension import load
from typing import Tuple, List

_read_nibbles_cpp = load(
    name="read_nibbles_cpp",
    sources=[os.path.join(os.path.dirname(__file__), "read_nibbles.cpp")],
    extra_cflags=["-O3"],
)

def read_nibbles(chunk_bytes: bytes,
                 n_bits: int,
                 carry: int = 0,
                 carry_bits: int = 0
                ) -> Tuple[List[int], int, int]:
    values, carry_out, carry_bits_out = _read_nibbles_cpp.read_nibbles(
        chunk_bytes, n_bits, carry, carry_bits
    )
    return values, carry_out, carry_bits_out