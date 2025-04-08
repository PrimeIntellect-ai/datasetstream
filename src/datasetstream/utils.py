import numpy as np
from typing import Dict, Type

UINT_DTYPE_MAP: Dict[int, Type[np.number]] = {
    1: np.uint8,
    2: np.uint16,
    4: np.uint32,
    8: np.uint64
}


def get_np_dtype(byte_size: int) -> Type[np.number]:
    if byte_size not in UINT_DTYPE_MAP:
        raise ValueError(
            f"Unsupported byte_size: {byte_size}. "
            f"Must be one of: {list(UINT_DTYPE_MAP.keys())}"
        )
    return UINT_DTYPE_MAP[byte_size]
