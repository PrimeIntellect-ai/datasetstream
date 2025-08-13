import struct
from typing import List
import numpy as np
from typing import Dict, Type, List

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


MAGIC = b"NPKT"
VERSION = 1


def pack_batches(batches: List[List[np.ndarray]]) -> bytes:
    """
    batches := list of batches; each batch is a list of ndarrays.
    Returns one bytes object you can send via ws.send_bytes(...)
    """
    parts = [MAGIC, struct.pack("<B", VERSION), struct.pack("<I", len(batches))]

    for batch in batches:
        parts.append(struct.pack("<I", len(batch)))
        for arr in batch:
            if not isinstance(arr, np.ndarray):
                raise TypeError("All items must be numpy arrays")
            if arr.dtype == np.object_:
                raise TypeError("object dtype not supported")
            if not arr.flags["C_CONTIGUOUS"]:
                arr = np.ascontiguousarray(arr)

            dtype_bytes = arr.dtype.str.encode("ascii")  # e.g. b'<f4'
            if len(dtype_bytes) > 255:
                raise ValueError("dtype string too long")

            parts.append(struct.pack("<B", len(dtype_bytes)))
            parts.append(dtype_bytes)

            ndim = arr.ndim
            if ndim > 255:
                raise ValueError("too many dimensions")
            parts.append(struct.pack("<B", ndim))
            parts.append(struct.pack("<{}Q".format(ndim), *arr.shape))

            parts.append(struct.pack("<Q", arr.nbytes))
            parts.append(memoryview(arr).cast("B"))

    return b"".join(parts)


def unpack_batches(buf: bytes) -> List[List[np.ndarray]]:
    """
    Reverse of pack_batches. Returns list[list[np.ndarray]].
    Arrays are read-only views into `buf` (zero-copy).
    """
    mv = memoryview(buf)
    off = 0

    if mv[off:off + 4].tobytes() != MAGIC:
        raise ValueError("bad magic")
    off += 4

    version = mv[off]
    off += 1
    if version != VERSION:
        raise ValueError(f"unsupported version {version}")

    (nbatches,) = struct.unpack_from("<I", mv, off)
    off += 4

    out: List[List[np.ndarray]] = []
    for _ in range(nbatches):
        (narrays,) = struct.unpack_from("<I", mv, off)
        off += 4
        batch: List[np.ndarray] = []
        for _ in range(narrays):
            (dtype_len,) = struct.unpack_from("<B", mv, off)
            off += 1
            dtype_str = mv[off:off + dtype_len].tobytes().decode("ascii")
            off += dtype_len

            (ndim,) = struct.unpack_from("<B", mv, off)
            off += 1
            shape = struct.unpack_from("<{}Q".format(ndim), mv, off)
            off += 8 * ndim

            (nbytes,) = struct.unpack_from("<Q", mv, off)
            off += 8
            data = mv[off:off + nbytes]
            off += nbytes

            dt = np.dtype(dtype_str)
            arr = np.frombuffer(data, dtype=dt).reshape(shape)
            arr.flags.writeable = False  # optional: make explicit
            batch.append(arr)
        out.append(batch)

    return out
