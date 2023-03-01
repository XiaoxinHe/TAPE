import gc
import numpy as np
from tqdm import tqdm as tqdm


def _judge_type(data):
    min_val, max_val = data.min(), data.max()
    _dtype = type(min_val)
    if np.issubdtype(_dtype, np.integer):
        if max_val <= 1 and min_val >= 0:
            _dtype = np._bool
        if max_val <= 255 and min_val >= 0:
            _dtype = np.uint8
        elif max_val <= 65535 and min_val >= 0:
            _dtype = np.uint16
        elif max_val <= 2147483647 and min_val >= -2147483647:
            _dtype = np.int32
    elif np.issubdtype(_dtype, np.float):
        _dtype = np.float16
    return _dtype


def save_memmap(data: np.ndarray, path, dtype=None, node_chunk_size=1000000, log=print):
    # ! Determine the least memory cost type

    dtype = _judge_type(data) if dtype is None else dtype

    # ! Store memory map
    x = np.memmap(path, dtype=dtype, mode='w+',
                  shape=data.shape)

    # for i in tqdm(range(0, data.shape[0], node_chunk_size)):
    for i in range(0, data.shape[0], node_chunk_size):
        j = min(i + node_chunk_size, data.shape[0])
        x[i:j] = data[i:j]
    log(f'Saved {path} as {dtype}...')
    del x
    gc.collect()
    # log('releas x')
    return  # SN(type=dtype, path=path, shape=data.shape)
