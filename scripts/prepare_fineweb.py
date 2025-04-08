import os
from tqdm import tqdm
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer

num_proc = 16
num_proc_load_dataset = num_proc

if __name__ == '__main__':
    dataset = load_dataset("HuggingFaceFW/fineweb-edu", num_proc=num_proc_load_dataset, streaming=True)

    split_dataset = dataset["train"].train_test_split(test_size=0.0005, seed=2357, shuffle=True)
    split_dataset['val'] = split_dataset.pop('test')  # rename the test split to val
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B", use_fast=True)


    def process(example):
        ids = tokenizer.encode(example['text'])
        ids.append(tokenizer.eos_token_id)
        out = {'ids': ids, 'len': len(ids)}
        return out


    tokenized = split_dataset.map(
        process,
        remove_columns=['text'],
        desc="tokenizing the splits",
        num_proc=num_proc,
    )

    for split, dset in tokenized.items():
        arr_len = np.sum(dset['len'], dtype=np.uint64)
        filename = os.path.join(os.path.dirname(__file__), f'data/fineweb-edu/{split}.bin')

        os.makedirs(os.path.dirname(filename), exist_ok=True)

        dtype = np.uint32
        arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))
        total_batches = 1024

        idx = 0
        for batch_idx in tqdm(range(total_batches), desc=f'writing {filename}'):
            batch = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format('numpy')
            arr_batch = np.concatenate(batch['ids'])
            arr[idx: idx + len(arr_batch)] = arr_batch
            idx += len(arr_batch)
        arr.flush()