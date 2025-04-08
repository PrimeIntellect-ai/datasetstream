# Datasetstream

A simple performant dataset streaming server & client for tokenized webtext datasets

## Example
```python
dataset_id = "openwebtext_train"
stream_url = f"http://localhost:8080/api/v1/datasets/{dataset_id}/stream"        

with DatasetClientIteratorSync(stream_url, seed=42, batch_size=32, seq_len=1024) as iterator:
    item: np.array
    for tokens in iterator:
        count += 1
        total_bytes_received += tokens.nbytes
        tokens = torch.from_numpy(tokens.astype(dtype=np.int64))
```