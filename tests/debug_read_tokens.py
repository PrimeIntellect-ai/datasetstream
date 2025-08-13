import time

import numpy as np

from datasetstream.dataset_client import DatasetClientIteratorSync


def main():
    dataset_id = "fineweb_edu_val"
    stream_url = f"http://localhost:8080/api/v1/datasets/{dataset_id}/stream"

    start_time = time.perf_counter()
    with DatasetClientIteratorSync(stream_url, seed=42, batch_size=32, seq_len=1024, prefetch_size=32) as iterator:
        print(f"Connected to dataset: {dataset_id}")

        count = 0
        total_bytes_received = 0

        item: np.array
        for tokens in iterator:
            count += 1
            total_bytes_received += sum(item.nbytes for batch in tokens for item in batch)
            end_time = time.perf_counter()
            time_elapsed = end_time - start_time
            print(f"Average speed: {total_bytes_received / time_elapsed / (1024 ** 2):.2f} MB/s, "
                  f"batch {count}, batches/second: {count / time_elapsed:.2f}")


if __name__ == '__main__':
    main()
