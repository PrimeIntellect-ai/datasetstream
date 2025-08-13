import time

import torch

from src.datasetstream.dataset_client import DatasetClientIteratorSync
from src.datasetstream.tokenizer.detokenizer import HuggingfaceDetokenizer

if __name__ == '__main__':
    def test():
        dataset_id = "fineweb_edu_val"
        stream_url = f"http://localhost:8080/api/v1/datasets/{dataset_id}/stream"

        # OpenWebText was encoded with the original GPT-2 tokenizer
        detokenizer = HuggingfaceDetokenizer.from_hf("meta-llama/Meta-Llama-3-8B")
        with DatasetClientIteratorSync(stream_url, seed=42, batch_size=32, seq_len=1024) as iterator:
            print(f"Connected to dataset: {dataset_id}")

            count = 0

            for tokens in iterator:
                count += 1
                for batch in tokens:
                    batch = torch.from_numpy(batch)
                    strings = detokenizer.detokenize(batch)
                    assert len(strings) == 1
                    print(strings[0])
                    print("\033[31m--------------------------------\033[0m")
                time.sleep(0.05)

            print(f"Successfully received {count} items from the dataset")


    test()
