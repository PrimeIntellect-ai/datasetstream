from pathlib import Path
import torch
from src.datasetstream.dataset import TokenDataset, DatasetConfig, TokenizerConfig, TokenDatasetIterator
from src.datasetstream.tokenizer.detokenizer import HuggingfaceDetokenizer


def test_read_and_detokenize_sequences():
    """Test reading and detokenizing sequences from OpenWebText dataset"""
    
    config = DatasetConfig(
        data_files=[Path("data/fineweb-edu-sample/train_0.bin")],
        token_size_bits=17,
        tokenizer_config=TokenizerConfig(
            document_separator_token=128001,
            vocab_size=128256,
        )
    )
    
    dataset = TokenDataset(config.data_files[0], config.token_size_bits)
    iterator = TokenDatasetIterator(dataset, batch_size=1, seq_len=128, seed=42)
    
    print(f"Dataset size: {dataset.num_tokens} tokens")

    detokenizer = HuggingfaceDetokenizer.from_hf("meta-llama/Meta-Llama-3-8B")

    # Read and print a few sequences
    for i in range(5):
        seq = next(iterator)
        # Convert to torch tensor for detokenizer
        tensor = torch.tensor(seq, dtype=torch.int64)
        text = detokenizer.detokenize(tensor)[0]
        print(f"\nSequence {i}:")
        print(text)
        print("-" * 80)

if __name__ == '__main__':
    test_read_and_detokenize_sequences() 