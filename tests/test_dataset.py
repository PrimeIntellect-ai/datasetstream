from pathlib import Path
import numpy as np
import json
import pytest
from src.datasetstream.dataset import TokenDataset, DatasetConfig, TokenizerConfig, TokenDatasetIterator


def test_dataset_basic_functionality():
    """Test basic dataset functionality without any tokenizer dependencies"""
    config = DatasetConfig(
        data_files=[Path("data/fineweb-edu-sample/train_0.bin")],
        token_size_bits=16,
        tokenizer_config=TokenizerConfig(
            document_separator_token=50256,
            vocab_size=50257
        )
    )

    dataset = TokenDataset(config.data_files[0], config.token_size_bits, config.tokenizer_config.document_separator_token)

    # Test basic properties
    assert dataset.num_tokens > 0, "Dataset should not be empty"

    # Test sequence reading
    seq_len = 64
    seq = dataset.read_sequence(0, seq_len)
    assert seq is not None, "Should be able to read from start of file"
    assert len(seq) == seq_len, "Should return requested sequence length"
    assert seq.dtype == np.int64, "Should always be int64"

    # Test invalid reads return None
    assert dataset.read_sequence(-1, seq_len) is None, "Should handle negative position"
    assert dataset.read_sequence(dataset.file_size_bytes, seq_len) is None, "Should handle position past end"
    assert dataset.read_sequence(dataset.file_size_bytes - seq_len + 1, seq_len) is None, "Should handle incomplete sequence"


def test_dataset_iterator():
    """Test the dataset iterator functionality"""
    config = DatasetConfig(
        data_files=[Path("data/fineweb-edu-sample/train_0.bin")],
        token_size_bits=17,
        tokenizer_config=TokenizerConfig(
            document_separator_token=128001,
            vocab_size=128256,
        )
    )

    dataset = TokenDataset(config.data_files[0], config.token_size_bits, config.tokenizer_config.document_separator_token)
    iterator = TokenDatasetIterator(dataset, seq_len=64, seed=42, batch_size=32)

    # Test we can get multiple sequences
    sequences = [next(iterator) for _ in range(5)]
    assert len(sequences) == 5, "Should be able to get multiple sequences"

    # Test all sequences are the right shape and type
    for seq in sequences:
        assert len(seq[0]) == 64, "All sequences should be the requested length"
        assert len(seq) == 32, "All sequences should be the requested batch size"
        assert seq.max() < config.tokenizer_config.vocab_size, "Tokens should be within vocab size"


def test_dataset_config_from_json(tmp_path):
    """Test loading dataset config from JSON"""
    # Test valid config
    valid_config = {
        "data_files": ["data/fineweb-edu-sample/train_0.bin"],
        "token_size_bits": 17,
        "tokenizer_config": {
            "document_separator_token": 128001,
            "vocab_size": 128256
        }
    }
    config_path = tmp_path / "valid_config.json"
    with open(config_path, "w") as f:
        json.dump(valid_config, f)

    config = DatasetConfig.from_json(config_path)
    assert isinstance(config, DatasetConfig)
    assert config.data_files == [Path("data/fineweb-edu-sample/train_0.bin")]
    assert config.token_size_bits == 17
    assert config.tokenizer_config.document_separator_token == 128001
    assert config.tokenizer_config.vocab_size == 128256

    # Test missing data_files
    invalid_config = {
        "token_size_bits": 17,
        "tokenizer_config": {
            "document_separator_token": 128001,
            "vocab_size": 128256
        }
    }
    config_path = tmp_path / "missing_path_config.json"
    with open(config_path, "w") as f:
        json.dump(invalid_config, f)

    with pytest.raises(ValueError, match="Missing required field: data_files"):
        DatasetConfig.from_json(config_path)

    # Test missing tokenizer_config
    invalid_config = {
        "data_files": "data/fineweb-edu-sample/train_0.bin"
    }
    config_path = tmp_path / "missing_tokenizer_config.json"
    with open(config_path, "w") as f:
        json.dump(invalid_config, f)

    with pytest.raises(ValueError, match="Missing required field: tokenizer_config"):
        DatasetConfig.from_json(config_path)

    # Test missing tokenizer fields
    invalid_config = {
        "data_files": ["data/fineweb-edu-sample/train_0.bin"],
        "token_size_bits": 17,
        "tokenizer_config": {
            "document_separator_token": 128001
            # Missing vocab_size
        }
    }
    config_path = tmp_path / "missing_tokenizer_fields_config.json"
    with open(config_path, "w") as f:
        json.dump(invalid_config, f)

    with pytest.raises(ValueError, match="Missing required fields in tokenizer_config: vocab_size"):
        DatasetConfig.from_json(config_path)


if __name__ == '__main__':
    test_dataset_basic_functionality()
    test_dataset_iterator()
    test_dataset_config_from_json(Path("/tmp"))
    print("All tests passed!")
