from abc import abstractmethod
from typing import List, Union

import torch
import tiktoken
from transformers import AutoTokenizer


class AbstractDetokenizer:

    @abstractmethod
    def detokenize(self, tokens: Union[torch.Tensor, torch.nested.Tensor]) -> List[str]:
        """
        Accepts a tensor/nested tensor of tokens and returns a list of strings.
        The tensor is expected to have shape ([batch_size], block_size) where batch_size is optional and
        should have a datatype of torch.int64.
        block_size may differ between batches, if the supplied tensor is a nested tensor.
        :param tokens: the tensor of tokens
        :return: a list of strings where each string corresponds to a batch
        """
        pass


class TiktokenDetokenizer(AbstractDetokenizer):

    def __init__(self, tokenizer: tiktoken.Encoding):
        self.tokenizer = tokenizer

    @staticmethod
    def from_encoding(encoding_name: str) -> 'TiktokenDetokenizer':
        """
        Create a detokenizer from the given encoding name
        :param encoding_name: the name of the encoding
        :return: a detokenizer object
        """
        return TiktokenDetokenizer(tiktoken.get_encoding(encoding_name))

    def detokenize(self, tokens: Union[torch.Tensor, torch.nested.Tensor]) -> List[str]:
        # Ensure tokens is (batch_size, block_size)
        if tokens.ndim == 1:
            tokens = tokens.unsqueeze(0)

        assert tokens.ndim == 2, f"Invalid shape {tokens.shape}"
        assert tokens.dtype == torch.int64, f"Invalid datatype {tokens.dtype}"

        batch_size: int
        if isinstance(tokens, torch.Tensor):
            batch_size = tokens.size(0)
        else:
            assert isinstance(tokens, torch.nested.Tensor), f"Invalid tensor type {type(tokens)}"
            batch_size = tokens._nested_tensor_strides().size(0)

        return [self.tokenizer.decode(tokens[i].tolist()) for i in range(batch_size)]


class HuggingfaceDetokenizer(AbstractDetokenizer):

    def __init__(self, tokenizer: tiktoken.Encoding):
        self.tokenizer = tokenizer

    @staticmethod
    def from_hf(hf_path: str) -> 'HuggingfaceDetokenizer':
        """
        Create a detokenizer from the given encoding name
        :param hf_path: the path to the Huggingface model
        :return: a detokenizer object
        """
        tokenizer = AutoTokenizer.from_pretrained(hf_path, use_fast=True)
        return HuggingfaceDetokenizer(tokenizer)

    def detokenize(self, tokens: Union[torch.Tensor, torch.nested.Tensor]) -> List[str]:
        # Ensure tokens is (batch_size, block_size)
        if tokens.ndim == 1:
            tokens = tokens.unsqueeze(0)

        assert tokens.ndim == 2, f"Invalid shape {tokens.shape}"
        assert tokens.dtype == torch.int64, f"Invalid datatype {tokens.dtype}"

        batch_size: int
        if isinstance(tokens, torch.Tensor):
            batch_size = tokens.size(0)
        else:
            assert isinstance(tokens, torch.nested.Tensor), f"Invalid tensor type {type(tokens)}"
            batch_size = tokens._nested_tensor_strides().size(0)

        return [self.tokenizer.decode(tokens[i].tolist()) for i in range(batch_size)]
