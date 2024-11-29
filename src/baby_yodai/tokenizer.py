"""
Module containing a simple character-level tokenizer for the Baby Yoda model.
"""


class BabyYodAITokenizer:
    """
    A simple character-level tokenizer for the Baby Yoda model.
    """

    def __init__(self, vocab: str):
        """
        Initializes the tokenizer with a given vocabulary.
        """
        self.vocab = vocab
        self.vocab_size = len(vocab)
        self._stoi = {char: i for i, char in enumerate(vocab)}
        self._itos = dict(enumerate(vocab))

    def encode(self, text: str) -> list:
        """
        Encodes the input text into a list of integers.
        """
        return [self._stoi[char] for char in text]

    def decode(self, tokens: list[int]) -> str:
        """
        Decodes the input list of integers into a string.
        """
        return "".join([self._itos[token] for token in tokens])
