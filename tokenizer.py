from abc import ABC, abstractmethod
from enum import Enum, auto
from transformers import AutoTokenizer


special_chars = ("?", ".", "!", ":", ";", ",", " ") # more to come


class Tokenizer(ABC):
    """
        Base tokenizer interface.

        Defines the contract every tokenizer must fulfil. The concrete
        ``encode``/``decode`` logic lives in the inheriting classes
        (e.g. ``CharacterTokenizer``), since how text is split into units and
        mapped to token indices differs per tokenizer.
    """

    @abstractmethod
    def encode(self, text: str | None = None) -> list[int]:
        """Encode text into a list of token indices."""

    @abstractmethod
    def decode(self, indices: list[int]) -> str:
        """Decode a list of token indices back into a string."""

    @abstractmethod
    def token_map(self):
        """"Process input text to character/index/indice map"""



class CharacterTokenizer(Tokenizer):
    """Tokenizer whose unit is a single character."""

    def __init__(self, input_text: str):
        self.input_text = input_text
        self.vocab_map = self._char_mapping(input_text)

    def _char_mapping(self, text_to_process: str) -> dict[str, int]:
        """
            Process input text_to_process -> sort each unique character the
            input text consists of and assign an index to it.
            Args:
                text_to_process: str - input text to process
        """
        char_vocab = sorted(set(text_to_process))
        return {c: i for i, c in enumerate(char_vocab)}

    def _encode(self, ch: str) -> int:
        """
            Transform a single character to its token index.
            Args:
                ch: str - character contained in char_map
            Raises:
                KeyError: if ``ch`` is not present in char_map.
        """
        if ch not in self.vocab_map:
            raise KeyError(f"char '{ch}' is not defined in char_map!")
        return self.vocab_map[ch]

    def _decode(self, tk_indice: int) -> str | None:
        """Transform a single token index back to its character."""
        return next((k for k, v in self.vocab_map.items() if v == tk_indice), None)

    def encode(self, text: str | None = None) -> list[int]:
        """Encode ``text`` (or ``input_text`` if omitted) to token indices."""
        text = self.input_text if text is None else text
        return [self._encode(ch) for ch in text]

    def decode(self, indices: list[int]) -> str:
        """Decode a sequence of token indices back into a string."""
        chars = [self._decode(i) for i in indices]
        return "".join(c for c in chars if c is not None)
    
    def token_map(self) -> list[tuple[str, tuple[int, int]]]:
        return [(v, (i, self.vocab_map[v])) for i, v in enumerate(self.input_text)]
    

class WordTokenizer(Tokenizer):

    def __init__(self, input_text):
        self.input_text = input_text
        self.vocab_map = {w: i for i, w in enumerate(sorted(set(self.input_text.split(" "))))}

    def _encode(self, word):
        if word not in self.vocab_map:
            raise KeyError(f"word {word} is not in vocab map!")
        return self.vocab_map[word]

    def encode(self):
        return [self._encode(w) for w in self.input_text.split(" ")]
        
    def decode():
        raise NotImplementedError("WordTokenizer.decode() not implemented yet.")

    def token_map(self): 
        return [(t, (i, self.vocab_map[t])) for i, t in enumerate(self.input_text.split(" "))]
    

class PretrainedTokenizer(Tokenizer):

    def __init__(self, input_text, pretrained_model: str = 'gpt2'):
        self.input_text = input_text
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
        self.tokens = self.tokenizer.encode(self.input_text)

    def encode(self): raise NotImplementedError()

    def decode(self):
        return {self.tokenizer.decode([tok]): tok for tok in self.tokens}

    def token_map(self):
        return [(t, (self.tokenizer.encode(t), i)) for i, t in enumerate(self.input_text.split(" "))]


class Stat(Enum):
    CH_CNT = auto()
    CH_UNIQ = auto()
    W_CNT = auto()
    W_UNIQ = auto()
    T_CNT = auto()
    T_UNIQ = auto()


class TokenStats:

    @staticmethod
    def get_stats(data_acquiring: Tokenizer) -> dict[Stat, int]:
        words = data_acquiring.input_text.split(" ")
        tokens = TokenStats._tokens(data_acquiring)
        return {
            Stat.CH_CNT: len(data_acquiring.input_text),
            Stat.CH_UNIQ: len(set(data_acquiring.input_text)),
            Stat.W_CNT: len(words),
            Stat.W_UNIQ: len(set(words)),
            Stat.T_CNT: len(tokens),
            Stat.T_UNIQ: len(set(tokens)),
        }

    @staticmethod
    def _tokens(data_acquiring: Tokenizer) -> list:
        """
            Best-effort token list for a tokenizer: prefer a precomputed
            ``tokens`` attribute, otherwise fall back to ``encode()``. Returns
            an empty list when neither yields tokens (e.g. ``encode`` not
            implemented), so token stats degrade to 0 rather than raising.
        """
        tokens = getattr(data_acquiring, "tokens", None)
        if tokens is None:
            try:
                tokens = data_acquiring.encode()
            except (NotImplementedError, TypeError):
                tokens = []
        return list(tokens)