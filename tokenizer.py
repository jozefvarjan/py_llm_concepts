from abc import ABC, abstractmethod

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


class CharacterTokenizer(Tokenizer):
    """Tokenizer whose unit is a single character."""

    def __init__(self, input_text: str):
        self.input_text = input_text
        self.char_map = self._char_mapping(input_text)

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
        if ch not in self.char_map:
            raise KeyError(f"char '{ch}' is not defined in char_map!")
        return self.char_map[ch]

    def _decode(self, tk_indice: int) -> str | None:
        """Transform a single token index back to its character."""
        return next((k for k, v in self.char_map.items() if v == tk_indice), None)

    def encode(self, text: str | None = None) -> list[int]:
        """Encode ``text`` (or ``input_text`` if omitted) to token indices."""
        text = self.input_text if text is None else text
        return [self._encode(ch) for ch in text]

    def decode(self, indices: list[int]) -> str:
        """Decode a sequence of token indices back into a string."""
        chars = [self._decode(i) for i in indices]
        return "".join(c for c in chars if c is not None)
