special_chars = ("?", ".", "!", ":", ";", ",", " ") # more to come


class CharacterTokenizer:
    
    def __init__(self, input_text):
        self.input_text = input_text
        self.char_map = self._char_mapping(self.input_text)


    def _char_mapping(self, text_to_process: str) -> dict[str, int]:
        """"
            Process input text_to_process -> sort each unique letter input text consists of.
            Args:
                text_to_process: str - input text to process
        """
        chars = [c for c in text_to_process]
        char_vocab = list(sorted(set(chars)))

        char_map: dict[str, int] = {}
        for i, c in enumerate(char_vocab):
            char_map[c] = i
        return char_map


    def _encode(self, ch: str): 
        """
            transform character to token indice
            Args:
                ch: str - character, which is contained in char_map
                char_map: dict[str, int] - input text is processed as character map
                        char_map contains every character input text is consisting of.
                        Characters are sorted and indexes are assigned.
            Raises:
                KeyError: if input ch is not present in char_map (which is processed text)
        """
        if ch not in self.char_map:
            raise KeyError(f"char '{ch}' is not defined in char_map!")
        return self.char_map[ch]


    def _decode(self, tk_indice: int):
        """
            transform token indice to character
        """
        return next((k for k, v in self.char_map.items() if v == tk_indice), None)
    

    def encode(self) -> list[int]:
        encoded_list = []
        for c in self.input_text:
            encoded_list.append(self._encode(c))
        return encoded_list