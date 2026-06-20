from tokenizer import Tokenizer, CharacterTokenizer


def tokenize(input_text: str, tokenizer: type[Tokenizer]) -> list[int]:
    """
        Encode ``input_text`` into token indices using the given tokenizer.
        Works with any ``Tokenizer`` subclass (e.g. ``CharacterTokenizer``)
        since it relies only on the abstract ``encode`` interface.
        Args:
            input_text: str - text to encode
            tokenizer: type[Tokenizer] - a Tokenizer subclass to instantiate
    """
    return tokenizer(input_text).encode()


def main():
    txt: str = 'The way you do anything is the way you do everything.'
    print(tokenize(txt, CharacterTokenizer))
    

    
if __name__ == "__main__":
    main()
