from tokenizer import CharacterTokenizer

def main():
    txt: str = 'The way you do anything is the way you do everything.'
    t = CharacterTokenizer(txt)
    dlist = t.encode()
    print(t.char_map)
    print(t.input_text)
    print(dlist)
    

    
if __name__ == "__main__":
    main()
