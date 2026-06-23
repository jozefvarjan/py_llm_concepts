from tokenizer import CharacterTokenizer, WordTokenizer
from visualizer import TokenScatterVisualizer, PlotSettings


def main():
    txt: str = 'The way you do anything is the way you do everything.'
 
    char_tokenizer = CharacterTokenizer(txt)
    visualizer = TokenScatterVisualizer(PlotSettings(), char_tokenizer)
    visualizer.show()

    word_tokenizer = WordTokenizer(txt)
    word_visualizer = TokenScatterVisualizer(PlotSettings(), word_tokenizer)
    word_visualizer.show()


if __name__ == "__main__":
    main()
