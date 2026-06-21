from tokenizer import CharacterTokenizer
from visualizer import TokenScatterVisualizer, PlotSettings


def main():
    txt: str = 'The way you do anything is the way you do everything.'
   
    tokenizer = CharacterTokenizer(txt)

    visualizer = TokenScatterVisualizer(PlotSettings(), tokenizer)
    visualizer.show()


if __name__ == "__main__":
    main()
