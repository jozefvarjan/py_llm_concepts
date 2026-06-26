from tokenizer import CharacterTokenizer, WordTokenizer, PretrainedTokenizer, TokenStats, Stat
from visualizer import TokenScatterVisualizer, TokenBarVisualizer, PlotSettings


def main():
    txt: str = 'The way you do anything is the way you do everything.'

    txt2 = 'Each token corresponds to a whole word or punctuation. That happened here ' \
    'because all the words in this sentence are common English words. There is, ' \
    'in general, no guarantee or constraint that tokens correspond to whole words.'
 
    # -----  
    char_tokenizer = CharacterTokenizer(txt)
    char_tokenizer2 = CharacterTokenizer(txt2)
    visualizer = TokenScatterVisualizer(PlotSettings(), char_tokenizer)
    # visualizer.show()
    # -----
    word_tokenizer = WordTokenizer(txt)
    word_tokenizer2 = WordTokenizer(txt2)
    word_visualizer = TokenScatterVisualizer(PlotSettings(), word_tokenizer)
    # word_visualizer.show()
    # -----
    
    auto_tokenizer = PretrainedTokenizer(txt)
    auto_tokenizer2 = PretrainedTokenizer(txt2)

    bar_visualizer = TokenBarVisualizer(
        PlotSettings(),
        {
            'char_tokenizer': char_tokenizer,
            'char_tokenizer2': char_tokenizer2,
            'word_tokenizer': word_tokenizer,
            'word_tokenizer2': word_tokenizer2,
            'auto_tokenizer': auto_tokenizer,
            'auto_tokenizer2': auto_tokenizer2,
        },
    )
    bar_visualizer.show()

    


if __name__ == "__main__":
    main()
