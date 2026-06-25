from tokenizer import CharacterTokenizer, WordTokenizer, PretrainedTokenizer, TokenStats, Stat
from visualizer import TokenScatterVisualizer, PlotSettings


def main():
    txt: str = 'The way you do anything is the way you do everything.'

    txt = 'Each token corresponds to a whole word or punctuation. That happened here ' \
    'because all the words in this sentence are common English words. There is, ' \
    'in general, no guarantee or constraint that tokens correspond to whole words.'
 
    # -----  
    # char_tokenizer = CharacterTokenizer(txt)
    # visualizer = TokenScatterVisualizer(PlotSettings(), char_tokenizer)
    # visualizer.show()
    # -----
    # word_tokenizer = WordTokenizer(txt)
    # word_visualizer = TokenScatterVisualizer(PlotSettings(), word_tokenizer)
    # word_visualizer.show()
    # -----
    
    auto_tokenizer = PretrainedTokenizer(txt)
    print(auto_tokenizer.decode())
    print(auto_tokenizer.token_map())

    at_stats = TokenStats.get_stats(auto_tokenizer)
    print(at_stats[Stat.CH_CNT])
    print(at_stats[Stat.CH_UNIQ])
    print(at_stats[Stat.W_CNT])
    print(at_stats[Stat.W_UNIQ])




if __name__ == "__main__":
    main()
