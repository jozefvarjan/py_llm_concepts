from abc import ABC, abstractmethod
from pydantic import BaseModel, Field
from tokenizer import Tokenizer, TokenStats, Stat
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.ticker import MaxNLocator


class PlotSettings(BaseModel):
    plt_style: str = '_mpl-gallery'
    figsize: tuple[int, int] = (7, 3)            # base / minimum figure size
    height_per_char: float = Field(default=0.09, ge=0.0)  # inches added per char
    width_per_char: float = Field(default=0.045, ge=0.0)  # half of height rate
    max_height: float = Field(default=40.0, gt=0.0)       # cap so it stays usable
    max_width: float = Field(default=30.0, gt=0.0)        # cap so it stays usable
    subpl_left: float = Field(default=0.08, ge=0.0, le=1.0)
    subpl_right: float = Field(default=0.92, ge=0.0, le=1.0)
    subpl_top: float = Field(default=0.95, ge=0.0, le=1.0)
    subpl_botton: float = Field(default=0.08, ge=0.0, le=1.0)

    @staticmethod
    def _scale(base: float, rate: float, text_len: int, cap: float) -> float:
        """Grow ``base`` by ``rate`` per char, clamped to ``[base, cap]``."""
        return min(max(base, base + text_len * rate), cap)

    def figsize_for(self, text_len: int) -> tuple[float, float]:
        """
            Scale the figure with the input text length so a longer text does
            not squash the plotted features. Height is the preferred growth
            dimension (``height_per_char`` per char); width grows at half that
            rate (``width_per_char``). Both stay within
            ``[base, max_height/max_width]``.
        """
        base_width, base_height = self.figsize
        width = self._scale(base_width, self.width_per_char, text_len, self.max_width)
        height = self._scale(base_height, self.height_per_char, text_len, self.max_height)
        return (width, height)


class DataVisualizer(ABC):
    """
        Base visualizer.

        Prepares a figure/axes from ``PlotSettings`` and pulls its data from a
        ``Tokenizer``. Subclasses implement ``plot`` to draw a specific chart;
        the figure setup and ``show`` plumbing is inherited.
    """

    def __init__(self, settings: PlotSettings, data_aquirer: Tokenizer = None):
        self.settings = settings
        self.data_aquirer = data_aquirer

    def _prepare_axes(self, text_len: int | None = None):
        """Apply the style and create a figure/axes from the settings.

        When ``text_len`` is given the figure width scales with it so longer
        inputs are not squashed; otherwise the base ``figsize`` is used.
        """
        plt.style.use(self.settings.plt_style)
        figsize = (
            self.settings.figsize_for(text_len)
            if text_len is not None
            else self.settings.figsize
        )
        fig, ax = plt.subplots(figsize=figsize)
        plt.subplots_adjust(
            left=self.settings.subpl_left,
            right=self.settings.subpl_right,
            top=self.settings.subpl_top,
            bottom=self.settings.subpl_botton,
        )
        return fig, ax

    @abstractmethod
    def plot(self):
        """Draw the chart and return the primary axes."""

    def show(self):
        self.plot()
        plt.show()


class TokenScatterVisualizer(DataVisualizer):
    """
        Scatter every character of the tokenizer's input text by its position
        in the text (x) against its position in ``char_map`` (left y axis),
        with the character itself shown on the right y axis.
    """

    def plot(self):
        # token_map item: (char, (position_in_text, token_index_in_char_map))
        token_map = self.data_aquirer.token_map()
        x = [position for _, (position, _) in token_map]
        y = [token_index for _, (_, token_index) in token_map]

        fig, ax_l = self._prepare_axes(text_len=len(x))

        # x axis spans the length of the input text, left y axis is the token index
        ax_l.scatter(x, y, marker='s', s=50, facecolor='C0', edgecolor='k')
        ax_l.set_xlabel('char position in text [index]')
        ax_l.set_ylabel('char position in char_map [token index]')
        ax_l.set(xlim=(-1, len(x)))
        # x ticks are index positions, keep them integer (no .0 floats)
        ax_l.xaxis.set_major_locator(MaxNLocator(integer=True))
        # avoid vertical grid lines (keep only horizontal)
        ax_l.xaxis.grid(False)

        # ticks mapping shared by both y axes: char_map position <-> character
        token_to_char = {token_index: char for char, (_, token_index) in token_map}
        ticks = sorted(token_to_char)  # integer char positions in char_map

        # left y axis: integer char position in char_map
        ax_l.set_yticks(ticks)
        ax_l.set_yticklabels([str(t) for t in ticks])

        # right y axis: the character mapped to each char_map position
        ax_r = ax_l.twinx()
        ax_r.set_ylim(ax_l.get_ylim())
        ax_r.set_yticks(ticks)
        ax_r.set_yticklabels([repr(token_to_char[t]) for t in ticks])
        ax_r.set_ylabel('character')

        return ax_l


class TokenBarVisualizer(DataVisualizer):
    """
        Grouped bar chart of token statistics pulled from ``TokenStats``.

        All ``Tokenizer`` instances share one set of axes. The x axis holds the
        three category groups — characters, words, tokens — and within each
        group every tokenizer contributes a total/unique bar pair. Each
        tokenizer keeps its own base colour; the total bar uses the base hue
        and the unique bar a lighter hue of the same colour, so a tokenizer is
        recognisable across all categories.
    """

    # (category label, total stat, unique stat) for each bar pair
    CATEGORIES = (
        ('characters', Stat.CH_CNT, Stat.CH_UNIQ),
        ('words', Stat.W_CNT, Stat.W_UNIQ),
        ('tokens', Stat.T_CNT, Stat.T_UNIQ),
    )

    BAR_WIDTH = 0.2  # half of the previous 0.4 — thinner bars

    def __init__(
        self,
        settings: PlotSettings,
        tokenizers: Tokenizer | list[Tokenizer] | dict[str, Tokenizer],
    ):
        if isinstance(tokenizers, Tokenizer):
            tokenizers = [tokenizers]
        # normalise to a list of (instance_name, tokenizer); a plain list has
        # no names so the legend falls back to just the class name
        if isinstance(tokenizers, dict):
            named = list(tokenizers.items())
        else:
            named = [(None, tokenizer) for tokenizer in tokenizers]
        if not named:
            raise ValueError("TokenBarVisualizer needs at least one tokenizer.")
        # keep the first as the inherited single data_aquirer for compatibility
        super().__init__(settings, named[0][1])
        self.named_tokenizers = named
        self.tokenizers = [tokenizer for _, tokenizer in named]

    @staticmethod
    def _lighten(color, amount: float = 0.55):
        """Blend ``color`` toward white to produce a lighter hue of it."""
        r, g, b = mcolors.to_rgb(color)
        return (r + (1 - r) * amount, g + (1 - g) * amount, b + (1 - b) * amount)

    def plot(self):
        plt.style.use(self.settings.plt_style)

        n = len(self.tokenizers)
        n_categories = len(self.CATEGORIES)
        width = self.BAR_WIDTH
        bars_per_group = 2 * n              # total + unique for each tokenizer
        group_span = bars_per_group * width
        # space category centres so neighbouring groups do not overlap
        category_step = group_span + 0.5
        centers = [c * category_step for c in range(n_categories)]

        base_width, base_height = self.settings.figsize
        # single, landscape-oriented figure that widens with the bar count
        fig_width = max(base_width, n_categories * category_step + 2)
        fig, ax = plt.subplots(figsize=(fig_width, base_height))

        palette = plt.get_cmap('tab10')
        labels = [label for label, _, _ in self.CATEGORIES]

        for idx, (inst_name, tokenizer) in enumerate(self.named_tokenizers):
            stats = TokenStats.get_stats(tokenizer)
            totals = [stats.get(total, 0) for _, total, _ in self.CATEGORIES]
            uniques = [stats.get(unique, 0) for _, _, unique in self.CATEGORIES]

            base = palette(idx % 10)
            cls = type(tokenizer).__name__
            # e.g. "CharacterTokenizer (char_tokenizer)" when a name is given
            name = f'{cls} ({inst_name})' if inst_name else cls

            # slot positions inside each group: total then unique per tokenizer
            total_slot = 2 * idx
            unique_slot = 2 * idx + 1
            total_offset = (total_slot - (bars_per_group - 1) / 2) * width
            unique_offset = (unique_slot - (bars_per_group - 1) / 2) * width

            total_x = [c + total_offset for c in centers]
            unique_x = [c + unique_offset for c in centers]

            ax.bar(total_x, totals, width, color=base, edgecolor='k',
                   label=f'{name}: total')
            ax.bar(unique_x, uniques, width, color=self._lighten(base),
                   edgecolor='k', label=f'{name}: unique')

        ax.set_ylabel('count')
        ax.set_xticks(centers)
        ax.set_xticklabels(labels)
        # counts are integers, keep the y axis free of .0 floats
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.yaxis.grid(True)
        ax.xaxis.grid(False)
        # legend to the right of the plot area, stacked vertically
        ax.legend(ncol=1, loc='center left', bbox_to_anchor=(1.02, 0.5))

        fig.tight_layout()
        return ax
