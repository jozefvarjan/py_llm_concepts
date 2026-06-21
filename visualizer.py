from abc import ABC, abstractmethod
from pydantic import BaseModel, Field
from tokenizer import Tokenizer, CharacterTokenizer
import matplotlib.pyplot as plt
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
