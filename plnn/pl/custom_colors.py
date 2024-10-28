"""Custom colormaps.

Defines and registers the following custom colormaps:
    pastel_plasma
    
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap


# CHIR_COLOR = (0.57, 0.26, 0.98)  # Signal colors used in Sáez et al.
# FGF_COLOR  = (0.70, 0.09, 0.32)

# CHIR_COLOR = (0.56, 0.96, 1.00)  # Plasma Blue
# FGF_COLOR  = (0.87, 0.26, 0.77)  # Plasma Pink

CHIR_COLOR = (0.18, 0.22, 0.73)  # Dark Plasma Blue
FGF_COLOR  = (0.70, 0.09, 0.32)

CELL_BLUE = (0.11, 0.49, 0.74)  # bluish
CELL_GREEN = (0.29, 0.52, 0.08)  # greenish
CELL_BROWN = (0.58, 0.20, 0.18)  # brownish

CELL_COLORS = {
    'cell blue': CELL_BLUE,
    'cell green': CELL_GREEN,
    'cell brown': CELL_BROWN,
}

def _lighten_color(color, amount=0.5):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be a matplotlib color string, hex string, or RGB tuple.
    """
    import matplotlib.colors as mc
    import colorsys

    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])


#~~~  Pastel Plasma  ~~~#
_plasma = plt.get_cmap('plasma')
new_colors = [_lighten_color(_plasma(i), 0.5) 
              for i in np.linspace(0, 1, _plasma.N)]

pastel_plasma = LinearSegmentedColormap.from_list(
    'pastel_plasma', new_colors, N=_plasma.N
)

#~~~  Signal Colormap  ~~~#
_tab10 = plt.get_cmap('tab10')
existing_colors = [_tab10(i) for i in range(_tab10.N)]
signal_colors = [CHIR_COLOR, FGF_COLOR]
new_colors = signal_colors + existing_colors  # prepend signal colors
signal_cmap = ListedColormap(new_colors, name="signal_cmap")


#~~~  Register  ~~~#
plt.colormaps.register(cmap=pastel_plasma)
plt.colormaps.register(cmap=signal_cmap)
