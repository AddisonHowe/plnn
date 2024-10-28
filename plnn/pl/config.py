"""Plotting configurations.

"""

import yaml
from pathlib import Path

DEFAULT_CMAP = "pastel_plasma"  # defined in module custom_colors

from .custom_colors import CHIR_COLOR  # defined in module custom_colors
from .custom_colors import FGF_COLOR   # defined in module custom_colors
SIG1_COLOR = CHIR_COLOR
SIG2_COLOR = FGF_COLOR

from .custom_colors import CELL_COLORS

PHI1_C_ATTRACTOR_COLOR = CELL_COLORS['cell blue']
PHI1_L_ATTRACTOR_COLOR = CELL_COLORS['cell green']
PHI1_R_ATTRACTOR_COLOR = CELL_COLORS['cell brown']
PHI1_SADDLE_COLOR = 'r'

PHI1_COLOR_DICT = {
    'saddle': PHI1_SADDLE_COLOR,
    'c': PHI1_C_ATTRACTOR_COLOR,
    'l': PHI1_L_ATTRACTOR_COLOR,
    'r': PHI1_R_ATTRACTOR_COLOR,
}

PHI2_C_ATTRACTOR_COLOR = CELL_COLORS['cell brown']
PHI2_T_ATTRACTOR_COLOR = 'cyan'
PHI2_B_ATTRACTOR_COLOR = 'pink'
PHI2_M_ATTRACTOR_COLOR = 'grey'
PHI2_FLIP_CURVE_COLOR = 'purple'
PHI2_SADDLE_COLOR = 'r'

PHI2_COLOR_DICT = {
    'saddle': PHI2_SADDLE_COLOR,
    'c': PHI2_C_ATTRACTOR_COLOR,
    't': PHI2_T_ATTRACTOR_COLOR,
    'b': PHI2_B_ATTRACTOR_COLOR,
    'maximum': PHI2_M_ATTRACTOR_COLOR,
}

# Check for config.yml and load if present
try:
    config_fpath = Path("config.yml")
    if Path(config_fpath).is_file():
        with open(config_fpath, 'r') as f:
            config_dict = yaml.safe_load(f)
            config_dict = config_dict if config_dict else {}
            print(f"Loaded configuration file {config_fpath}")
        keys = config_dict.keys()
        if "DEFAULT_CMAP" in keys:
            DEFAULT_CMAP = config_dict["DEFAULT_CMAP"]
        if "CHIR_COLOR" in keys:
            CHIR_COLOR = config_dict["CHIR_COLOR"]
        if "FGF_COLOR" in keys:
            FGF_COLOR = config_dict["FGF_COLOR"]
        if "SIG1_COLOR" in keys:
            SIG1_COLOR = config_dict["SIG1_COLOR"]
        if "SIG2_COLOR" in keys:
            SIG2_COLOR = config_dict["SIG2_COLOR"]
    
except FileNotFoundError:
    pass
else:
    pass
