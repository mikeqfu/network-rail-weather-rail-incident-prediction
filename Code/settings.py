""" Settings """

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Set preferences for plotting
def mpl_preferences(use_cambria=False, reset=False):
    """
    Get a list of supported file formats for matplotlib savefig() function
      plt.gcf().canvas.get_supported_filetypes()  # Aside: "gcf" is short for "get current fig" manager
      plt.gcf().canvas.get_supported_filetypes_grouped()
    """
    if not reset:
        if use_cambria:  # Use the font, 'Cambria'
            # Add 'Cambria' and 'Cambria Math' to the front of the 'font.serif' list
            plt.rcParams['font.serif'] = ['Cambria'] + plt.rcParams['font.serif']
            plt.rcParams['font.serif'] = ['Cambria Math'] + plt.rcParams['font.serif']
            # Set 'font.family' to 'serif', so that matplotlib will use that list
            plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.size'] = 13
        plt.rcParams['font.weight'] = 'normal'
        plt.rcParams['legend.labelspacing'] = 0.7
        plt.style.use('ggplot')
    else:
        plt.rcParams = plt.rcParamsDefault
        plt.style.use('classic')


# Set preferences for displaying results
def np_preferences(reset=False):
    if not reset:
        np.core.arrayprint._line_width = 120
    else:
        np.core.arrayprint._line_width = 80  # 75


# Set preferences for displaying results
def pd_preferences(reset=False):
    if not reset:
        pd.set_option('display.precision', 2)
        pd.set_option('expand_frame_repr', False)  # Set the representation of DataFrame NOT to wrap
        pd.set_option('display.width', 600)  # Set the display width
        pd.set_option('precision', 4)
        pd.set_option('display.max_columns', 100)
        pd.set_option('display.max_rows', 20)
        pd.set_option('io.excel.xlsx.writer', 'xlsxwriter')
        pd.set_option('mode.chained_assignment', None)
        pd.set_option('display.float_format', lambda x: '%.4f' % x)
    else:
        pd.reset_option('all')
