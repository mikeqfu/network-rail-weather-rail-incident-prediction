import os

from pyhelpers.store import save_fig

from utils import cd, cdd

# ====================================================================================================================
""" Change directories """


def cdd_prototype(*sub_dir):
    path = cdd("Models\\prototype")
    for x in sub_dir:
        path = os.path.join(path, x)
    return path


# Change directory to "Data\\Model" and sub-directories
def cd_prototype_dat(*sub_dir):
    path = cdd_prototype("dat")
    for x in sub_dir:
        path = os.path.join(path, x)
    return path


# Change directory to "5 - Publications\\...\\Figures"
def cd_prototype_fig_pub(*sub_dir):
    path = cd("Paperwork\\5 - Publications\\1 - Prototype\\0 - Ingredients", "1 - Figures")
    os.makedirs(path, exist_ok=True)
    for x in sub_dir:
        path = os.path.join(path, x)
    return path


# A function for saving the plots
def save_hotpots_fig(fig, keyword, show_metex_weather_cells, show_osm_landuse_forest, show_nr_hazardous_trees,
                     save_as, dpi):
    """
    :param fig: [matplotlib.figure.Figure]
    :param keyword: [str] a keyword for specifying the filename
    :param show_metex_weather_cells: [bool]
    :param show_osm_landuse_forest: [bool]
    :param show_nr_hazardous_trees: [bool]
    :param save_as: [str]
    :param dpi: [int] or None
    :return:
    """
    if save_as is not None:
        if save_as.lstrip('.') in fig.canvas.get_supported_filetypes():
            suffix = zip([show_metex_weather_cells, show_osm_landuse_forest, show_nr_hazardous_trees],
                         ['cell', 'veg', 'haz'])
            filename = '_'.join([keyword] + [v for s, v in suffix if s is True])
            path_to_file = cd_prototype_fig_pub("Hotspots", filename + save_as)
            save_fig(path_to_file, dpi=dpi)
