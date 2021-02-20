"""
Tools for modelling.
"""

from pyhelpers import cd

from utils import cdd_models


# == Change directories ================================================================

def cdd_prototype(*sub_dir, mkdir=False):
    """
    Change directory to "..\\data\\models\\prototype\\" and sub-directories / a file.

    :param sub_dir: name of directory or names of directories (and/or a filename)
    :type sub_dir: str
    :param mkdir: whether to create a directory, defaults to ``False``
    :type mkdir: bool
    :return: full path to "..\\data\\models\\prototype\\" and sub-directories / a file
    :rtype: str
    """

    path = cdd_models("prototype", *sub_dir, mkdir=mkdir)

    return path


def cd_prototype_dat(*sub_dir, mkdir=True):
    """
    Change directory to "..\\data\\models\\prototype\\dat\\" and sub-directories / a file.

    :param sub_dir: name of directory or names of directories (and/or a filename)
    :type sub_dir: str
    :param mkdir: whether to create a directory, defaults to ``False``
    :type mkdir: bool
    :return: full path to "..\\data\\models\\prototype\\dat\\" and sub-directories / a file
    :rtype: str
    """

    path = cdd_prototype("dat", *sub_dir, mkdir=mkdir)

    return path


def cd_prototype_fig_pub(*sub_dir, mkdir=False):
    """
    Change directory to
    "docs\\5 - Publications\\1 - Prototype\\0 - Ingredients\\1 - Figures\\"
    and sub-directories / a file.

    :param sub_dir: name of directory or names of directories (and/or a filename)
    :type sub_dir: str
    :param mkdir: whether to create a directory, defaults to ``False``
    :type mkdir: bool
    :return: full path to
        "docs\\5 - Publications\\1 - Prototype\\0 - Ingredients\\1 - Figures\\"
        and sub-directories / a file
    :rtype: str
    """

    path = cd("docs\\5 - Publications\\1 - Prototype\\0 - Ingredients\\1 - Figures",
              *sub_dir, mkdir=mkdir)

    return path


def cdd_intermediate(*sub_dir, mkdir=False):
    """
    Change directory to "..\\data\\models\\intermediate\\" and sub-directories / a file.

    :param sub_dir: name of directory or names of directories (and/or a filename)
    :type sub_dir: str
    :param mkdir: whether to create a directory, defaults to ``False``
    :type mkdir: bool
    :return: full path to "..\\data\\models\\intermediate\\" and sub-directories / a file
    :rtype: str
    """

    path = cdd_models("intermediate", *sub_dir, mkdir=mkdir)

    return path


def cd_intermediate_dat(*sub_dir, mkdir=False):
    """
    Change directory to "..\\data\\models\\intermediate\\dat\\" and sub-directories / a file.

    :param sub_dir: name of directory or names of directories (and/or a filename)
    :type sub_dir: str
    :param mkdir: whether to create a directory, defaults to ``False``
    :type mkdir: bool
    :return: full path to "..\\data\\models\\intermediate\\dat\\" and sub-directories / a file
    :rtype: str
    """

    path = cdd_intermediate("dat", *sub_dir, mkdir=mkdir)

    return path


def cd_intermediate_fig_pub(*sub_dir, mkdir=False):
    """
    Change directory to "docs\\5 - Publications\\2 - Intermediate\\0 - Ingredients\\1 - Figures\\"
    and sub-directories / a file.

    :param sub_dir: name of directory or names of directories (and/or a filename)
    :type sub_dir: str
    :param mkdir: whether to create a directory, defaults to ``False``
    :type mkdir: bool
    :return: full path to "docs\\5 - Publications\\2 - Intermediate\\0 - Ingredients\\1 - Figures\\"
        and sub-directories / a file
    :rtype: str
    """

    path = cd("docs\\5 - Publications\\2 - Intermediate\\0 - Ingredients\\1 - Figures",
              *sub_dir, mkdir=mkdir)

    return path


def cdd_prototype_wind(*sub_dir, mkdir=False):
    """
    Change directory to "..\\data\\models\\prototype\\wind\\dat\\" and sub-directories / a file.

    :param sub_dir: name of directory or names of directories (and/or a filename)
    :type sub_dir: str
    :param mkdir: whether to create a directory, defaults to ``False``
    :type mkdir: bool
    :return: full path to "..\\data\\models\\prototype\\wind\\dat\\" and sub-directories / a file
    :rtype: str
    """

    path = cdd_prototype("wind", *sub_dir, mkdir=mkdir)

    return path


def cdd_prototype_wind_trial(trial_id, *sub_dir, mkdir=False):
    """
    Change directory to "..\\data\\models\\prototype\\wind\\<``trial_id``>"
    and sub-directories / a file.

    :param trial_id:
    :type trial_id:
    :param sub_dir: name of directory or names of directories (and/or a filename)
    :type sub_dir: str
    :param mkdir: whether to create a directory, defaults to ``False``
    :type mkdir: bool
    :return: full path to "..\\data\\models\\prototype\\wind\\data\\" and sub-directories / a file
    :rtype: str
    """

    path = cdd_prototype("wind", "{}".format(trial_id), *sub_dir, mkdir=mkdir)

    return path


def cdd_prototype_heat(*sub_dir, mkdir=False):
    """
    Change directory to "..\\data\\models\\prototype\\heat\\dat\\" and sub-directories / a file.

    :param sub_dir: name of directory or names of directories (and/or a filename)
    :type sub_dir: str
    :param mkdir: whether to create a directory, defaults to ``False``
    :type mkdir: bool
    :return: full path to "..\\data\\models\\prototype\\heat\\dat\\" and sub-directories / a file
    :rtype: str
    """

    path = cdd_prototype("heat", *sub_dir, mkdir=mkdir)

    return path


def cdd_prototype_heat_trial(trial_id, *sub_dir, mkdir=False):
    """
    Change directory to "..\\data\\models\\prototype\\heat\\<``trial_id``>"
    and sub-directories / a file.

    :param trial_id:
    :type trial_id:
    :param sub_dir: name of directory or names of directories (and/or a filename)
    :type sub_dir: str
    :param mkdir: whether to create a directory, defaults to ``False``
    :type mkdir: bool
    :return: full path to "..\\data\\models\\prototype\\heat\\data\\"
        and sub-directories / a file
    :rtype: str
    """

    path = cdd_prototype("heat", "{}".format(trial_id), *sub_dir, mkdir=mkdir)

    return path


def cdd_intermediate_heat(*sub_dir, mkdir=False):
    """
    Change directory to "..\\data\\models\\intermediate\\heat\\dat\\" and sub-directories / a file.

    :param sub_dir: name of directory or names of directories (and/or a filename)
    :type sub_dir: str
    :param mkdir: whether to create a directory, defaults to ``False``
    :type mkdir: bool
    :return: full path to "..\\data\\models\\intermediate\\heat\\dat\\" and sub-directories / a file
    :rtype: str
    """

    path = cdd_intermediate("heat", *sub_dir, mkdir=mkdir)
    return path


def cdd_intermediate_heat_trial(trial_id, *sub_dir, mkdir=False):
    """
    Change directory to "..\\data\\models\\intermediate\\heat\\<``trial_id``>"
    and sub-directories / a file.

    :param trial_id:
    :type trial_id: int, str
    :param sub_dir: name of directory or names of directories (and/or a filename)
    :type sub_dir: str
    :param mkdir: whether to create a directory, defaults to ``False``
    :type mkdir: bool
    :return: full path to "..\\data\\models\\intermediate\\heat\\data\\"
        and sub-directories / a file
    :rtype: str
    """

    path = cdd_intermediate("heat", "{}".format(trial_id), *sub_dir, mkdir=mkdir)
    return path

