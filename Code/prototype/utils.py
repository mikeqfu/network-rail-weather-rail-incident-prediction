import os

from uni_utils import cd

# ====================================================================================================================
""" Change directories """


# Change directory to "5 - Publications\\...\\Figures"
def cdd_pub_fig_prototype(*sub_dir):
    path = cd("Paperwork\\5 - Publications\\1 - Prototype\\0 - Ingredients", "1 - Figures")
    os.makedirs(path, exist_ok=True)
    for x in sub_dir:
        path = os.path.join(path, x)
    return path
