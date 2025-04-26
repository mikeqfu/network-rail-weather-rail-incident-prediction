"""
Test the subpackage ``src.utils``.
"""

import pytest


def test_make_filename():
    from src.utils import make_filename

    name = "filename"  # None

    route_name, weather_category = None, None
    filename = make_filename(name, route_name, weather_category)
    assert filename == 'filename.pkl'

    route_name, weather_category = None, 'Heat'
    filename = make_filename(None, route_name, weather_category, "test1")
    assert filename == 'Heat-test1.pkl'

    filename = make_filename(name, route_name, weather_category, "test1", "test2")
    assert filename == 'filename-Heat-test1-test2.pkl'

    filename = make_filename(name, 'Anglia', weather_category, "test2")
    assert filename == 'filename-Anglia-Heat-test2.pkl'

    filename = make_filename(name, 'North and East', 'Heat', "test1", "test2")
    assert filename == 'filename-N&E-Heat-test1-test2.pkl'


def test_get_index_of_dict_in_list():
    from src.utils import get_index_of_dict_in_list

    lst_of_dict = [{'a': 1}, {'b': 2}, {'c': 3}]
    key = 'b'
    val = 2

    index_of_dict = get_index_of_dict_in_list(lst_of_dict, key, val)
    assert index_of_dict == 1


if __name__ == '__main__':
    pytest.main()
