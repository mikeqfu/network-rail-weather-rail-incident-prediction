import os

import pandas as pd
import pytest
from pyhelpers.dirs import normalize_pathname

from src.shaft.furlong import FurlongHandler


class TestFurlongHandler:

    @pytest.fixture(scope='class')
    def fur(self):
        # fur = FurlongHandler()
        return FurlongHandler()

    def test_cdd(self, fur):
        path = os.path.relpath(fur.cdd())
        assert normalize_pathname(path) == 'data/network/furlongs'

    def test_adjust_incident_mileages(self, fur):
        fur.VEGETATION.view_vegetation_condition2(route_name='Anglia')
        assert isinstance(fur.VEGETATION.vegetation_condition2, pd.DataFrame)

        fur.METEX.view_schedule8_incident_locations('Anglia', 'Wind', start_end_elr=True)
        assert isinstance(fur.METEX.schedule8_incident_locations, pd.DataFrame)


if __name__ == '__main__':
    pytest.main()
