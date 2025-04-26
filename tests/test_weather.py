import datetime

import pandas as pd
import pytest

from src.preprocessor.weather import MIDAS


class TestMIDAS:

    @pytest.fixture(scope='class')
    def midas(self, db_instance=None):
        # midas = MIDAS(db_instance)
        return MIDAS(db_instance=db_instance)

    @pytest.mark.parametrize('update', [True, False])
    def test_read_radiation_monitoring_stations(self, midas, update):
        midas.read_radiation_monitoring_stations(update=update, verbose=True)
        assert midas.radiation_monitoring_stations.shape == (240, 11)

    @pytest.mark.parametrize('prep', [True, False])
    def test__radtob_table(self, midas, prep):
        midas._radtob_table(prep=prep, verbose=True)
        midas._radtob_table(prep=prep, verbose=True)
        assert len(midas.radtob_table_header) == 22

    def test__supplement_data(self, midas):
        dat_ = midas._supplement_data(verbose=True)
        assert isinstance(dat_, pd.DataFrame)

    def test_query_radtob_by_grid_datetime(self, midas):
        src_id = 9
        route_name = 'Anglia'

        start_dt = datetime.datetime(2018, 6, 1, 12)  # '2018-06-01 12:00:00'
        end_dt = datetime.datetime(2018, 6, 1, 13)  # '2018-06-01 13:00:00'
        period = pd.date_range(start=start_dt, end=end_dt, freq='H')
        midas_radtob = midas.query_radtob_by_grid_datetime(src_id, period, route_name)
        assert isinstance(midas_radtob, pd.DataFrame)

        start_dt = datetime.datetime(2018, 6, 1, 12)  # '2018-06-01 12:00:00'
        end_dt = datetime.datetime(2018, 12, 1, 12)  # '2018-12-01 12:00:00'
        period = pd.date_range(start=start_dt, end=end_dt, freq='H')
        midas_radtob = midas.query_radtob_by_grid_datetime(src_id, period, route_name)
        assert isinstance(midas_radtob, pd.DataFrame)

        start_dt = datetime.datetime(2018, 6, 1, 12)  # '2018-06-01 12:00:00'
        end_dt = datetime.datetime(2019, 6, 1, 12)  # '2018-12-01 12:00:00'
        period = pd.date_range(start=start_dt, end=end_dt, freq='H')
        midas_radtob = midas.query_radtob_by_grid_datetime(src_id, period, route_name)
        assert isinstance(midas_radtob, pd.DataFrame)


if __name__ == '__main__':
    pytest.main()
