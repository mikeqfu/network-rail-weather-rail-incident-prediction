import os

import pytest
from pyhelpers.dirs import normalize_pathname

from src.preprocessor.metex import METEX


class TestMETExLite:

    @pytest.fixture(scope='class')
    def mtx(self, db_instance=None, use_old_db=False):
        # mtx = METEX()
        return METEX(db_instance=db_instance, use_old_db=use_old_db)

    def test_cdd(self, mtx):
        path = os.path.relpath(mtx.cdd())
        assert normalize_pathname(path) == 'data/metex/database'


if __name__ == '__main__':
    pytest.main()
