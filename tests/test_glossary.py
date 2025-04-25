import os

import pytest
from pyhelpers.dirs import normalize_pathname

from src.preprocessor.glossary import DelayAttributionGlossary


class TestDelayAttributionGlossary:

    @pytest.fixture(scope='class')
    def dag(self):
        # dag = DelayAttributionGlossary()
        return DelayAttributionGlossary()

    def test_cdd(self, dag):
        path = os.path.relpath(dag._cdd())
        assert normalize_pathname(path) == 'data/metex/incidents/delay_attribution/glossary'

    def test_path_to_original_file(self, dag):
        path = os.path.relpath(dag.path_to_original_file())
        assert normalize_pathname(path) == ('data/metex/incidents/delay_attribution/glossary/'
                                            'historic_delay_attribution_glossary.xlsx')

    def test_download_dag(self, dag, capfd):
        dag.download_dag(confirmation_required=False, verbose=True)
        out, _ = capfd.readouterr()
        assert dag.FILENAME in out
        assert os.path.exists(dag.path_to_original_file()) and os.path.isfile(
            dag.path_to_original_file())

    @pytest.mark.parametrize('update', [True, False])
    def test_read_delay_attr_glossary(self, dag, update):
        delay_attr_glossary = dag.read_data(update=update, verbose=True)

        assert list(delay_attr_glossary.keys()) == [
            'Stanox Codes',
            'Period Dates',
            'Incident Reason',
            'Responsible Manager',
            'Reactionary Reason Code',
            'Performance Event Code',
            'Service Group Code',
            'Operator Name',
            'Train Service Code']


if __name__ == '__main__':
    pytest.main()
