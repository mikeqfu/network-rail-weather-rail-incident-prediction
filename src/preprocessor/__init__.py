"""
Preprocess all the available data.
"""

from .glossary import DelayAttributionGlossary
from .metex import METEX
from .network import Anglia
from .schedule8 import Schedule8IncidentReports
from .threshold import WeatherThresholds
from .vegetation import Vegetation
from .weather import MIDAS, UKCP09

__all__ = [
    'glossary', 'DelayAttributionGlossary',
    'metex', 'METEX',
    'threshold', 'WeatherThresholds',
    'schedule8', 'Schedule8IncidentReports',
    'network', 'Anglia',
    'vegetation', 'Vegetation',
    'weather', 'MIDAS', 'UKCP09',
]
