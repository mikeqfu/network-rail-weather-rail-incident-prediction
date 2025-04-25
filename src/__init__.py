"""
Initialisation.
"""

import datetime
import json
import pkgutil

from . import modeller, preprocessor, shaft

metadata = json.loads(pkgutil.get_data(__name__, ".metadata").decode())

__project__ = metadata['Project']
__desc__ = metadata['Description']
__author__ = metadata['Authors']
__copyright__ = f'2024-{datetime.datetime.now().year}, {__author__}'
__members__ = metadata['Team Members']
__version__ = metadata['Version']
__license__ = metadata['License']
__kickoff__ = metadata['Project Start']

__all__ = ['modeller', 'preprocessor', 'shaft']
