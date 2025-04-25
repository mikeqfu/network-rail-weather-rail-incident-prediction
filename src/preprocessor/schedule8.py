"""
Preprocess the data (Microsoft Excel spreadsheet) of Schedule 8 weather incidents.
"""

import itertools
import os
import re

import numpy as np
import pandas as pd
import shapely
import shapely.wkt
from pyhelpers.dirs import cd, cdd
from pyhelpers.geom import osgb36_to_wgs84, wgs84_to_osgb36
from pyhelpers.store import load_data, xlsx_to_csv
from pyhelpers.text import find_similar_str
from pyrcs.converter import fix_stanox, mileage_str_to_num
from pyrcs.line_data import ELRMileages, LocationIdentifiers
from pyrcs.other_assets import Stations
from pyrcs.utils import fetch_location_names_errata

from src.preprocessor import DelayAttributionGlossary, METEX
from src.utils import Handler, WxRailIncidentsPred, get_subset


class Schedule8IncidentReports(Handler):
    """
    Reports of Schedule 8 Incidents.
    """

    #: Name of the data.
    DATA_NAME: str = 'Schedule 8 weather incidents'
    #: Pathname of the local data directory.
    DATA_DIR: str = os.path.relpath(cdd("metex/incidents/reports"))
    #: Pathname of the local data directory of railway codes.
    RC_DATA_DIR: str = os.path.relpath(cdd("network/railway_codes"))

    #: Filename 1.
    FILENAME_1: str = "schedule8_weather_incidents"
    #: Filename 2.
    FILENAME_2: str = "schedule8_weather_incidents_02062006_31032014"

    #: Schema name.
    SCHEMA_NAME: str = 'NR_Schedule8IncidentReports'

    #: Table name for the data of location identifiers.
    LOCID_TABLE_NAME: str = 'Location_Identifiers'

    def __init__(self, db_instance=None):
        """
        :param db_instance: An instance of the class :class:`~src.utils.WxRailIncidentsPred`,
            connecting to the project database in a PostgreSQL server; defaults to ``None``.
        :type db_instance: WxRailIncidentsPred | None

        :ivar pandas.DataFrame | None location_identifiers:
        :ivar pandas.DataFrame | None location_identifiers_:
        :ivar pandas.DataFrame | None schedule8_weather_incidents:
        :ivar collections.OrderedDict | None schedule8_weather_incidents_02062006_31032014:

        **Examples**::

            >>> from src.preprocessor.schedule8 import Schedule8IncidentReports
            >>> s8wir = Schedule8IncidentReports()
            >>> s8wir.DATA_NAME
            'Schedule 8 weather incidents'
        """

        super().__init__(db_instance=db_instance)

        self.location_identifiers = None
        self.location_identifiers_ = None

        self.schedule8_weather_incidents = None
        self.schedule8_weather_incidents_02062006_31032014 = None

    def _cdd(self, *sub_dir, mkdir=False):
        """
        Change to the data directory, any of its subdirectories or a file in it.

        :param sub_dir: Subdirectory name(s) or filename(s).
        :type sub_dir: str
        :param mkdir: Whether to create a directory; defaults to ``False``.
        :type mkdir: bool
        :return: Path to the data directory, any of its subdirectories or
            a file in it, which is associated with this class
            :py:class:`~src.preprocessor.schedule8.Schedule8IncidentReports`.
        :rtype: str

        **Examples**::

            >>> from src.preprocessor.schedule8 import Schedule8IncidentReports
            >>> import os
            >>> s8wir = Schedule8IncidentReports()
            >>> os.path.relpath(s8wir._cdd())
            'data\\metex\\incidents\\reports'
        """

        path = cd(self.DATA_DIR, *sub_dir, mkdir=mkdir)

        return path

    # == Location metadata =======================================================================

    @staticmethod
    def _preprocess_sheet_data(data):
        """
        Preprocess the location data.

        :return: Preprocessed data of location metadata.
        :rtype: pandas.DataFrame
        """

        dat = data.copy()

        dat.columns = [x.upper().replace(' ', '_') for x in data.columns]
        dat.replace({'\xa0\xa0': ' '}, regex=True, inplace=True)
        dat.replace({'\xa0': ''}, regex=True, inplace=True)
        dat.replace({'LINE_DESCRIPTION': {re.compile('[Ss]tn'): 'Station',
                                          re.compile('[Ll]oc'): 'Location',
                                          re.compile('[Jj]n'): 'Junction',
                                          re.compile('[Ss]dg'): 'Siding',
                                          re.compile('[Ss]dgs'): 'Sidings'}}, inplace=True)
        dat['LOOKUP_NAME_Raw'] = dat['LOOKUP_NAME']
        dat.fillna({'STANOX': '', 'STANME': ''}, inplace=True)

        dat['STANOX'] = dat['STANOX'].map(fix_stanox)

        return dat

    @staticmethod
    def _extract_one(lookup_name, ref_loc):
        if isinstance(ref_loc, (list, tuple)):
            n = find_similar_str(lookup_name.replace(' ', ''), ref_loc, engine='fuzz')
        elif pd.isnull(ref_loc):
            n = lookup_name
        else:
            n = ref_loc

        return n

    def _cleanse_tiploc_locationlyr_sheet(self, data):
        """
        Cleanse 'TIPLOC_LocationsLyr' sheet in location metadata.

        :param data: Location data.
        :type data: pandas.DataFrame
        :return: Location data with cleansed 'TIPLOC_LocationsLyr' sheet.
        :rtype: pandas.DataFrame
        """

        dat = self._preprocess_sheet_data(data)
        # dat = s8ir._preprocess_sheet_data(data)

        cols_with_na = ['STATUS', 'LINE_DESCRIPTION', 'QC_STATUS', 'DESCRIPTIO', 'BusinessRef']
        dat.fillna({x: '' for x in cols_with_na}, inplace=True)

        dat['TIPLOC'] = dat['TIPLOC'].fillna('').str.upper()

        # Rectify errors in STANOX; in errata_tiploc, {'CLAPS47': 'CLPHS47'} may be problematic.
        errata = load_data(cd(self.RC_DATA_DIR, "metex_location_name_errata.json"))

        errata_stanox, errata_tiploc, errata_stanme = [{k: v} for k, v in errata.items()]
        dat = dat.replace(errata_stanox)
        dat = dat.replace(errata_stanme)
        dat = dat.replace(errata_tiploc)

        # Rectify known issues for the location names in the data set
        dat = dat.replace(fetch_location_names_errata('LOOKUP_NAME'))
        dat = dat.replace(fetch_location_names_errata('LOOKUP_NAME', regex=True))

        lid = LocationIdentifiers()

        ref_cols = ['STANOX', 'STANME', 'TIPLOC']

        # Fill in missing location names
        na_name = dat[dat['LOOKUP_NAME'].isnull()]
        ref_dict = lid.make_xref_dict(ref_cols)
        temp = na_name.join(ref_dict, on=ref_cols)
        dat.loc[na_name.index, 'LOOKUP_NAME'] = temp[['TIPLOC', 'Location']].apply(
            lambda x: find_similar_str(x.TIPLOC, x.Location, engine='fuzz')
            if isinstance(x.Location, (list, tuple)) else x.Location,
            axis=1)

        # Rectify 'LOOKUP_NAME' according to 'TIPLOC'
        na_name = dat[dat['LOOKUP_NAME'].isnull()]
        ref_dict = lid.make_xref_dict('TIPLOC')
        temp = na_name.join(ref_dict, on='TIPLOC')
        dat.loc[na_name.index, 'LOOKUP_NAME'] = temp['Location'].values

        not_na_name = dat[dat['LOOKUP_NAME'].notnull()]
        temp = not_na_name.join(ref_dict, on='TIPLOC')

        dat.loc[not_na_name.index, 'LOOKUP_NAME'] = temp.apply(
            lambda x: self._extract_one(x['LOOKUP_NAME'], x['Location']), axis=1).values

        # Rectify 'STANOX'+'STANME'
        loc_codes = lid.fetch_codes()[lid.KEY]
        loc_codes = loc_codes.drop_duplicates(['TIPLOC', 'Location']).set_index(
            ['TIPLOC', 'Location'])
        temp = dat.join(loc_codes, on=['TIPLOC', 'LOOKUP_NAME'], rsuffix='_Ref').fillna('')
        dat.loc[temp.index, ['STANOX', 'STANME']] = temp[['STANOX_Ref', 'STANME_Ref']].values

        # Try further to find missing 'LOOKUP_NAME'
        na_name = dat[dat['LOOKUP_NAME'].isnull()]

        em = ELRMileages()

        na_name_elrs_mileages = na_name[['BUSINESSREF', 'MEASURE']]
        for i, x in na_name_elrs_mileages.iterrows():
            elr, mileage = x['BUSINESSREF'], mileage_str_to_num(x['MEASURE'])
            elr_file = em.fetch_mileage_file(elr=elr)['Mileage']
            mileages = elr_file['Mileage'].map(mileage_str_to_num)
            idx = np.where(
                mileages == min(mileages, key=lambda x: abs(x - mileage_str_to_num(mileage))))
            node_name_ = elr_file.loc[idx[0][0], 'Node']

            initial = node_name_[0]
            loc_names = lid.fetch_loc_id(initial=initial)[initial.upper()]['Location']
            node_name = find_similar_str(node_name_, loc_names, engine='fuzz')

            dat.loc[i, 'LOOKUP_NAME'] = node_name

        # Update coordinates by referencing the data sourced from Railway Codes
        stn = Stations()

        station_data = stn.fetch_locations()[stn.KEY_TO_STN]
        station_data = station_data[['Station', 'Degrees Longitude', 'Degrees Latitude']].dropna()
        station_data = station_data.drop_duplicates(subset=['Station']).set_index('Station')
        temp = dat.join(station_data, on='LOOKUP_NAME')
        na_i = temp['Degrees Longitude'].notnull() & temp['Degrees Latitude'].notnull()
        dat.loc[na_i, ['DEG_LONG', 'DEG_LAT']] = temp.loc[
            na_i, ['Degrees Longitude', 'Degrees Latitude']].values

        # Finalise
        dat.dropna(subset=['LOOKUP_NAME'] + ref_cols, inplace=True)
        dat.fillna('', inplace=True)
        dat.sort_values(['OBJECTID', 'LOOKUP_NAME'], ignore_index=True, inplace=True)

        tiploc_locationlyr = dat.set_index(['STANOX', 'LOOKUP_NAME'])

        return tiploc_locationlyr

    def _cleanse_stanox_sheet(self, data, tiploc_locationlyr):
        dat = self._preprocess_sheet_data(data)

        dat.fillna({'LOOKUP_NAME_Raw': ''}, inplace=True)

        ref_cols = ['SHAPE_LENG', 'EASTING', 'NORTHING', 'GRIDREF']
        dat.drop_duplicates(ref_cols, inplace=True)

        ref_data = tiploc_locationlyr.set_index(ref_cols)
        temp = dat.join(ref_data, on=ref_cols, rsuffix='_Ref').drop_duplicates(ref_cols)

        ref_cols_ok = [c for c in temp.columns if '_Ref' in c]
        dat.loc[:, [c.replace('_Ref', '') for c in ref_cols_ok]] = temp[ref_cols_ok].values

        stanox = dat.set_index(['STANOX', 'LOOKUP_NAME'])

        return stanox

    def _location_identifiers(self, verbose=False, **kwargs):
        """
        Get data of location identifiers.

        :param verbose: Whether to print relevant information in console; defaults to ``False``.
        :type verbose: bool | int
        :return: Data of location identifiers.
        :rtype: dict

        **Examples**::

            >>> from src.preprocessor.schedule8 import Schedule8IncidentReports
            >>> s8wir = Schedule8IncidentReports()
            >>> s8wir._location_identifiers()
        """

        path_to_xlsx = cd(self.RC_DATA_DIR, "location_data.xlsx")

        # -- 'TIPLOC_LocationsLyr' -----------------------------------------------------------------
        sheet1_pathname = xlsx_to_csv(path_to_xlsx, sheet_name='1')
        sheet1_data = pd.read_csv(
            sheet1_pathname, parse_dates=['LASTEDITED', 'LAST_UPD_1'], encoding='ISO-8859-1',
            dtype={'gridref': str, 'Measure': str, 'Offset': str})
        os.remove(sheet1_pathname)

        tiploc_locationlyr = self._cleanse_tiploc_locationlyr_sheet(data=sheet1_data)

        # -- 'STANOX' ------------------------------------------------------------------------------
        sheet2_pathname = xlsx_to_csv(path_to_xlsx, sheet_name='2')
        sheet2_data = pd.read_csv(sheet2_pathname, encoding='ISO-8859-1', dtype={'gridref': str})
        os.remove(sheet2_pathname)

        stanox = self._cleanse_stanox_sheet(data=sheet2_data, tiploc_locationlyr=tiploc_locationlyr)

        # -- Dump the preprocessed data ------------------------------------------------------------

        if self.db_instance is None:
            self.db_instance = WxRailIncidentsPred(verbose=False)

        self.dump_preprocessed_data(
            data=tiploc_locationlyr, table_name='TIPLOC_LocationsLyr', verbose=verbose, pkey=[],
            **kwargs)
        self.db_instance.null_text_to_empty_string(
            table_name='TIPLOC_LocationsLyr', schema_name=self.SCHEMA_NAME)
        self.dump_preprocessed_data(
            data=stanox, table_name='STANOX', verbose=verbose, pkey=[], **kwargs)
        self.db_instance.null_text_to_empty_string(
            table_name='STANOX', schema_name=self.SCHEMA_NAME)

        # -- Create a dict -------------------------------------------------------------------------
        location_identifiers = {'TIPLOC_LocationsLyr': tiploc_locationlyr, 'STANOX': stanox}

        return location_identifiers

    def read_location_identifiers(self, update=False, verbose=False, **kwargs):
        """
        Read data of location identifiers.

        :param update:
        :param verbose:
        :param kwargs:
        :return:

        **Examples**::

            >>> from src.preprocessor.schedule8 import Schedule8IncidentReports
            >>> s8wir = Schedule8IncidentReports()
            >>> s8wir.read_location_identifiers()
            >>> list(s8wir.location_identifiers.keys())
            ['TIPLOC_LocationsLyr', 'STANOX']
            >>> s8wir.location_identifiers['TIPLOC_LocationsLyr'].shape
            (8885, 21)
            >>> s8wir.location_identifiers['STANOX'].shape
            (7774, 9)
        """

        if self.db_instance is None:
            self.db_instance = WxRailIncidentsPred(verbose=False)

        table_names = ['TIPLOC_LocationsLyr', 'STANOX']
        tables_exist = all(self.db_instance.table_exists(x, self.SCHEMA_NAME) for x in table_names)

        if tables_exist and not update:
            data = []
            for table_name in table_names:
                sql_query = self.specify_sql_query(table_name)
                dat = self.db_instance.read_sql_query(
                    sql_query, index_col=['STANOX', 'LOOKUP_NAME'], **kwargs)

                text_cols = [x for x in dat.columns if dat[x].dtype.name == 'object']
                dat.loc[:, text_cols] = dat[text_cols].fillna('')

                data.append(dat)

            location_identifiers = dict(zip(table_names, data))

        else:
            location_identifiers = self._location_identifiers(verbose=verbose)

        self.location_identifiers = location_identifiers

    # == Location data from METEX ==================================================================

    def _location_identifiers_plus(self, update=False, verbose=False, **kwargs):
        """
        Get data of location codes by assembling resources from the project database.

        :param update: Whether to check on update and proceed to update the package data;
            defaults to ``False``.
        :type update: bool
        :param verbose: Whether to print relevant information in console; defaults to ``False``.
        :type verbose: bool | int
        :return: Data of location codes
        :rtype: pandas.DataFrame | None

        **Examples**::

            >>> from src.preprocessor.schedule8 import Schedule8IncidentReports
            >>> s8wir = Schedule8IncidentReports()
            >>> loc_dat_plus = s8wir._location_identifiers_plus(update=True, verbose=True)
            >>> loc_dat_plus.shape
            (10522, 30)
        """

        mt = METEX()
        mt.read_location()
        mt.read_stanox_section()

        # STANOX section
        stanox_section = mt.stanox_section.copy()

        # Get metadata by 'LocationId'
        loc_id = stanox_section.join(mt.location, on='LocationId').dropna().reset_index()
        loc_id[['WeatherCell', 'SMDCell']] = loc_id[['WeatherCell', 'SMDCell']].astype(int)

        # --------------------------------------------------------------------------
        if self.location_identifiers is None:
            self.read_location_identifiers(update=update, verbose=verbose)
        stanox_sheet = self.location_identifiers['STANOX']
        stanox_sheet = stanox_sheet[~stanox_sheet.index.duplicated(keep='first')]

        # Replace metadata coordinates data with ref coordinates if available
        temp = loc_id.join(stanox_sheet, on=['StartStanox', 'StartLocation'])
        temp = temp[temp['DEG_LAT'].notnull() & temp['DEG_LONG'].notnull()]
        loc_id.loc[temp.index, 'StartLongitude':'StartLatitude'] = temp[
            ['DEG_LONG', 'DEG_LAT']].values
        loc_id.loc[temp.index, 'ApproximateStartLocation'] = False
        loc_id.loc[loc_id.index.difference(temp.index), 'ApproximateStartLocation'] = True

        temp = loc_id.join(stanox_sheet, on=['EndStanox', 'EndLocation'])
        temp = temp[temp['DEG_LAT'].notnull() & temp['DEG_LONG'].notnull()]
        loc_id.loc[temp.index, 'EndLongitude':'EndLatitude'] = temp[['DEG_LONG', 'DEG_LAT']].values
        loc_id.loc[temp.index, 'ApproximateEndLocation'] = False
        loc_id.loc[loc_id.index.difference(temp.index), 'ApproximateEndLocation'] = True

        # Get reference metadata from Railway-codes
        stn = Stations()
        stn_loc = stn.fetch_locations()[stn.KEY_TO_STN]
        stn_loc = stn_loc[['Station', 'Degrees Longitude', 'Degrees Latitude']].dropna()
        stn_loc.drop_duplicates(subset=['Station'], inplace=True)
        stn_loc.set_index('Station', inplace=True)

        temp = loc_id.join(stn_loc, on='StartLocation')
        na_i = temp['Degrees Longitude'].notnull() & temp['Degrees Latitude'].notnull()
        loc_id.loc[na_i, 'StartLongitude':'StartLatitude'] = \
            temp.loc[na_i, ['Degrees Longitude', 'Degrees Latitude']].values
        loc_id.loc[na_i, 'ApproximateStartLocation'] = False
        loc_id.loc[~na_i, 'ApproximateStartLocation'] = True

        temp = loc_id.join(stn_loc, on='EndLocation')
        na_i = temp['Degrees Longitude'].notnull() & temp['Degrees Latitude'].notnull()
        loc_id.loc[na_i, 'EndLongitude':'EndLatitude'] = \
            temp.loc[na_i, ['Degrees Longitude', 'Degrees Latitude']].values
        loc_id.loc[na_i, 'ApproximateEndLocation'] = False
        loc_id.loc[~na_i, 'ApproximateEndLocation'] = True

        loc_id.ApproximateLocation = (
                loc_id.ApproximateStartLocation | loc_id.ApproximateEndLocation)

        # Finalise
        selected_cols = [
            'STANME', 'LINE_DESCRIPTION', 'SHAPE_LENG', 'EASTING', 'NORTHING', 'GRIDREF',
            'LOOKUP_NAME_Raw']

        start_cols = ['Start' + x.replace('_', ' ').title().replace(' ', '') for x in selected_cols]
        loc_id = loc_id.join(stanox_sheet[selected_cols], on=['StartStanox', 'StartLocation'])
        loc_id.rename(columns=dict(zip(selected_cols, start_cols)), inplace=True)

        end_cols = ['End' + x.replace('_', ' ').title().replace(' ', '') for x in selected_cols]
        loc_id = loc_id.join(stanox_sheet[selected_cols], on=['EndStanox', 'EndLocation'])
        loc_id.rename(columns=dict(zip(selected_cols, end_cols)), inplace=True)

        text_cols = [x for x in start_cols + end_cols if loc_id[x].dtype.name == 'object']
        loc_id.loc[:, text_cols] = loc_id[text_cols].fillna('')

        loc_id.set_index('StanoxSectionId', inplace=True)

        self.dump_preprocessed_data(
            loc_id, table_name=self.LOCID_TABLE_NAME, verbose=verbose, pkey=[], **kwargs)
        self.db_instance.null_text_to_empty_string(
            table_name=self.LOCID_TABLE_NAME, schema_name=self.SCHEMA_NAME)

        return loc_id

    def read_location_identifiers_plus(self, **kwargs):
        """
        Read (supplementary) data of location codes.

        **Examples**::

            >>> from src.preprocessor import Schedule8IncidentReports
            >>> s8wir = Schedule8IncidentReports()
            >>> s8wir.read_location_identifiers_plus()
            >>> s8wir.location_identifiers_.shape
            (10522, 30)
        """

        self.read_data(
            table_name='location_identifiers_', true_values=['t', 'true'],
            false_values=['f', 'false'], low_memory=False, index_col='StanoxSectionId', **kwargs)

    # == Incidents data ============================================================================

    @staticmethod
    def _tidy_alt_codes(rc_code_dict, raw_loc):
        """
        Get rid of duplicated records.
        """

        dat = raw_loc.copy()
        dat.columns = [x.replace('_Raw', '') for x in dat.columns]

        temp = dat.join(rc_code_dict, on='StartLocation')
        temp = temp[temp.Location.notnull()]
        dat.loc[temp.index, 'StartLocation'] = temp.Location.values

        temp = dat.join(rc_code_dict, on='EndLocation')
        temp = temp[temp.Location.notnull()]
        dat.loc[temp.index, 'EndLocation'] = temp.Location.values

        return dat

    def _cleanse_stanox_section(self, dat):
        """
        Cleanse the column that contains location information in the data of incident records.

        :param dat: Data of incident records.
        :type dat: pandas.DataFrame
        :return: Data of incident records with cleansed information of location coded by STANOX.
        :rtype: pandas.DataFrame
        """

        dat_ = dat.copy()

        col_name, sep = 'StanoxSection', ' : '

        old_column_name = col_name + '_Raw'
        dat_.rename(columns={col_name: old_column_name}, inplace=True)

        # Raw data of the start and end locations
        start_end_raw = dat_[old_column_name].str.split(sep, expand=True)
        start_end_raw.columns = ['StartLocation_Raw', 'EndLocation_Raw']
        start_end_raw['EndLocation_Raw'] = start_end_raw['EndLocation_Raw'].fillna(
            start_end_raw['StartLocation_Raw'])
        dat_ = dat_.join(start_end_raw)

        # In errata_tiploc, {'CLAPS47': 'CLPHS47'} might be problematic.
        errata = load_data(cd(self.RC_DATA_DIR, "metex_location_name_errata.json"))
        errata_stanox, errata_tiploc, errata_stanme = errata.values()
        start_end = start_end_raw.replace(errata_stanox)
        start_end = start_end.replace(errata_tiploc)
        start_end = start_end.replace(errata_stanme)

        lid = LocationIdentifiers()

        #
        stanox_dict = lid.make_xref_dict('STANOX')
        start_end = self._tidy_alt_codes(stanox_dict, start_end)
        #
        stanme_dict = lid.make_xref_dict('STANME')
        start_end = self._tidy_alt_codes(stanme_dict, start_end)
        #
        tiploc_dict = lid.make_xref_dict('TIPLOC')
        start_end = self._tidy_alt_codes(tiploc_dict, start_end)

        #
        errata_regex = fetch_location_names_errata(regex=True)
        for col in start_end.columns:
            start_end.replace({col: errata_regex}, regex=True, inplace=True)
        start_end.replace(errata_regex, regex=True, inplace=True)

        errata_plain = fetch_location_names_errata()
        start_end.replace(errata_plain, inplace=True)

        # Create new StanoxSection column
        dat_[col_name] = start_end['StartLocation'] + sep + start_end['EndLocation']
        # Index of single-point locations
        idx = (
            start_end['StartLocation'][start_end['StartLocation'] == start_end['EndLocation']].index)
        dat_.loc[idx, col_name] = start_end.loc[idx, 'StartLocation']

        # Resort column order
        temp = itertools.groupby(list(dat_.columns), lambda x: x == old_column_name)
        col_names = [list(v) for k, v in temp if not k]
        add_names = [old_column_name] + col_names[1][-3:] + ['StartLocation', 'EndLocation']
        col_names = col_names[0] + add_names + col_names[1][:-3]

        cleansed_data = dat_.join(start_end)[col_names]

        return cleansed_data

    def _cleanse_geographical_coordinates(self, dat):
        """
        Look up geographical coordinates for each incident location.

        :param dat: Data of incident records.
        :type dat: pandas.DataFrame
        :return: Data of incident records with cleansed geographical coordinates.
        :rtype: pandas.DataFrame
        """

        dat_ = dat.copy()

        # Find geographical coordinates for each incident location
        if self.location_identifiers is None:
            self.read_location_identifiers()
        ref_loc1_ = self.location_identifiers['TIPLOC_LocationsLyr']
        raw_coords_col_names = ['EASTING', 'NORTHING', 'DEG_LONG', 'DEG_LAT']
        col_names = ['Easting', 'Northing', 'Longitude', 'Latitude']
        ref_loc1_.rename(columns=dict(zip(raw_coords_col_names, col_names)), inplace=True)
        ref_loc1 = ref_loc1_.reset_index().drop_duplicates(['LOOKUP_NAME']).set_index('LOOKUP_NAME')

        dat_ = dat_.join(ref_loc1[col_names], on='StartLocation')
        dat_.rename(columns=dict(zip(col_names, ['Start' + c for c in col_names])), inplace=True)
        dat_ = dat_.join(ref_loc1[col_names], on='EndLocation')
        dat_.rename(columns=dict(zip(col_names, ['End' + c for c in col_names])), inplace=True)

        # == Reference data 1 ======================================================================
        if self.location_identifiers_ is None:
            self.read_location_identifiers_plus()
        loc_dat_plus = self.location_identifiers_.copy()
        start_locs = loc_dat_plus[['StartLocation', 'StartLongitude', 'StartLatitude']]
        start_locs.columns = [c.replace('Start', '') for c in start_locs.columns]
        end_locs = loc_dat_plus[['EndLocation', 'EndLongitude', 'EndLatitude']]
        end_locs.columns = [c.replace('End', '') for c in end_locs.columns]
        loc_dat_plus = pd.concat([start_locs, end_locs], ignore_index=True)
        loc_dat_plus = loc_dat_plus.drop_duplicates('Location').set_index('Location')

        # Fill in NA coordinates
        temp = dat_[dat_['StartEasting'].isna() | dat_['StartLongitude'].isna()]
        temp = temp.join(loc_dat_plus, on='StartLocation')
        idx = temp.index
        dat_.loc[idx, ['StartLongitude', 'StartLatitude']] = temp[['Longitude', 'Latitude']].values
        dat_.loc[idx, ['StartEasting', 'StartNorthing']] = wgs84_to_osgb36(
            temp['Longitude'], temp['Latitude'], as_array=True)

        temp = dat_[dat_['EndEasting'].isna() | dat_['EndLongitude'].isna()]
        temp = temp.join(loc_dat_plus, on='EndLocation')
        dat_.loc[temp.index, ['EndLongitude', 'EndLatitude']] = temp[
            ['Longitude', 'Latitude']].values

        # Dalston Junction (East London Line)     --> Dalston Junction [-0.0751, 51.5461]
        # Ashford West Junction (CTRL)            --> Ashford West Junction [0.86601557, 51.146927]
        # Southfleet Junction                     --> ? [0.34262910, 51.419354]
        # Channel Tunnel Eurotunnel Boundary CTRL --> ? [1.1310482, 51.094808]
        na_loc = [
            'Dalston Junction (East London Line)',
            'Ashford West Junction (CTRL)',
            'Southfleet Junction',
            'Channel Tunnel Eurotunnel Boundary CTRL',
        ]
        na_loc_longlat = [
            [-0.0751, 51.5461],
            [0.86601557, 51.146927],
            [0.34262910, 51.419354],
            [1.1310482, 51.094808],
        ]
        for x_, longlat in zip(na_loc, na_loc_longlat):
            if x_ in set(temp.EndLocation):
                idx = temp[temp.EndLocation == x_].index
                temp.loc[idx, 'EndLongitude':'Latitude'] = longlat * 2
                dat_.loc[idx, ['EndLongitude', 'EndLatitude']] = longlat

        dat_.loc[temp.index, ['EndEasting', 'EndNorthing']] = wgs84_to_osgb36(
            temp.Longitude.values, temp.Latitude.values, as_array=True)

        dat_.index = range(len(dat_))

        # == Reference data 2 ======================================================================
        stn_loc = Stations()
        ref_loc2 = stn_loc.fetch_locations()[stn_loc.KEY_TO_STN]
        ref_loc2 = ref_loc2[['Station', 'Degrees Longitude', 'Degrees Latitude']]
        ref_loc2 = ref_loc2.dropna().drop_duplicates('Station')
        ref_loc2.columns = [x.replace('Degrees ', '') for x in ref_loc2.columns]
        ref_loc2.set_index('Station', inplace=True)

        temp = dat_.join(ref_loc2, on='StartLocation').drop_duplicates()
        temp_start = temp[temp.Longitude.notnull() & temp.Latitude.notnull()]
        idx = temp_start.index
        dat_.loc[idx, ['StartLongitude', 'StartLatitude']] = temp_start[
            ['Longitude', 'Latitude']].values

        temp = dat_.join(ref_loc2, on='EndLocation')
        temp_end = temp[temp.Longitude.notnull() & temp.Latitude.notnull()]
        idx = temp_end.index
        dat_.loc[idx, 'EndLongitude':'EndLatitude'] = temp_end[['Longitude', 'Latitude']].values

        # Let (Longitude, Latitude) be almost equivalent to (Easting, Northing)
        dat_.loc[:, ['StartLongitude', 'StartLatitude']] = osgb36_to_wgs84(
            dat_['StartEasting'], dat_['StartNorthing'], as_array=True)
        dat_.loc[:, ['EndLongitude', 'EndLatitude']] = osgb36_to_wgs84(
            dat_['EndEasting'], dat_['EndNorthing'], as_array=True)

        # Convert coordinates to shapely.geometry.Point
        dat_.loc[:, 'StartXY'] = [
            shapely.geometry.Point(x, y).wkt for x, y in dat_[
                ['StartEasting', 'StartNorthing']].values]
        dat_.loc[:, 'EndXY'] = [
            shapely.geometry.Point(x, y).wkt for x, y in dat_[['EndEasting', 'EndNorthing']].values]
        dat_.loc[:, 'StartLongLat'] = [
            shapely.geometry.Point(lon, lat).wkt
            for lon, lat in dat_[['StartLongitude', 'StartLatitude']].values]
        dat_.loc[:, 'EndLongLat'] = [
            shapely.geometry.Point(lon, lat).wkt
            for lon, lat in dat_[['EndLongitude', 'EndLatitude']].values]

        return dat_

    def _schedule8_weather_incidents(self, route_name=None, weather_category=None, verbose=False,
                                     **kwargs):
        """
        Get data of Schedule 8 weather incidents from spreadsheet file.

        :param route_name: Name of a Network Rail's Route;
            when ``route_name=None`` (default), it takes all available Routes.
        :type route_name: str | None
        :param weather_category: Weather category;
            when ``weather_category=None`` (default), it takes all available weather categories.
        :type weather_category: str | None
        :param verbose: Whether to print relevant information to the console; defaults to ``False``.
        :type verbose: bool | int
        :param kwargs: [Optional] parameters of the method
            `pyhelpers.dbms.PostgreSQL.import_data()`_;
            see also
            :py:meth:`Handler.dump_preprocessed_data()<src.utils.Handler.dump_preprocessed_data>`.
        :return: Data of Schedule 8 weather incidents.
        :rtype: pandas.DataFrame | None

        .. _`pyhelpers.dbms.PostgreSQL.import_data()`:
            https://pyhelpers.readthedocs.io/en/latest/_generated/
            pyhelpers.dbms.PostgreSQL.import_data.html

        **Examples**::

            >>> from src.preprocessor.schedule8 import Schedule8IncidentReports
            >>> s8wir = Schedule8IncidentReports()
            >>> schedule8_weather_incidents = s8wir._schedule8_weather_incidents()
            >>> schedule8_weather_incidents.shape
            (178955, 36)
        """

        # Load data from the raw file
        xlsx_pathname = self._cdd(self.FILENAME_1 + ".xlsx")
        temp_csv_pathname = xlsx_to_csv(xlsx_pathname, sheet_name='1')
        raw_data = pd.read_csv(
            temp_csv_pathname, parse_dates=['StartDate', 'EndDate'], low_memory=False,
            keep_default_na=False)

        new_column_names = {
            'StartDate': 'StartDateTime',
            'EndDate': 'EndDateTime',
            'stanoxSection': 'StanoxSection',
            'imdm': 'IMDM',
            'weather_category': 'WeatherCategoryCode',
            'weather_category.1': 'weather_category',
            'Reason': 'IncidentReason',
            # 'Minutes': 'DelayMinutes',
            'Description': 'IncidentReasonDescription',
            'Category': 'IncidentCategory',
            'CategoryDescription': 'IncidentCategoryDescription'
        }

        raw_dat = raw_data.rename(columns=new_column_names)

        # Add information about incident reason
        dag = DelayAttributionGlossary()

        incident_reason = dag._read_dag_data_from_db("Incident Reason")
        incident_reason.columns = [x.replace('_', '') for x in incident_reason.columns]
        temp = pd.merge(raw_dat, incident_reason, on='IncidentReason', suffixes=('', '_add'))
        dat = temp.drop([x for x in temp.columns if '_add' in x], axis=1)

        # Cleanse the location data
        dat = self._cleanse_stanox_section(dat=dat)

        # Look up geographical coordinates for each incident location
        cleansed_data = self._cleanse_geographical_coordinates(dat=dat)

        # Retain data for specific Route and Weather category
        data = get_subset(cleansed_data, route_name=route_name, weather_category=weather_category)

        data.set_index('IncidentNumber', inplace=True)

        self.dump_preprocessed_data(
            data, table_name=self.FILENAME_1, verbose=verbose, pkey=[], **kwargs)
        self.db_instance.null_text_to_empty_string(self.FILENAME_1, schema_name=self.SCHEMA_NAME)

        return data

    def read_schedule8_weather_incidents(self, route_name=None, weather_category=None,
                                         ret_data=False, **kwargs):
        """
        Read data of Schedule 8 weather incidents.

        :param route_name: Name of a Network Rail's Route;
            when ``route_name=None`` (default), it takes all available Routes.
        :type route_name: str | None
        :param weather_category: Weather category;
            when ``weather_category=None`` (default), it takes all available weather categories.
        :type weather_category: str | None
        :param ret_data: Whether to return the data; defaults to ``False``.
        :type ret_data: bool
        :param kwargs: [Optional] parameters of the method
            `pyhelpers.dbms.PostgreSQL.read_sql_query()`_;
            see also :py:meth:`Handler.read_data()<src.utils.dbms.Handler.read_data>`.
        :return: Data of Schedule 8 weather incidents (if ``ret_data=True``).
        :rtype: pandas.DataFrame

        .. _`pyhelpers.dbms.PostgreSQL.read_sql_query()`:
            https://pyhelpers.readthedocs.io/en/latest/_generated/
            pyhelpers.dbms.PostgreSQL.read_sql_query.html

        **Examples**::

            >>> from src.preprocessor.schedule8 import Schedule8IncidentReports
            >>> s8wir = Schedule8IncidentReports()
            >>> s8wir.read_schedule8_weather_incidents()
            >>> s8wir.schedule8_weather_incidents.shape
            (178955, 36)
        """

        kwargs.update({'low_memory': False})
        self.read_data(
            table_name=self.FILENAME_1, ivar_name='schedule8_weather_incidents',
            index_col='IncidentNumber', **kwargs)

        self.schedule8_weather_incidents = get_subset(
            self.schedule8_weather_incidents, route_name=route_name,
            weather_category=weather_category)

        geom_cols = ['StartXY', 'EndXY', 'StartLongLat', 'EndLongLat']
        self.schedule8_weather_incidents.loc[:, geom_cols] = (
            self.schedule8_weather_incidents[geom_cols].map(shapely.wkt.loads))

        if ret_data:
            return self.schedule8_weather_incidents

    def _schedule8_weather_incidents_02062006_31032014(self, route_name=None, weather_category=None,
                                                       verbose=False, **kwargs):
        """
        Read data of Schedule 8 weather incidents (02/06/2006 - 31/03/2014)
        from the Excel spreadsheet "Schedule8WeatherIncidents-02062006-31032014.xlsm".

        :param route_name: Name of a Network Rail's Route;
            when ``route_name=None`` (default), it takes all available Routes.
        :type route_name: str | None
        :param weather_category: Weather category;
            when ``weather_category=None`` (default), it takes all available weather categories.
        :type weather_category: str | None
        :param verbose: Whether to print relevant information to the console; defaults to ``False``.
        :type verbose: bool | int
        :param kwargs: [Optional] parameters of the method
            `pyhelpers.dbms.PostgreSQL.import_data()`_;
            see also
            :meth:`Handler.dump_preprocessed_data()<src.utils.Handler.dump_preprocessed_data>`.
        :return: Data of Schedule 8 weather incidents (02/06/2006 - 31/03/2014).
        :rtype: pandas.DataFrame | None

        .. _`pyhelpers.dbms.PostgreSQL.import_data()`:
            https://pyhelpers.readthedocs.io/en/latest/_generated/
            pyhelpers.dbms.PostgreSQL.import_data.html

        **Examples**::

            >>> from src.preprocessor.schedule8 import Schedule8IncidentReports
            >>> s8wir = Schedule8IncidentReports()
            >>> incidents_data = s8wir._schedule8_weather_incidents_02062006_31032014()
            >>> list(incidents_data.keys())
            ['WeatherThresholds',
             'Schedule8WeatherIncidents_02062006_31032014',
             'WeatherCategoryLookup']
            >>> incidents_data['WeatherThresholds'].shape
            (29, 4)
            >>> incidents_data['Schedule8WeatherIncidents_02062006_31032014'].shape
            (178960, 51)
            >>> incidents_data['WeatherCategoryLookup'].shape
            (9, 1)
        """

        xlsm_pathname = self._cdd(self.FILENAME_2 + ".xlsm")

        # == 'WeatherThresholds' ===================================================================
        sheet1_pathname = xlsx_to_csv(xlsm_pathname, sheet_name='1')
        thresholds = pd.read_csv(sheet1_pathname, usecols=list(range(0, 6))).dropna()
        thresholds.columns = [x.replace(' ', '') for x in thresholds.columns]
        thresholds.WeatherHazard = thresholds.WeatherHazard.str.strip().str.upper()
        thresholds.Threshold = thresholds.Threshold.astype(int)

        thresholds.set_index(['WeatherType', 'WeatherHazard'], inplace=True)

        if self.db_instance is None:
            self.db_instance = WxRailIncidentsPred(verbose=False)

        table1_name = 'weather_thresholds'
        self.dump_preprocessed_data(
            data=thresholds, table_name=table1_name, verbose=verbose, pkey=[], **kwargs)
        self.db_instance.null_text_to_empty_string(table1_name, schema_name=self.SCHEMA_NAME)

        os.remove(sheet1_pathname)

        # == 'Data' ================================================================================
        sheet2_pathname = xlsx_to_csv(xlsm_pathname, sheet_name='2')
        raw_data = pd.read_csv(
            sheet2_pathname, parse_dates=['StartDate', 'EndDate'], low_memory=False,
            keep_default_na=False)

        rename_columns = {
            'StartDate': 'StartDateTime',
            'EndDate': 'EndDateTime',
            'Year': 'FinancialYear',
            'stanoxSection': 'StanoxSection',
            'imdm': 'IMDM',
            'Reason': 'IncidentReason',
            'Minutes': 'DelayMinutes',
            'Cost': 'DelayCost',
            'CategoryDescription': 'IncidentCategoryDescription',
        }
        raw_dat = raw_data.rename(columns=rename_columns)

        hazard_cols = [x for x in enumerate(raw_dat.columns) if 'Weather Hazard' in x[1]]
        obs_cols = [(i - 1, re.search(r'(?<= \()\w+', x).group().upper()) for i, x in hazard_cols]
        hazard_cols = [(i + 1, x + '_WeatherHazard') for i, x in obs_cols]
        for i, x in obs_cols + hazard_cols:
            raw_dat.rename(columns={raw_dat.columns[i]: x}, inplace=True)

        # data.weather_category = data.weather_category.replace('Heat Speed/Buckle', 'Heat')
        dag = DelayAttributionGlossary()

        incident_reason = dag._read_dag_data_from_db("Incident Reason")
        incident_reason.columns = [x.replace('_', '') for x in incident_reason.columns]
        incident_reason.set_index('IncidentReason', inplace=True)

        raw_dat = raw_dat.join(incident_reason, on='IncidentReason', rsuffix='_add')
        dat = raw_dat.drop([x for x in raw_dat.columns if '_add' in x], axis=1)

        # Cleanse the location data
        dat = self._cleanse_stanox_section(dat=dat)

        # Look up geographical coordinates for each incident location
        cleansed_data = self._cleanse_geographical_coordinates(dat=dat)

        # Retain data for specific Route and Weather category
        data = get_subset(cleansed_data, route_name=route_name, weather_category=weather_category)

        data.set_index('IncidentNumber', inplace=True)

        os.remove(sheet2_pathname)

        self.dump_preprocessed_data(
            data, table_name=self.FILENAME_2, verbose=verbose, pkey=[], **kwargs)
        self.db_instance.null_text_to_empty_string(self.FILENAME_2, schema_name=self.SCHEMA_NAME)

        # == Weather category lookup ===============================================================
        sheet3_pathname = xlsx_to_csv(xlsm_pathname, sheet_name='3')
        category_lookup = pd.read_csv(sheet3_pathname)
        category_lookup.columns = ['WeatherCategoryCode', 'weather_category']

        category_lookup.set_index('WeatherCategoryCode', inplace=True)

        table3_name = 'weather_category_lookup'
        self.dump_preprocessed_data(
            category_lookup, table_name=table3_name, verbose=verbose, pkey=[], **kwargs)

        os.remove(sheet3_pathname)

        # Make a dictionary
        sheet_names = [table1_name, self.FILENAME_2, table3_name]
        s8weather_incidents = dict(zip(sheet_names, [thresholds, data, category_lookup]))

        return s8weather_incidents

    def read_schedule8_weather_incidents_02062006_31032014(self, route_name=None,
                                                           weather_category=None, update=False,
                                                           verbose=False, ret_data=False,
                                                           **kwargs):
        """
        Read data of Schedule 8 weather incidents (02/06/2006 - 31/03/2014).

        :param route_name: Name of a Network Rail's Route;
            when ``route_name=None`` (default), it takes all available Routes.
        :type route_name: str | None
        :param weather_category: Weather category;
            when ``weather_category=None`` (default), it takes all available weather categories.
        :type weather_category: str | None
        :param update: Whether read the data from the original data file; defaults to ``False``.
        :type update: bool
        :param verbose: Whether to print relevant information to the console; defaults to ``False``.
        :type verbose: bool | int
        :param ret_data: Whether to return the data; defaults to ``False``.
        :type ret_data: bool
        :param kwargs: [Optional] parameters of the method
            `pyhelpers.dbms.PostgreSQL.read_sql_query()`_
        :return: Data of Schedule 8 weather incidents (02/06/2006 - 31/03/2014)
            (if ``ret_data=True``).
        :rtype: dict

        .. _`pyhelpers.dbms.PostgreSQL.read_sql_query()`:
            https://pyhelpers.readthedocs.io/en/latest/_generated/
            pyhelpers.dbms.PostgreSQL.read_sql_query.html

        **Examples**::

            >>> from src.preprocessor.schedule8 import Schedule8IncidentReports
            >>> s8wir = Schedule8IncidentReports()
            >>> s8wir.read_schedule8_weather_incidents_02062006_31032014(update=True, verbose=True)
            >>> dat_ = s8wir.schedule8_weather_incidents_02062006_31032014
            >>> list(dat_.keys())
            ['weather_thresholds',
             'schedule8_weather_incidents_02062006_31032014',
             'weather_category_lookup']
            >>> dat_['weather_thresholds'].shape
            (29, 4)
            >>> dat_['schedule8_weather_incidents_02062006_31032014'].shape
            (178960, 51)
            >>> dat_['weather_category_lookup'].shape
            (9, 1)
        """

        table_names = ['weather_thresholds', self.FILENAME_2, 'weather_category_lookup']

        if self.db_instance is None:
            self.db_instance = WxRailIncidentsPred(verbose=verbose)

        if (all(self.db_instance.table_exists(x, self.SCHEMA_NAME) for x in table_names) and
                not update):
            data_list = []

            for table_name in table_names:
                sql_query = self.specify_sql_query(table_name)

                if table_name == 'weather_thresholds':
                    index_col = ['WeatherType', 'WeatherHazard']
                elif table_name == 'weather_category_lookup':
                    index_col = 'WeatherCategoryCode'
                else:
                    index_col = 'IncidentNumber'

                dat = self.db_instance.read_sql_query(sql_query, index_col=index_col, **kwargs)

                text_cols = [x for x in dat.columns if dat[x].dtype.name == 'object']
                if bool(text_cols):
                    dat.loc[:, text_cols] = dat[text_cols].fillna('')

                if table_name == self.FILENAME_2:
                    geom_cols = ['StartXY', 'EndXY', 'StartLongLat', 'EndLongLat']
                    dat.loc[:, geom_cols] = dat[geom_cols].map(shapely.wkt.loads)

                    dat = get_subset(dat, route_name=route_name, weather_category=weather_category)

                data_list.append(dat.sort_index())

            data = dict(zip(table_names, data_list))

        else:
            data = getattr(self, '_schedule8_weather_incidents_02062006_31032014')(verbose=verbose)

        self.schedule8_weather_incidents_02062006_31032014 = data

        if ret_data:
            return data
