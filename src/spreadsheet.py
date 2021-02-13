"""
Schedule 8 Incidents.
"""

import itertools
import os
import re

import numpy as np
import pandas as pd
import shapely.geometry
from pyhelpers.dir import cd
from pyhelpers.geom import osgb36_to_wgs84, wgs84_to_osgb36
from pyhelpers.store import load_json, load_pickle, save_pickle
from pyhelpers.text import find_similar_str
from pyrcs.line_data import LocationIdentifiers
from pyrcs.other_assets import Stations
from pyrcs.utils import fetch_loc_names_repl_dict

from misc.dag import DelayAttributionGlossary
from mssqlserver.metex import read_metex_table
from utils import cdd_incidents, cdd_railway_codes, get_subset, make_filename


class Schedule8IncidentsSpreadsheet:

    def __init__(self):
        self.Name = 'Schedule 8 Incidents'

        self.DataDir = os.path.relpath(cdd_incidents("spreadsheets"))

        self.DataFilename1 = "Schedule8WeatherIncidents"
        self.DataFilename2 = "Schedule8WeatherIncidents-02062006-31032014"

    def cdd(self, *sub_dir, mkdir=False):
        """
        Change directory to "data\\incidents\\spreadsheets" and sub-directories (or files).

        :param sub_dir: sub-directory name(s) or filename(s)
        :type sub_dir: str
        :param mkdir: whether to create a directory, defaults to ``False``
        :type mkdir: bool
        :return: "data\\incidents\\spreadsheets"
        :rtype: str

        **Test**::

            >>> import os
            >>> from spreadsheet import Schedule8IncidentsSpreadsheet

            >>> sis = Schedule8IncidentsSpreadsheet()

            >>> dat_dir = sis.cdd()

            >>> print(os.path.relpath(dat_dir))
            data\\incidents\\spreadsheets
        """

        path = cd(self.DataDir, *sub_dir, mkdir=mkdir)

        return path

    # == Location metadata ============================================================================

    @staticmethod
    def _pre_cleanse_location_metadata(metadata):
        """
        Pre-cleanse location metadata spreadsheet.

        :param metadata: raw data frame of location metadata
        :type metadata: pandas.DataFrame
        :return: cleansed data frame of location metadata
        :rtype: pandas.DataFrame
        """

        meta_dat = metadata.copy(deep=True)

        meta_dat.columns = [x.upper().replace(' ', '_') for x in metadata.columns]
        meta_dat.replace({'\xa0\xa0': ' '}, regex=True, inplace=True)
        meta_dat.replace({'\xa0': ''}, regex=True, inplace=True)
        meta_dat.replace({'LINE_DESCRIPTION': {re.compile('[Ss]tn'): 'Station',
                                               re.compile('[Ll]oc'): 'Location',
                                               re.compile('[Jj]n'): 'Junction',
                                               re.compile('[Ss]dg'): 'Siding',
                                               re.compile('[Ss]dgs'): 'Sidings'}}, inplace=True)
        meta_dat['LOOKUP_NAME_Raw'] = meta_dat.LOOKUP_NAME
        meta_dat.fillna({'STANOX': '', 'STANME': ''}, inplace=True)
        meta_dat.STANOX = meta_dat.STANOX.map(lambda x: '0' * (5 - len(x)) + x if x != '' else x)

        return meta_dat

    def _cleanse_location_metadata_tiploc_sheet(self, metadata, update_dict=False):
        """
        Cleanse 'TIPLOC_LocationsLyr' sheet.

        :param metadata: data frame of the 'TIPLOC_LocationsLyr' sheet in location metadata
        :type metadata: pandas.DataFrame
        :param update_dict: whether to update the location codes dictionary
        :type update_dict: bool
        :return: cleansed data frame of the 'TIPLOC_LocationsLyr' sheet in location metadata
        :rtype: pandas.DataFrame
        """

        meta_dat = self._pre_cleanse_location_metadata(metadata)

        meta_dat.TIPLOC = meta_dat.TIPLOC.fillna('').str.upper()

        ref_cols = ['STANOX', 'STANME', 'TIPLOC']
        dat = meta_dat[ref_cols + ['LOOKUP_NAME', 'DEG_LONG', 'DEG_LAT', 'LOOKUP_NAME_Raw']]

        # Rectify errors in STANOX; in errata_tiploc, {'CLAPS47': 'CLPHS47'} may be problematic.
        errata = load_json(cdd_railway_codes("metex-errata.json"))
        # noinspection PyTypeChecker
        errata_stanox, errata_tiploc, errata_stanme = [{k: v} for k, v in errata.items()]
        dat.replace(errata_stanox, inplace=True)
        dat.replace(errata_stanme, inplace=True)
        dat.replace(errata_tiploc, inplace=True)

        # Rectify known issues for the location names in the data set
        dat.replace(fetch_loc_names_repl_dict('LOOKUP_NAME'), inplace=True)
        dat.replace(fetch_loc_names_repl_dict('LOOKUP_NAME', regex=True), inplace=True)

        # Fill in missing location names
        na_name = dat[dat.LOOKUP_NAME.isnull()]
        lid = LocationIdentifiers()
        ref_dict = lid.make_loc_id_dict(ref_cols, update=update_dict)
        temp = na_name.join(ref_dict, on=ref_cols)
        temp = temp[['TIPLOC', 'Location']]
        dat.loc[na_name.index, 'LOOKUP_NAME'] = temp.apply(
            lambda x: find_similar_str(x[0], x[1]) if isinstance(x[1], list) else x[1], axis=1)

        # Rectify 'LOOKUP_NAME' according to 'TIPLOC'
        na_name = dat[dat.LOOKUP_NAME.isnull()]
        ref_dict = lid.make_loc_id_dict('TIPLOC', update=update_dict)
        temp = na_name.join(ref_dict, on='TIPLOC')
        dat.loc[na_name.index, 'LOOKUP_NAME'] = temp.Location.values

        not_na_name = dat[dat.LOOKUP_NAME.notnull()]
        temp = not_na_name.join(ref_dict, on='TIPLOC')

        def extract_one(lookup_name, ref_loc):
            if isinstance(ref_loc, list):
                n = find_similar_str(lookup_name.replace(' ', ''), ref_loc)
            elif pd.isnull(ref_loc):
                n = lookup_name
            else:
                n = ref_loc
            return n

        dat.loc[not_na_name.index, 'LOOKUP_NAME'] = temp.apply(lambda x: extract_one(x[3], x[7]),
                                                               axis=1)

        # Rectify 'STANOX'+'STANME'
        location_codes = lid.fetch_location_codes()['Location codes']
        location_codes = location_codes.drop_duplicates(['TIPLOC', 'Location']).set_index(
            ['TIPLOC', 'Location'])
        temp = dat.join(location_codes, on=['TIPLOC', 'LOOKUP_NAME'], rsuffix='_Ref').fillna('')
        dat.loc[temp.index, 'STANOX':'STANME'] = temp[['STANOX_Ref', 'STANME_Ref']].values

        # Update coordinates with reference data from RailwayCodes
        stn = Stations()
        station_data = stn.fetch_station_data()['Railway station data']
        station_data = station_data[['Station', 'Degrees Longitude', 'Degrees Latitude']].dropna()
        station_data = station_data.drop_duplicates(subset=['Station']).set_index('Station')
        temp = dat.join(station_data, on='LOOKUP_NAME')
        na_i = temp['Degrees Longitude'].notnull() & temp['Degrees Latitude'].notnull()
        dat.loc[na_i, 'DEG_LONG':'DEG_LAT'] = temp.loc[
            na_i, ['Degrees Longitude', 'Degrees Latitude']].values

        # Finalising...
        meta_dat.update(dat)
        meta_dat.dropna(subset=['LOOKUP_NAME'] + ref_cols, inplace=True)
        meta_dat.fillna('', inplace=True)
        meta_dat.sort_values(['LOOKUP_NAME', 'SHAPE_LENG'], ascending=[True, False], inplace=True)
        meta_dat.index = range(len(meta_dat))

        return meta_dat

    def get_location_metadata(self, update=False, verbose=False):
        """
        Get data of location STANOX.

        :param update: whether to check on update and proceed to update the package data,
            defaults to ``False``
        :type update: bool
        :param verbose: whether to print relevant information in console as the function runs,
            defaults to ``False``
        :type verbose: bool, int
        :return: data of location STANOX
        :rtype: dict

        **Test**::

            from spreadsheet.incidents import get_location_metadata

            update = True
            verbose = True

            location_metadata = get_location_metadata(update, verbose)
            print(location_metadata)
            # {'TIPLOC_LocationsLyr': <data>,
            #  'STANOX': <data>}
        """

        pickle_filename = "stanox-locations-data.pickle"
        path_to_pickle = cdd_railway_codes(pickle_filename)

        if os.path.isfile(path_to_pickle) and not update:
            location_metadata = load_pickle(path_to_pickle)

        else:
            try:
                workbook = pd.ExcelFile(path_to_pickle.replace(".pickle", ".xlsx"))

                # 'TIPLOC_LocationsLyr' -----------------------------------------
                tiploc_loc_lyr = workbook.parse(sheet_name='TIPLOC_LocationsLyr',
                                                parse_dates=['LASTEDITED', 'LAST_UPD_1'], dayfirst=True,
                                                converters={'STANOX': str})

                cols_with_na = ['STATUS', 'LINE_DESCRIPTION', 'QC_STATUS', 'DESCRIPTIO', 'BusinessRef']
                tiploc_loc_lyr.fillna({x: '' for x in cols_with_na}, inplace=True)

                tiploc_loc_lyr = self._cleanse_location_metadata_tiploc_sheet(
                    tiploc_loc_lyr, update_dict=update)

                # 'STANOX' -----------------------------------------------------------------------------
                stanox = workbook.parse(sheet_name='STANOX', converters={'STANOX': str, 'gridref': str})

                stanox = self._pre_cleanse_location_metadata(stanox)
                stanox.fillna({'LOOKUP_NAME_Raw': ''}, inplace=True)

                ref_cols = ['SHAPE_LENG', 'EASTING', 'NORTHING', 'GRIDREF']
                ref_data = tiploc_loc_lyr.set_index(ref_cols)
                stanox.drop_duplicates(ref_cols, inplace=True)
                temp = stanox.join(ref_data, on=ref_cols, rsuffix='_Ref').drop_duplicates(ref_cols)

                ref_cols_ok = [c for c in temp.columns if '_Ref' in c]
                stanox.loc[:, [c.replace('_Ref', '') for c in ref_cols_ok]] = temp[ref_cols_ok].values

                # Collect the above two dataframes and store them in a dictionary ---------------------
                location_metadata = dict(zip(workbook.sheet_names, [tiploc_loc_lyr, stanox]))

                workbook.close()

                save_pickle(location_metadata, path_to_pickle, verbose=verbose)

            except Exception as e:
                print("Failed to fetch \"STANOX locations data\" from {}. {}".format(
                    os.path.dirname(path_to_pickle), e))
                location_metadata = None

        return location_metadata

    # == Location metadata from NR_METEX database =====================================================

    @staticmethod
    def get_metex_location(update=False, verbose=False):
        """
        Get location data with LocationId and WeatherCell from NR_METEX_* database.

        :param update: whether to check on update and proceed to update the package data,
            defaults to ``False``
        :type update: bool
        :param verbose: whether to print relevant information in console as the function runs,
            defaults to ``False``
        :type verbose: bool, int
        :return: location data
        :rtype: pandas.DataFrame, None

        **Test**::

            from spreadsheet.incidents import get_metex_location

            update = True
            verbose = True

            metex_location = get_metex_location(update, verbose)
            print(metex_location)
        """

        pickle_filename = "metex-location.pickle"
        path_to_pickle = cdd_railway_codes(pickle_filename)

        if os.path.isfile(path_to_pickle) and not update:
            metex_location = load_pickle(path_to_pickle)

        else:
            try:
                metex_location = read_metex_table('Location')
                metex_location.rename(columns={'Id': 'LocationId'}, inplace=True)
                metex_location.set_index('LocationId', inplace=True)
                metex_location.WeatherCell = metex_location.WeatherCell.map(
                    lambda x: '' if pd.isna(x) else str(int(x)))
                # # Correct a known error
                # location.loc['610096', 'StartLongitude':'EndLatitude'] = \
                #     [-0.0751, 51.5461, -0.0751, 51.5461]
                save_pickle(metex_location, path_to_pickle, verbose=verbose)

            except Exception as e:
                print("Failed to get NR_METEX location data. {}".format(e))
                metex_location = pd.DataFrame()

        return metex_location

    @staticmethod
    def get_metex_stanox_location(update=False, verbose=False):
        """
        Get data of STANOX location from NR_METEX_* database.

        :param update: whether to check on update and proceed to update the package data,
            defaults to ``False``
        :type update: bool
        :param verbose: whether to print relevant information in console as the function runs,
            defaults to ``False``
        :type verbose: bool, int
        :return: data of STANOX location
        :rtype: pandas.DataFrame, None

        **Test**::

            from spreadsheet.incidents import get_metex_stanox_location

            update = True
            verbose = True

            metex_stanox_location = get_metex_stanox_location(update, verbose)
            print(metex_stanox_location)
        """

        pickle_filename = "metex-stanox-location.pickle"
        path_to_pickle = cdd_railway_codes(pickle_filename)

        if os.path.isfile(path_to_pickle) and not update:
            metex_stanox_location = load_pickle(path_to_pickle)

        else:
            try:
                metex_stanox_location = read_metex_table('StanoxLocation')

                # Cleanse "stanox_location"
                errata = load_json(cdd_railway_codes("metex-errata.json"))
                # In errata_tiploc, {'CLAPS47': 'CLPHS47'} might be problematic.
                errata_stanox, errata_tiploc, errata_stanme = errata.values()
                # Note that in errata_tiploc, {'CLAPS47': 'CLPHS47'} might be problematic.
                metex_stanox_location.replace({'Stanox': errata_stanox}, inplace=True)
                metex_stanox_location.replace({'Description': errata_tiploc}, inplace=True)
                metex_stanox_location.replace({'Name': errata_stanme}, inplace=True)

                lid = LocationIdentifiers()
                # Get reference data from the Railway Codes website
                rc_codes = lid.fetch_location_codes()['Location codes']
                rc_codes = rc_codes[['Location', 'TIPLOC', 'STANME', 'STANOX']].drop_duplicates()

                # Fill in NA 'Description's (i.e. Location names)
                na_desc = metex_stanox_location[metex_stanox_location.Description.isnull()]
                temp = na_desc.join(rc_codes.set_index('STANOX'), on='Stanox')
                metex_stanox_location.loc[na_desc.index, 'Description':'Name'] = temp[
                    ['Location', 'STANME']].values

                # Fill in NA 'Name's (i.e. STANME)
                na_name = metex_stanox_location[metex_stanox_location.Name.isnull()]
                # Some 'Description's are recorded by 'TIPLOC' instead
                rc_tiploc_dict = lid.make_loc_id_dict('TIPLOC')
                temp = na_name.join(rc_tiploc_dict, on='Description')
                temp = temp.join(rc_codes.set_index(['STANOX', 'TIPLOC', 'Location']),
                                 on=['Stanox', 'Description', 'Location'])
                temp = temp[temp.Location.notnull() & temp.STANME.notnull()]
                metex_stanox_location.loc[temp.index, 'Description':'Name'] = temp[
                    ['Location', 'STANME']].values
                # Still, there are some NA 'Name's remaining
                # due to incorrect spelling of 'TIPLOC' ('Description')
                na_name = metex_stanox_location[metex_stanox_location.Name.isnull()]
                temp = na_name.join(rc_codes.set_index('STANOX'), on='Stanox')
                metex_stanox_location.loc[temp.index, 'Description':'Name'] = temp[
                    ['Location', 'STANME']].values

                # Apply manually-created dictionaries
                loc_name_replacement_dict = fetch_loc_names_repl_dict('Description')
                metex_stanox_location.replace(loc_name_replacement_dict, inplace=True)
                loc_name_regexp_replacement_dict = fetch_loc_names_repl_dict('Description', regex=True)
                metex_stanox_location.replace(loc_name_regexp_replacement_dict, inplace=True)

                # Check if 'Description' has STANOX code
                # instead of location name using STANOX-dictionary
                rc_stanox_dict = lid.make_loc_id_dict('STANOX')
                temp = metex_stanox_location.join(rc_stanox_dict, on='Description')
                valid_loc = temp[temp.Location.notnull()][['Description', 'Name', 'Location']]
                if not valid_loc.empty:
                    metex_stanox_location.loc[valid_loc.index, 'Description'] = valid_loc.apply(
                        lambda x: find_similar_str(x[1], x[2]) if isinstance(x[2], list) else x[2],
                        axis=1)

                # Check if 'Description' has TIPLOC code
                # instead of location name using STANOX-TIPLOC-dictionary
                rc_stanox_tiploc_dict = lid.make_loc_id_dict(['STANOX', 'TIPLOC'])
                temp = metex_stanox_location.join(rc_stanox_tiploc_dict, on=['Stanox', 'Description'])
                valid_loc = temp[temp.Location.notnull()][['Description', 'Name', 'Location']]
                if not valid_loc.empty:
                    metex_stanox_location.loc[valid_loc.index, 'Description'] = valid_loc.apply(
                        lambda x: find_similar_str(x[1], x[2]) if isinstance(x[2], list) else x[2],
                        axis=1)

                # Check if 'Description' has STANME code instead of location name
                # using STANOX-STANME-dictionary

                rc_stanox_stanme_dict = lid.make_loc_id_dict(['STANOX', 'STANME'])
                temp = metex_stanox_location.join(rc_stanox_stanme_dict, on=['Stanox', 'Description'])
                valid_loc = temp[temp.Location.notnull()][['Description', 'Name', 'Location']]
                if not valid_loc.empty:
                    metex_stanox_location.loc[valid_loc.index, 'Description'] = valid_loc.apply(
                        lambda x: find_similar_str(x[1], x[2]) if isinstance(x[2], list) else x[2],
                        axis=1)

                # Finalise cleansing 'Description' (i.e. location names)
                temp = metex_stanox_location.join(rc_stanox_dict, on='Stanox')
                temp = temp[['Description', 'Name', 'Location']]
                metex_stanox_location.Description = temp.apply(
                    lambda x: find_similar_str(x[0], x[2]) if isinstance(x[2], list) else x[2], axis=1)

                # Cleanse 'Name' (i.e. STANME)
                metex_stanox_location.Name = metex_stanox_location.Name.str.upper()

                loc_stanme_dict = rc_codes.groupby(['STANOX', 'Location']).agg({'STANME': list})
                loc_stanme_dict.STANME = loc_stanme_dict.STANME.map(
                    lambda x: x[0] if len(x) == 1 else x)
                temp = metex_stanox_location.join(loc_stanme_dict, on=['Stanox', 'Description'])
                metex_stanox_location.Name = temp.apply(
                    lambda x: find_similar_str(x[2], x[6]) if isinstance(x[6], list) else x[6], axis=1)

                # Below is not available in any reference data
                metex_stanox_location.loc[298, 'Stanox':'Name'] = ['03330', 'Inverkeithing PPM Point',
                                                                   '']

                # Cleanse remaining NA values in 'ELR', 'Yards' and 'LocationId'
                metex_stanox_location.ELR.fillna('', inplace=True)
                metex_stanox_location.Yards = metex_stanox_location.Yards.map(
                    lambda x: '' if pd.isna(x) else str(int(x)))

                # Clear duplicates by 'STANOX'
                temp = metex_stanox_location[
                    metex_stanox_location.duplicated(subset=['Stanox'], keep=False)]
                # temp.sort_values(['Description', 'LocationId'], ascending=[True, False], inplace=True)
                metex_stanox_location.drop(index=temp[temp.LocationId.isna()].index, inplace=True)

                metex_stanox_location.LocationId = metex_stanox_location.LocationId.map(
                    lambda x: '' if pd.isna(x) else str(int(x)))

                # Finish
                metex_stanox_location.index = range(len(metex_stanox_location))

                save_pickle(metex_stanox_location, path_to_pickle, verbose=verbose)

            except Exception as e:
                print("Failed to get NR_METEX STANOX location data. {}.".format(e))
                metex_stanox_location = None

        return metex_stanox_location

    def get_metex_stanox_section(self, update=False, verbose=False):
        """
        Get data of STANOX section from NR_METEX_* database.

        :param update: whether to check on update and proceed to update the package data,
            defaults to ``False``
        :type update: bool
        :param verbose: whether to print relevant information in console as the function runs,
            defaults to ``False``
        :type verbose: bool, int
        :return: data of STANOX location
        :rtype: pandas.DataFrame, None

        **Test**::

            from spreadsheet.incidents import get_metex_stanox_section

            update = True
            verbose = True

            metex_stanox_section = get_metex_stanox_section(update, verbose)
            print(metex_stanox_section)
        """

        pickle_filename = "metex-stanox-section.pickle"
        path_to_pickle = cdd_railway_codes(pickle_filename)

        if os.path.isfile(path_to_pickle) and not update:
            metex_stanox_section = load_pickle(path_to_pickle)

        else:
            try:
                metex_stanox_section = read_metex_table('StanoxSection')

                # In errata_tiploc, {'CLAPS47': 'CLPHS47'} might be problematic.
                errata = load_json(cdd_railway_codes("metex-errata.json"))
                errata_stanox = errata['STANOX']

                metex_stanox_section.rename(
                    columns={'Id': 'StanoxSectionId', 'Description': 'StanoxSection'}, inplace=True)
                metex_stanox_section.replace({'StartStanox': errata_stanox}, inplace=True)
                metex_stanox_section.replace({'EndStanox': errata_stanox}, inplace=True)
                metex_stanox_section.LocationId = metex_stanox_section.LocationId.map(
                    lambda x: '' if pd.isna(x) else str(int(x)))

                stanox_sec = metex_stanox_section.copy(deep=True)

                stanox_sec[['Start_Raw', 'End_Raw']] = stanox_sec.StanoxSection.str.split(' : ').apply(
                    pd.Series)
                stanox_sec[['Start', 'End']] = stanox_sec[['Start_Raw', 'End_Raw']]
                # Solve duplicated STANOX
                unknown_stanox_loc = load_json(cdd_railway_codes("problematic-stanox-locations.json"))
                stanox_sec.replace({'Start': unknown_stanox_loc}, inplace=True)
                stanox_sec.replace({'End': unknown_stanox_loc}, inplace=True)

                #
                stanox_location = self.get_metex_stanox_location(update)
                stanox_loc = stanox_location.set_index(['Stanox', 'LocationId'])
                temp = stanox_sec.join(stanox_loc, on=['StartStanox', 'LocationId'])
                temp = temp[temp.Description.notnull()]
                stanox_sec.loc[temp.index, 'Start'] = temp.Description
                temp = stanox_sec.join(stanox_loc, on=['EndStanox', 'LocationId'])
                temp = temp[temp.Description.notnull()]
                stanox_sec.loc[temp.index, 'End'] = temp.Description

                # Check if 'Start' and 'End' have STANOX codes
                # instead of location names using STANOX-dictionary
                lid = LocationIdentifiers()
                rc_stanox_dict = lid.make_loc_id_dict('STANOX')
                temp = stanox_sec.join(rc_stanox_dict, on='Start')
                valid_loc = temp[temp.Location.notnull()]
                if not valid_loc.empty:
                    stanox_sec.loc[valid_loc.index, 'Start'] = valid_loc.Location
                temp = stanox_sec.join(rc_stanox_dict, on='End')
                valid_loc = temp[temp.Location.notnull()]
                if not valid_loc.empty:
                    stanox_sec.loc[valid_loc.index, 'End'] = valid_loc.Location

                # Check if 'Start' and 'End' have STANOX/TIPLOC codes using STANOX-TIPLOC-dictionary
                rc_stanox_tiploc_dict = lid.make_loc_id_dict(['STANOX', 'TIPLOC'])
                temp = stanox_sec.join(rc_stanox_tiploc_dict, on=['StartStanox', 'Start'])
                valid_loc = temp[temp.Location.notnull()][['Start', 'Location']]
                if not valid_loc.empty:
                    stanox_sec.loc[valid_loc.index, 'Start'] = valid_loc.apply(
                        lambda x: find_similar_str(x[0], x[1]) if isinstance(x[1], (list, tuple)) else
                        x[1], axis=1)
                temp = stanox_sec.join(rc_stanox_tiploc_dict, on=['EndStanox', 'End'])
                valid_loc = temp[temp.Location.notnull()][['End', 'Location']]
                if not valid_loc.empty:
                    stanox_sec.loc[valid_loc.index, 'End'] = valid_loc.apply(
                        lambda x: find_similar_str(x[0], x[1]) if isinstance(x[1], (list, tuple)) else
                        x[1], axis=1)

                # Check if 'Start' and 'End' have STANOX/STANME codes using STANOX-STANME-dictionary
                rc_stanox_stanme_dict = lid.make_loc_id_dict(['STANOX', 'STANME'])
                temp = stanox_sec.join(rc_stanox_stanme_dict, on=['StartStanox', 'Start'])
                valid_loc = temp[temp.Location.notnull()][['Start', 'Location']]
                if not valid_loc.empty:
                    stanox_sec.loc[valid_loc.index, 'Start'] = valid_loc.apply(
                        lambda x: find_similar_str(x[0], x[1]) if isinstance(x[1], (list, tuple)) else
                        x[1], axis=1)
                temp = stanox_sec.join(rc_stanox_stanme_dict, on=['EndStanox', 'End'])
                valid_loc = temp[temp.Location.notnull()][['End', 'Location']]
                if not valid_loc.empty:
                    stanox_sec.loc[valid_loc.index, 'End'] = valid_loc.apply(
                        lambda x: find_similar_str(x[0], x[1]) if isinstance(x[1], (list, tuple)) else
                        x[1], axis=1)

                # Apply manually-created dictionaries
                loc_name_replacement_dict = fetch_loc_names_repl_dict('Start')
                stanox_sec.replace(loc_name_replacement_dict, inplace=True)
                loc_name_regexp_replacement_dict = fetch_loc_names_repl_dict('Start', regex=True)
                stanox_sec.replace(loc_name_regexp_replacement_dict, inplace=True)
                loc_name_replacement_dict = fetch_loc_names_repl_dict('End')
                stanox_sec.replace(loc_name_replacement_dict, inplace=True)
                loc_name_regexp_replacement_dict = fetch_loc_names_repl_dict('End', regex=True)
                stanox_sec.replace(loc_name_regexp_replacement_dict, inplace=True)

                # Finalise cleansing
                stanox_sec.End.fillna(stanox_sec.Start, inplace=True)
                temp = stanox_sec.join(rc_stanox_dict, on='StartStanox')
                temp = temp[temp.Location.notnull()][['Start', 'Location']]
                stanox_sec.loc[temp.index, 'Start'] = temp.apply(
                    lambda x: find_similar_str(x[0], x[1]) if isinstance(x[1], (list, tuple)) else x[1],
                    axis=1)
                temp = stanox_sec.join(rc_stanox_dict, on='EndStanox')
                temp = temp[temp.Location.notnull()][['End', 'Location']]
                stanox_sec.loc[temp.index, 'End'] = temp.apply(
                    lambda x: find_similar_str(x[0], x[1]) if isinstance(x[1], (list, tuple)) else x[1],
                    axis=1)

                #
                section = stanox_sec.Start + ' : ' + stanox_sec.End
                non_sec = stanox_sec.Start.eq(stanox_sec.End)
                section[non_sec] = stanox_sec[non_sec].Start
                metex_stanox_section.StanoxSection = section
                metex_stanox_section.insert(2, 'StartLocation', stanox_sec.Start.values)
                metex_stanox_section.insert(3, 'EndLocation', stanox_sec.End.values)
                metex_stanox_section[['StartLocation', 'EndLocation']] = stanox_sec[['Start', 'End']]

                # Add raw data to the original dataframe
                raw_col_names = ['StanoxSection_Raw', 'StartLocation_Raw', 'EndLocation_Raw']
                metex_stanox_section[raw_col_names] = stanox_sec[
                    ['StanoxSection', 'Start_Raw', 'End_Raw']]

                save_pickle(metex_stanox_section, path_to_pickle, verbose=verbose)

            except Exception as e:
                print("Failed to get NR_METEX STANOX section data. {}.".format(e))
                metex_stanox_section = pd.DataFrame()

        return metex_stanox_section

    def get_location_metadata_plus(self, update=False, verbose=False):
        """
        Get data of location codes by assembling resources from NR_METEX_* database.

        :param update: whether to check on update and proceed to update the package data,
            defaults to ``False``
        :type update: bool
        :param verbose: whether to print relevant information in console as the function runs,
            defaults to ``False``
        :type verbose: bool, int
        :return: data of location codes
        :rtype: pandas.DataFrame, None

        **Test**::

            from spreadsheet.incidents import get_location_metadata_plus

            update = True
            verbose = True

            loc_meta_plus = get_location_metadata_plus(update, verbose)
            print(loc_meta_plus)
        """

        pickle_filename = "metex-location-metadata.pickle"
        path_to_pickle = cdd_railway_codes(pickle_filename)

        if os.path.isfile(path_to_pickle) and not update:
            loc_meta_plus = load_pickle(path_to_pickle)
        else:
            try:
                # Location
                location = self.get_metex_location(update=update, verbose=verbose)

                # # STANOX location
                stanox_location = self.get_metex_stanox_location(update=update, verbose=verbose)
                stanox_location = stanox_location[['Stanox', 'Description', 'Name', 'ELR', 'Yards']]
                stanox_location.drop([2964, 6049, 2258, 1068, 2361, 4390], inplace=True)
                stanox_location.set_index(['Stanox', 'Description'], inplace=True)

                # STANOX section
                stanox_section = self.get_metex_stanox_section(update=update, verbose=verbose)

                # Get metadata by 'LocationId'
                location.index = location.index.astype(str)
                loc_meta_plus = stanox_section.join(location, on='LocationId')

                # --------------------------------------------------------------------------
                ref = self.get_location_metadata(update=update, verbose=verbose)['STANOX']
                ref.drop_duplicates(subset=['STANOX', 'LOOKUP_NAME'], inplace=True)
                ref.set_index(['STANOX', 'LOOKUP_NAME'], inplace=True)

                # Replace metadata coordinates data with ref coordinates if available
                temp = loc_meta_plus.join(ref, on=['StartStanox', 'StartLocation'])
                temp = temp[temp.DEG_LAT.notnull() & temp.DEG_LONG.notnull()]
                loc_meta_plus.loc[temp.index, 'StartLongitude':'StartLatitude'] = temp[
                    ['DEG_LONG', 'DEG_LAT']].values
                loc_meta_plus.loc[temp.index, 'ApproximateStartLocation'] = False
                loc_meta_plus.loc[
                    loc_meta_plus.index.difference(temp.index), 'ApproximateStartLocation'] = True

                temp = loc_meta_plus.join(ref, on=['EndStanox', 'EndLocation'])
                temp = temp[temp.DEG_LAT.notnull() & temp.DEG_LONG.notnull()]
                loc_meta_plus.loc[temp.index, 'EndLongitude':'EndLatitude'] = temp[
                    ['DEG_LONG', 'DEG_LAT']].values
                loc_meta_plus.loc[temp.index, 'ApproximateEndLocation'] = False
                loc_meta_plus.loc[
                    loc_meta_plus.index.difference(temp.index), 'ApproximateEndLocation'] = True

                # Get reference metadata from RailwayCodes
                stn = Stations()
                station_data = stn.fetch_station_data()['Railway station data']
                station_data = station_data[
                    ['Station', 'Degrees Longitude', 'Degrees Latitude']].dropna()
                station_data.drop_duplicates(subset=['Station'], inplace=True)
                station_data.set_index('Station', inplace=True)

                temp = loc_meta_plus.join(station_data, on='StartLocation')
                na_i = temp['Degrees Longitude'].notnull() & temp['Degrees Latitude'].notnull()
                loc_meta_plus.loc[na_i, 'StartLongitude':'StartLatitude'] = \
                    temp.loc[na_i, ['Degrees Longitude', 'Degrees Latitude']].values
                loc_meta_plus.loc[na_i, 'ApproximateStartLocation'] = False
                loc_meta_plus.loc[~na_i, 'ApproximateStartLocation'] = True

                temp = loc_meta_plus.join(station_data, on='EndLocation')
                na_i = temp['Degrees Longitude'].notnull() & temp['Degrees Latitude'].notnull()
                loc_meta_plus.loc[na_i, 'EndLongitude':'EndLatitude'] = \
                    temp.loc[na_i, ['Degrees Longitude', 'Degrees Latitude']].values
                loc_meta_plus.loc[na_i, 'ApproximateEndLocation'] = False
                loc_meta_plus.loc[~na_i, 'ApproximateEndLocation'] = True

                loc_meta_plus.ApproximateLocation = \
                    loc_meta_plus.ApproximateStartLocation | loc_meta_plus.ApproximateEndLocation

                # Finalise
                location_cols = stanox_location.columns
                start_cols = ['Start' + x for x in location_cols]
                loc_meta_plus = loc_meta_plus.join(stanox_location.drop_duplicates('Name'),
                                                   on=['StartStanox', 'StartLocation'])
                loc_meta_plus.rename(columns=dict(zip(location_cols, start_cols)), inplace=True)
                end_cols = ['End' + x for x in location_cols]
                loc_meta_plus = loc_meta_plus.join(stanox_location.drop_duplicates('Name'),
                                                   on=['EndStanox', 'EndLocation'])
                loc_meta_plus.rename(columns=dict(zip(location_cols, end_cols)), inplace=True)
                loc_meta_plus[start_cols + end_cols] = loc_meta_plus[start_cols + end_cols].fillna('')

                save_pickle(loc_meta_plus, path_to_pickle, verbose=verbose)

            except Exception as e:
                print("Failed to get metadata of location codes. {}.".format(e))
                loc_meta_plus = None

        return loc_meta_plus

    # == Incidents data ===============================================================================

    @staticmethod
    def _cleanse_stanox_section_column(data, col_name='StanoxSection', sep=' : ', update_dict=False):
        """
        Cleanse the column that contains location information in the data of incident records.

        :param data: data of incident records
        :type data: pandas.DataFrame
        :param col_name: name of the column to be cleansed, defaults to ``'StanoxSection'``
        :type col_name: str
        :param sep: separator for the column, defaults to ``' : '``
        :type sep: str
        :param update_dict: whether to update the location codes dictionary
        :type update_dict: bool
        :return: cleansed data frame
        :rtype: pandas.DataFrame

        **Test**::

            col_name = 'StanoxSection'
            sep = ' : '
            update_dict = False

            data = s8weather_incidents.copy()
        """

        dat = data.copy(deep=True)
        old_column_name = col_name + '_Raw'
        dat.rename(columns={col_name: old_column_name}, inplace=True)
        if sep is not None and isinstance(sep, str):
            start_end_raw = dat[old_column_name].str.split(sep, expand=True)
            #
            start_end_raw.columns = ['StartLocation_Raw', 'EndLocation_Raw']
            start_end_raw.EndLocation_Raw.fillna(start_end_raw.StartLocation_Raw, inplace=True)
            dat = dat.join(start_end_raw)
        else:
            start_end_raw = dat[[old_column_name]]

        # In errata_tiploc, {'CLAPS47': 'CLPHS47'} might be problematic.
        errata = load_json(cdd_railway_codes("metex-errata.json"))
        errata_stanox, errata_tiploc, errata_stanme = errata.values()
        start_end = start_end_raw.replace(errata_stanox)
        start_end.replace(errata_tiploc, inplace=True)
        start_end.replace(errata_stanme, inplace=True)

        # Get rid of duplicated records
        def tidy_alt_codes(code_dict_df, raw_loc):
            raw_loc.columns = [x.replace('_Raw', '') for x in raw_loc.columns]
            tmp = raw_loc.join(code_dict_df, on='StartLocation')
            tmp = tmp[tmp.Location.notnull()]
            raw_loc.loc[tmp.index, 'StartLocation'] = tmp.Location.values
            tmp = raw_loc.join(code_dict_df, on='EndLocation')
            tmp = tmp[tmp.Location.notnull()]
            raw_loc.loc[tmp.index, 'EndLocation'] = tmp.Location.values
            return raw_loc

        lid = LocationIdentifiers()
        #
        stanox_dict = lid.make_loc_id_dict('STANOX', update=update_dict)
        start_end = tidy_alt_codes(stanox_dict, start_end)
        #
        stanme_dict = lid.make_loc_id_dict('STANME', update=update_dict)
        start_end = tidy_alt_codes(stanme_dict, start_end)
        #
        tiploc_dict = lid.make_loc_id_dict('TIPLOC', update=update_dict)
        start_end = tidy_alt_codes(tiploc_dict, start_end)

        #
        start_end.replace(fetch_loc_names_repl_dict('Description', regex=True), inplace=True)
        start_end.replace(fetch_loc_names_repl_dict(old_column_name, regex=True), inplace=True)
        start_end.replace(fetch_loc_names_repl_dict(regex=True), regex=True, inplace=True)
        start_end.replace(fetch_loc_names_repl_dict(), inplace=True)

        # ref = rc.get_location_codes()['Locations']
        # ref.drop_duplicates(subset=['Location'], inplace=True)
        # ref_loc = list(set(ref.Location))
        #
        # temp = start_end_raw.join(ref.set_index('Location'), on='StartLocation')
        # temp_loc = list(set(temp[temp.STANOX.isnull()].StartLocation))
        #
        # temp = start_end_raw.join(ref.set_index('Location'), on='EndLocation')
        # temp_loc = list(set(temp[temp.STANOX.isnull()].EndLocation))

        # Create new StanoxSection column
        dat[col_name] = start_end.StartLocation + sep + start_end.EndLocation
        mask_single = start_end.StartLocation == start_end.EndLocation
        dat[col_name][mask_single] = start_end.StartLocation[mask_single]

        # Resort column order
        col_names = [list(v) for k, v in
                     itertools.groupby(list(dat.columns), lambda x: x == old_column_name) if not k]
        add_names = [old_column_name] + col_names[1][-3:] + ['StartLocation', 'EndLocation']
        col_names = col_names[0] + add_names + col_names[1][:-3]

        cleansed_data = dat.join(start_end)[col_names]

        return cleansed_data

    def _cleanse_geographical_coordinates(self, data, update_metadata=False):
        """
        Look up geographical coordinates for each incident location.

        :param data: data of incident records
        :type data: pandas.DataFrame
        :param update_metadata: whether to update the location metadata, defaults to ``False``
        :type update_metadata: bool
        :return: cleansed data frame
        :rtype: pandas.DataFrame

        **Test**::

            update_metadata = False
            data = s8weather_incidents.copy()

            >>> from spreadsheet import Schedule8IncidentsSpreadsheet

            >>> sis = Schedule8IncidentsSpreadsheet()

        """

        dat = data.copy(deep=True)

        # Find geographical coordinates for each incident location
        ref_loc_dat_1, _ = self.get_location_metadata(update=update_metadata).values()
        coords_cols = ['EASTING', 'NORTHING', 'DEG_LONG', 'DEG_LAT']
        coords_cols_alt = ['Easting', 'Northing', 'Longitude', 'Latitude']
        ref_loc_dat_1.rename(columns=dict(zip(coords_cols, coords_cols_alt)), inplace=True)
        ref_loc_dat_1 = ref_loc_dat_1[['LOOKUP_NAME'] + coords_cols_alt].drop_duplicates('LOOKUP_NAME')

        tmp, idx = [], []
        for i in ref_loc_dat_1.index:
            x_ = ref_loc_dat_1.LOOKUP_NAME.loc[i]
            if isinstance(x_, tuple):
                for y in x_:
                    tmp.append([y] + ref_loc_dat_1[coords_cols_alt].loc[i].to_list())
        ref_colname_1 = ['LOOKUP_NAME'] + coords_cols_alt
        ref_loc_dat_1 = ref_loc_dat_1[ref_colname_1].append(pd.DataFrame(tmp, columns=ref_colname_1))
        ref_loc_dat_1.drop_duplicates(subset=coords_cols_alt, keep='last', inplace=True)
        ref_loc_dat_1.set_index('LOOKUP_NAME', inplace=True)

        dat = dat.join(ref_loc_dat_1[coords_cols_alt], on='StartLocation')
        dat.rename(columns=dict(zip(coords_cols_alt, ['Start' + c for c in coords_cols_alt])),
                   inplace=True)
        dat = dat.join(ref_loc_dat_1[coords_cols_alt], on='EndLocation')
        dat.rename(columns=dict(zip(coords_cols_alt, ['End' + c for c in coords_cols_alt])),
                   inplace=True)

        # Get location metadata for reference --------------------------------
        location_metadata = self.get_location_metadata_plus(update=update_metadata)
        start_locations = location_metadata[['StartLocation', 'StartLongitude', 'StartLatitude']]
        start_locations.columns = [c.replace('Start', '') for c in start_locations.columns]
        end_locations = location_metadata[['EndLocation', 'EndLongitude', 'EndLatitude']]
        end_locations.columns = [c.replace('End', '') for c in end_locations.columns]
        loc_metadata = pd.concat([start_locations, end_locations], ignore_index=True)
        loc_metadata = loc_metadata.drop_duplicates('Location').set_index('Location')

        # Fill in NA coordinates
        temp = dat[dat.StartEasting.isnull() | dat.StartLongitude.isnull()]
        temp = temp.join(loc_metadata, on='StartLocation')
        dat.loc[temp.index, 'StartLongitude':'StartLatitude'] = temp[['Longitude', 'Latitude']].values
        dat.loc[temp.index, 'StartEasting'], dat.loc[temp.index, 'StartNorthing'] = wgs84_to_osgb36(
            temp.Longitude.values, temp.Latitude.values)

        temp = dat[dat.EndEasting.isnull() | dat.EndLongitude.isnull()]
        temp = temp.join(loc_metadata, on='EndLocation')
        dat.loc[temp.index, 'EndLongitude':'EndLatitude'] = temp[['Longitude', 'Latitude']].values

        # Dalston Junction (East London Line)     --> Dalston Junction [-0.0751, 51.5461]
        # Ashford West Junction (CTRL)            --> Ashford West Junction [0.86601557, 51.146927]
        # Southfleet Junction                     --> ? [0.34262910, 51.419354]
        # Channel Tunnel Eurotunnel Boundary CTRL --> ? [1.1310482, 51.094808]
        na_loc = ['Dalston Junction (East London Line)', 'Ashford West Junction (CTRL)',
                  'Southfleet Junction', 'Channel Tunnel Eurotunnel Boundary CTRL']
        na_loc_longlat = [[-0.0751, 51.5461], [0.86601557, 51.146927], [0.34262910, 51.419354],
                          [1.1310482, 51.094808]]
        for x_, longlat in zip(na_loc, na_loc_longlat):
            if x_ in list(temp.EndLocation):
                idx = temp[temp.EndLocation == x_].index
                temp.loc[idx, 'EndLongitude':'Latitude'] = longlat * 2
                dat.loc[idx, 'EndLongitude':'EndLatitude'] = longlat

        dat.loc[temp.index, 'EndEasting'], dat.loc[temp.index, 'EndNorthing'] = wgs84_to_osgb36(
            temp.Longitude.values, temp.Latitude.values)

        # ref 2 ----------------------------------------------
        stn = Stations()
        ref_metadata_2 = stn.fetch_station_data()['Railway station data']
        ref_metadata_2 = ref_metadata_2[['Station', 'Degrees Longitude', 'Degrees Latitude']]
        ref_metadata_2 = ref_metadata_2.dropna().drop_duplicates('Station')
        ref_metadata_2.columns = [x.replace('Degrees ', '') for x in ref_metadata_2.columns]
        ref_metadata_2.set_index('Station', inplace=True)

        temp = dat.join(ref_metadata_2, on='StartLocation')
        temp_start = temp[temp.Longitude.notnull() & temp.Latitude.notnull()]
        dat.loc[temp_start.index, 'StartLongitude':'StartLatitude'] = temp_start[
            ['Longitude', 'Latitude']].values

        temp = dat.join(ref_metadata_2, on='EndLocation')
        temp_end = temp[temp.Longitude.notnull() & temp.Latitude.notnull()]
        dat.loc[temp_end.index, 'EndLongitude':'EndLatitude'] = temp_end[
            ['Longitude', 'Latitude']].values

        # Let (Longitude, Latitude) be almost equivalent to (Easting, Northing)
        dat.loc[:, 'StartLongitude':'StartLatitude'] = np.array(
            osgb36_to_wgs84(dat.StartEasting.values, dat.StartNorthing.values)).T
        dat.loc[:, 'EndLongitude':'EndLatitude'] = np.array(
            osgb36_to_wgs84(dat.EndEasting.values, dat.EndNorthing.values)).T

        # Convert coordinates to shapely.geometry.Point
        dat['StartXY'] = dat.apply(lambda x: shapely.geometry.Point(x.StartEasting, x.StartNorthing),
                                   axis=1)
        dat['EndXY'] = dat.apply(lambda x: shapely.geometry.Point(x.EndEasting, x.EndNorthing), axis=1)
        dat['StartLongLat'] = dat.apply(
            lambda x: shapely.geometry.Point(x.StartLongitude, x.StartLatitude), axis=1)
        dat['EndLongLat'] = dat.apply(lambda x: shapely.geometry.Point(x.EndLongitude, x.EndLatitude),
                                      axis=1)

        return dat

    def get_schedule8_weather_incidents(self, route_name=None, weather_category=None, update=False,
                                        verbose=False):
        """
        Get data of Schedule 8 weather incidents from spreadsheet file.

        :param route_name: name of a Route, defaults to ``None``
        :type route_name: str, None
        :param weather_category: weather category, defaults to ``None``
        :type weather_category: str, None
        :param update: whether to check on update and proceed to update the package data,
            defaults to ``False``
        :type update: bool
        :param verbose: whether to print relevant information in console as the function runs,
            defaults to ``False``
        :type verbose: bool, int
        :return: data of Schedule 8 weather incidents
        :rtype: pandas.DataFrame, None

        **Test**::

            from spreadsheet.incidents import get_schedule8_weather_incidents

            route_name = None
            weather_category = None
            update = False
            verbose = True

            s8weather_incidents = get_schedule8_weather_incidents(route_name, weather_category, update,
                                                                  verbose)
            print(s8weather_incidents)
        """

        pickle_filename = make_filename(self.DataFilename1, route_name, weather_category)
        path_to_pickle = self.cdd(pickle_filename)

        if os.path.isfile(path_to_pickle) and not update:
            s8weather_incidents = load_pickle(path_to_pickle)

        else:
            try:
                # Load data from the raw file
                s8weather_incidents = pd.read_excel(path_to_pickle.replace(".pickle", ".xlsx"),
                                                    parse_dates=['StartDate', 'EndDate'],
                                                    day_first=True,
                                                    converters={'stanoxSection': str})
                s8weather_incidents.rename(
                    columns={'StartDate': 'StartDateTime',
                             'EndDate': 'EndDateTime',
                             'stanoxSection': 'StanoxSection',
                             'imdm': 'IMDM',
                             'WeatherCategory': 'WeatherCategoryCode',
                             'WeatherCategory.1': 'WeatherCategory',
                             'Reason': 'IncidentReason',
                             # 'Minutes': 'DelayMinutes',
                             'Description': 'IncidentReasonDescription',
                             'Category': 'IncidentCategory',
                             'CategoryDescription': 'IncidentCategoryDescription'},
                    inplace=True)

                # Add information about incident reason
                dag = DelayAttributionGlossary()
                incident_reason_metadata = dag.read_incident_reason_metadata()
                incident_reason_metadata.columns = [
                    c.replace('_', '') for c in incident_reason_metadata.columns]
                s8weather_incidents = s8weather_incidents.join(
                    incident_reason_metadata, on='IncidentReason', rsuffix='_meta')
                s8weather_incidents.drop(
                    [x for x in s8weather_incidents.columns if '_meta' in x], axis=1, inplace=True)

                # Cleanse the location data
                s8weather_incidents = self._cleanse_stanox_section_column(
                    s8weather_incidents, col_name='StanoxSection', sep=' : ', update_dict=update)

                # Look up geographical coordinates for each incident location
                s8weather_incidents = self._cleanse_geographical_coordinates(s8weather_incidents)

                # Retain data for specific Route and Weather category
                s8weather_incidents = get_subset(
                    s8weather_incidents, route_name, weather_category, rearrange_index=False)

                save_pickle(s8weather_incidents, path_to_pickle, verbose=verbose)

            except Exception as e:
                print("Failed to get \"Schedule 8 Weather Incidents\". {}".format(e))
                s8weather_incidents = None

        return s8weather_incidents

    def get_schedule8_weather_incidents_02062006_31032014(self, route_name=None, weather_category=None,
                                                          update=False, verbose=False):
        """
        Get data of Schedule 8 weather incident from Schedule8WeatherIncidents-02062006-31032014.xlsm.

        :param route_name: name of a Route, defaults to ``None``
        :type route_name: str, None
        :param weather_category: weather category, defaults to ``None``
        :type weather_category: str, None
        :param update: whether to check on update and proceed to update the package data,
            defaults to ``False``
        :type update: bool
        :param verbose: whether to print relevant information in console as the function runs,
            defaults to ``False``
        :type verbose: bool, int
        :return: data of Schedule 8 weather incidents
        :rtype: pandas.DataFrame, None

        .. note::

            Description:

            "Details of schedule 8 Incidents together with Weather leading up to the incident.
            Although this file contains other Weather categories, the main focus of this prototype is
            adhesion."

            "* WORK IN PROGRESS *
            MET-9 - Report of Schedule 8 adhesion Incidents vs Weather conditions Done."

        **Test**::

            from spreadsheet.incidents import get_schedule8_weather_incidents_02062006_31032014

            route_name = None
            weather_category = None
            update = True
            verbose = True

            s8weather_incidents = get_schedule8_weather_incidents_02062006_31032014(
                route_name, weather_category, update, verbose)
            print(s8weather_incidents)
        """

        # Path to the file
        pickle_filename = make_filename(self.DataFilename2, route_name, weather_category)
        path_to_pickle = self.cdd(pickle_filename)

        if os.path.isfile(path_to_pickle) and not update:
            s8weather_incidents = load_pickle(path_to_pickle)

        else:
            try:
                # Open the original file
                path_to_xlsm = path_to_pickle.replace(".pickle", ".xlsm")
                workbook = pd.ExcelFile(path_to_xlsm)

                # 'Thresholds' -------------------------------------------------------------
                thresholds = workbook.parse(sheet_name='Thresholds', usecols='A:F').dropna()
                thresholds.columns = [col.replace(' ', '') for col in thresholds.columns]
                thresholds.WeatherHazard = thresholds.WeatherHazard.str.strip().str.upper()
                thresholds.index = range(len(thresholds))

                # 'Data' -----------------------------------------------------------------------------
                incident_records = workbook.parse(sheet_name='Data',
                                                  parse_dates=['StartDate', 'EndDate'], dayfirst=True,
                                                  converters={'stanoxSection': str})
                incident_records.rename(columns={'StartDate': 'StartDateTime',
                                                 'EndDate': 'EndDateTime',
                                                 'Year': 'FinancialYear',
                                                 'stanoxSection': 'StanoxSection',
                                                 'imdm': 'IMDM',
                                                 'Reason': 'IncidentReason',
                                                 # 'Minutes': 'DelayMinutes',
                                                 # 'Cost': 'DelayCost',
                                                 'CategoryDescription': 'IncidentCategoryDescription'},
                                        inplace=True)
                hazard_cols = [
                    x for x in enumerate(incident_records.columns) if 'Weather Hazard' in x[1]]
                obs_cols = [
                    (i - 1, re.search(r'(?<= \()\w+', x).group().upper()) for i, x in hazard_cols]
                hazard_cols = [(i + 1, x + '_WeatherHazard') for i, x in obs_cols]
                for i, x in obs_cols + hazard_cols:
                    incident_records.rename(columns={incident_records.columns[i]: x}, inplace=True)

                # data.WeatherCategory = data.WeatherCategory.replace('Heat Speed/Buckle', 'Heat')
                dag = DelayAttributionGlossary()
                incident_reason_metadata = dag.read_incident_reason_metadata()
                incident_reason_metadata.columns = [
                    c.replace('_', '') for c in incident_reason_metadata.columns]
                incident_records = incident_records.join(
                    incident_reason_metadata, on='IncidentReason', rsuffix='_meta')
                incident_records.drop(
                    [x for x in incident_records.columns if '_meta' in x], axis=1, inplace=True)

                # Cleanse the location data
                incident_records = self._cleanse_stanox_section_column(
                    incident_records, col_name='StanoxSection', sep=' : ', update_dict=update)

                # Look up geographical coordinates for each incident location
                incident_records = self._cleanse_geographical_coordinates(incident_records)

                # Retain data for specific Route and Weather category
                incident_records = get_subset(
                    incident_records, route_name, weather_category, rearrange_index=False)

                # Weather category lookup -------------------------------------------
                weather_category_lookup = workbook.parse(sheet_name='CategoryLookup')
                weather_category_lookup.columns = ['WeatherCategoryCode', 'WeatherCategory']

                # Make a dictionary
                s8weather_incidents = dict(
                    zip(workbook.sheet_names, [thresholds, incident_records, weather_category_lookup]))

                workbook.close()

                # Save the workbook data
                save_pickle(s8weather_incidents, path_to_pickle, verbose=verbose)

            except Exception as e:
                print("Failed to get/parse \"Schedule8WeatherIncidents-02062006-31032014.xlsm\". "
                      "{}.".format(e))
                s8weather_incidents = None

        return s8weather_incidents
