""" Read and cleanse data of NR_Vegetation_* database. """

import re

from pyhelpers.geom import osgb36_to_wgs84
from pyhelpers.ops import confirmed
from pyhelpers.store import load_pickle, save_pickle
from pyrcs.utils import nr_mileage_num_to_str, nr_mileage_str_to_num

from utils import *


class Vegetation:
    """
    Vegetation database.

    :ivar str Name: name of the data resource
    :ivar str Desc: brief description of the data resource
    :ivar str DatabaseName: name of the database that stores the data
    :ivar sqlalchemy.engine.Connection DatabaseConn: connection to the database

    **Test**::

        >>> from preprocessor import Vegetation

        >>> veg = Vegetation()

        >>> print(veg.Name)
        Vegetation
    """

    def __init__(self):
        self.Name = 'Vegetation'
        self.Desc = 'Vegetation'

        self.DatabaseName = 'NR_Vegetation_20141031'
        self.DatabaseConn = establish_mssql_connection(database_name=self.DatabaseName)

    # == Change directories ===========================================================================

    @staticmethod
    def cdd(*sub_dir, mkdir=False):
        """
        Change directory to "data\\vegetation\\database\\" and sub-directories / a file.

        :param sub_dir: name of directory or names of directories (and/or a filename)
        :type sub_dir: str
        :param mkdir: whether to create a directory, defaults to ``False``
        :type mkdir: bool
        :return: absolute path to "data\\vegetation\\database\\" and sub-directories / a file
        :rtype: str

        **Test**::

            >>> import os
            >>> from preprocessor import Vegetation

            >>> veg = Vegetation()

            >>> os.path.relpath(veg.cdd())
            'data\\vegetation\\database'
        """

        path = cdd_vegetation("database", *sub_dir, mkdir=mkdir)

        return path

    def cdd_tables(self, *sub_dir, mkdir=False):
        """
        Change directory to "..\\data\\vegetation\\database\\tables\\" and sub-directories / a file.

        :param sub_dir: name of directory or names of directories (and/or a filename)
        :type sub_dir: str
        :param mkdir: whether to create a directory, defaults to ``False``
        :type mkdir: bool
        :return: absolute path to "..\\data\\vegetation\\database\\tables\\" and sub-directories / a file
        :rtype: str
        """

        path = self.cdd("tables", *sub_dir, mkdir=mkdir)

        return path

    def cdd_views(self, *sub_dir, mkdir=False):
        """
        Change directory to "..\\data\\vegetation\\database\\views\\" and sub-directories / a file.

        :param sub_dir: name of directory or names of directories (and/or a filename)
        :type sub_dir: str
        :param mkdir: whether to create a directory, defaults to ``False``
        :type mkdir: bool
        :return: absolute path to "..\\data\\vegetation\\database\\views\\" and sub-directories / a file
        :rtype: str
        """

        path = self.cdd("views", *sub_dir, mkdir=mkdir)
        return path

    # == Read table data from the database ============================================================

    def read_table(self, table_name, index_col=None, route_name=None, schema_name='dbo', save_as=None,
                   update=False, **kwargs):
        """
        Read tables stored in NR_Vegetation_* database.
    
        :param table_name: name of a table
        :type table_name: str
        :param index_col: column(s) set to be index of the returned data frame, defaults to ``None``
        :type index_col: str or list or None
        :param route_name: name of a Route; if ``None`` (default), all Routes
        :type route_name: str or None
        :param schema_name: name of schema, defaults to ``'dbo'``
        :type schema_name: str
        :param save_as: file extension, defaults to ``None``
        :type save_as: str or None
        :param update: whether to check on update and proceed to update the package data,
            defaults to ``False``
        :type update: bool
        :param kwargs: optional parameters of `pandas.read_sql`_
        :return: data of the queried table stored in NR_Vegetation_* database
        :rtype: pandas.DataFrame
    
        .. _`pandas.read_sql`:
            https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_sql.html
    
        **Test**::
    
            >>> from preprocessor import Vegetation
            
            >>> veg = Vegetation()

            >>> adverse_wind_tbl = veg.read_table(table_name='AdverseWind', route_name='Anglia')

            >>> adverse_wind_tbl
                Route  DaysPerFurlongPerYear
            0  ANGLIA               0.000678
        """

        table = schema_name + '.' + table_name

        if route_name is None:
            # Get all data of a given table
            sql_query = "SELECT * FROM {}".format(table)
        else:
            # given a specific Route
            sql_query = "SELECT * FROM {} WHERE Route = '{}'".format(table, route_name)

        # Create a pd.DataFrame of the queried table
        data = pd.read_sql(sql=sql_query, con=self.DatabaseConn, index_col=index_col, **kwargs)

        # Save the DataFrame as a worksheet locally?
        if save_as:
            path_to_file = self.cdd_tables(table_name + save_as)
            if not os.path.isfile(path_to_file) or update:
                save(data, path_to_file, index=False if index_col is None else True)

        return data

    def get_primary_key(self, table_name):
        """
        Get primary keys of a table stored in database 'NR_Vegetation_20141031'.
    
        :param table_name: name of a table stored in the database 'NR_Vegetation_20141031'
        :type table_name: str
        :return: a (list of) primary key(s)
        :rtype: list
    
        **Test**::
    
            >>> from preprocessor import Vegetation
            
            >>> veg = Vegetation()

            >>> primary_key = veg.get_primary_key(table_name='AdverseWind')
            
            >>> primary_key
            ['Route']
        """

        pri_key = get_table_primary_keys(database_name=self.DatabaseName, table_name=table_name)

        return pri_key

    # == Get table data ===============================================================================

    def get_adverse_wind(self, update=False, save_original_as=None, verbose=False):
        """
        Get data of the table 'AdverseWind'.

        :param update: whether to check on update and proceed to update the package data,
            defaults to ``False``
        :type update: bool
        :param save_original_as: file extension for saving the original data, defaults to ``None``
        :type save_original_as: str or None
        :param verbose: whether to print relevant information in console as the function runs,
            defaults to ``False``
        :type verbose: bool or int
        :return: data of the table 'AdverseWind'
        :rtype: pandas.DataFrame or None

        **Test**::

            >>> from preprocessor import Vegetation
            
            >>> veg = Vegetation()

            >>> adverse_wind_tbl = veg.get_adverse_wind(update=True, verbose=True)
            Updating "AdverseWind.pickle" at "data\\vegetation\\database\\tables" ... Done.

            >>> adverse_wind_tbl.tail()
                                                   RouteAlias       DaysPerFurlongPerYear
            Route
            Scotland                                 SCOTLAND                    0.001052
            South East                         [KENT, SUSSEX]  [0.001615151, 0.002434749]
            Wales                                       WALES                     0.00201
            Wessex                                     WESSEX                    0.001116
            Western     [WESTERN Thames Valley, WESTERN West]  [0.001520801, 0.001573427]
        """

        table_name = 'AdverseWind'
        path_to_pickle = self.cdd_tables(table_name + ".pickle")

        if os.path.isfile(path_to_pickle) and not update:
            adverse_wind = load_pickle(path_to_pickle)

        else:
            try:
                adverse_wind = self.read_table(table_name, index_col=None, save_as=save_original_as,
                                               update=update)

                update_nr_route_names(adverse_wind, route_col_name='Route')  # Update route names
                adverse_wind = adverse_wind.groupby('Route').agg(list).applymap(
                    lambda x: x if len(x) > 1 else x[0])

                save_pickle(adverse_wind, path_to_pickle, verbose=verbose)

            except Exception as e:
                print("Failed to get \"{}\". {}.".format(table_name, e))
                adverse_wind = None

        return adverse_wind

    def get_cutting_angle_class(self, update=False, save_original_as=None, verbose=False):
        """
        Get data of the table 'CuttingAngleClass'.

        :param update: whether to check on update and proceed to update the package data,
            defaults to ``False``
        :type update: bool
        :param save_original_as: file extension for saving the original data, defaults to ``None``
        :type save_original_as: str or None
        :param verbose: whether to print relevant information in console as the function runs,
            defaults to ``False``
        :type verbose: bool or int
        :return: data of the table 'CuttingAngleClass'
        :rtype: pandas.DataFrame or None

        **Test**::

            >>> from preprocessor import Vegetation
            
            >>> veg = Vegetation()

            >>> cutting_angle_tbl = veg.get_cutting_angle_class(update=True, verbose=True)
            Updating "CuttingAngleClass.pickle" at "data\\vegetation\\database\\tables" ... Done.

            >>> cutting_angle_tbl.tail()
        """

        table_name = 'CuttingAngleClass'
        path_to_pickle = self.cdd_tables(table_name + ".pickle")

        if os.path.isfile(path_to_pickle) and not update:
            cutting_angle = load_pickle(path_to_pickle)

        else:
            try:
                cutting_angle = self.read_table(table_name, index_col=self.get_primary_key(table_name),
                                                save_as=save_original_as,
                                                update=update)

                save_pickle(cutting_angle, path_to_pickle, verbose=verbose)

            except Exception as e:
                print("Failed to get \"{}\". {}.".format(table_name, e))
                cutting_angle = None

        return cutting_angle

    def get_cutting_depth_class(self, update=False, save_original_as=None, verbose=False):
        """
        Get data of the table 'CuttingDepthClass'.

        :param update: whether to check on update and proceed to update the package data,
            defaults to ``False``
        :type update: bool
        :param save_original_as: file extension for saving the original data, defaults to ``None``
        :type save_original_as: str or None
        :param verbose: whether to print relevant information in console as the function runs,
            defaults to ``False``
        :type verbose: bool or int
        :return: data of the table 'CuttingDepthClass'
        :rtype: pandas.DataFrame or None

        **Test**::

            >>> from preprocessor import Vegetation
            
            >>> veg = Vegetation()

            >>> cutting_depth_tbl = veg.get_cutting_depth_class(update=True, verbose=True)
            Updating "CuttingDepthClass.pickle" at "data\\vegetation\\database\\tables" ... Done.

            >>> cutting_depth_tbl.tail()
        """

        table_name = 'CuttingDepthClass'
        path_to_pickle = self.cdd_tables(table_name + ".pickle")

        if os.path.isfile(path_to_pickle) and not update:
            cutting_depth = load_pickle(path_to_pickle)

        else:
            try:
                cutting_depth = self.read_table(table_name, index_col=self.get_primary_key(table_name),
                                                save_as=save_original_as,
                                                update=update)

                save_pickle(cutting_depth, path_to_pickle, verbose=verbose)

            except Exception as e:
                print("Failed to get \"{}\". {}.".format(table_name, e))
                cutting_depth = None

        return cutting_depth

    def get_du_list(self, index=True, update=False, save_original_as=None, verbose=False):
        """
        Get data of the table 'DUList'.

        :param index: whether to set an index column
        :type index: bool
        :param update: whether to check on update and proceed to update the package data,
            defaults to ``False``
        :type update: bool
        :param save_original_as: file extension for saving the original data, defaults to ``None``
        :type save_original_as: str or None
        :param verbose: whether to print relevant information in console as the function runs,
            defaults to ``False``
        :type verbose: bool or int
        :return: data of the table 'DUList'
        :rtype: pandas.DataFrame or None

        **Test**::

            >>> from preprocessor import Vegetation
            
            >>> veg = Vegetation()

            >>> du_list_tbl = veg.get_du_list(update=True, verbose=True)
            Updating "DUList-indexed.pickle" at "data\\vegetation\\database\\tables" ... Done.
            >>> du_list_tbl.tail()
                                                             StartMiles  EndMiles
            DUName         ELR  TrackID StartYards EndYards
            York MDU - HG7 HOS  3950     1555.0    1589.0        0.1555    0.1589
                           LEN2 1100    -44.0      16.0         -0.0044    0.0016
                                2100     87.0      129.0         0.0087    0.0129
                           LEN3 1901     75824.0   76007.0      43.0144   43.0327
                                1902     75912.0   76007.0      43.0232   43.0327

            >>> du_list_tbl = veg.get_du_list(index=False, update=True, verbose=True)
            Updating "DUList.pickle" at "data\\vegetation\\database\\tables" ... Done.
            >>> du_list_tbl.tail()
                          DUName   ELR  TrackID  StartMiles  StartYards  EndMiles  EndYards
            6260  York MDU - HG7   HOS     3950      0.1555      1555.0    0.1589    1589.0
            6261  York MDU - HG7  LEN2     1100     -0.0044       -44.0    0.0016      16.0
            6262  York MDU - HG7  LEN2     2100      0.0087        87.0    0.0129     129.0
            6263  York MDU - HG7  LEN3     1901     43.0144     75824.0   43.0327   76007.0
            6264  York MDU - HG7  LEN3     1902     43.0232     75912.0   43.0327   76007.0
        """

        table_name = 'DUList'
        path_to_pickle = self.cdd_tables(table_name + ("-indexed" if index else "") + ".pickle")

        if os.path.isfile(path_to_pickle) and not update:
            du_list = load_pickle(path_to_pickle)

        else:
            try:
                du_list = self.read_table(table_name,
                                          index_col=self.get_primary_key(table_name) if index else None,
                                          save_as=save_original_as, update=update)

                save_pickle(du_list, path_to_pickle, verbose=verbose)

            except Exception as e:
                print("Failed to get \"{}\". {}.".format(table_name, e))
                du_list = None

        return du_list

    def get_path_route(self, update=False, save_original_as=None, verbose=False):
        """
        Get data of the table 'PathRoute'.

        :param update: whether to check on update and proceed to update the package data,
            defaults to ``False``
        :type update: bool
        :param save_original_as: file extension for saving the original data, defaults to ``None``
        :type save_original_as: str or None
        :param verbose: whether to print relevant information in console as the function runs,
            defaults to ``False``
        :type verbose: bool or int
        :return: data of the table 'PathRoute'
        :rtype: pandas.DataFrame or None

        **Test**::

            >>> from preprocessor import Vegetation
            
            >>> veg = Vegetation()

            >>> path_route_tbl = veg.get_path_route(update=True, verbose=True)
            Updating "PathRoute.pickle" at "data\\vegetation\\database\\tables" ... Done.

            >>> path_route_tbl.tail()
                               ELR  TrackID  StartYards  ...  PercentUse  Comment Valid
            ZoneID Path   Seq                            ...
            SE     555087 30   XTD     1700      4359.0  ...       100.0           True
                          40   XTD     1507      4261.0  ...       100.0           True
                          50   LBW     3700       572.0  ...       100.0           True
                          60   LBW     3610       484.0  ...       100.0           True
                          70   LBW     3612       397.0  ...       100.0           True
            [5 rows x 7 columns]
        """

        table_name = 'PathRoute'
        path_to_pickle = self.cdd_tables(table_name + ".pickle")

        if os.path.isfile(path_to_pickle) and not update:
            path_route = load_pickle(path_to_pickle)

        else:
            try:
                path_route = self.read_table(table_name, index_col=self.get_primary_key(table_name),
                                             save_as=save_original_as,
                                             update=update)

                save_pickle(path_route, path_to_pickle, verbose=verbose)

            except Exception as e:
                print("Failed to get \"{}\". {}.".format(table_name, e))
                path_route = None

        return path_route

    def get_du_route(self, update=False, save_original_as=None, verbose=False):
        """
        Get data of the table 'Routes'.

        :param update: whether to check on update and proceed to update the package data,
            defaults to ``False``
        :type update: bool
        :param save_original_as: file extension for saving the original data, defaults to ``None``
        :type save_original_as: str or None
        :param verbose: whether to print relevant information in console as the function runs,
            defaults to ``False``
        :type verbose: bool or int
        :return: data of the table 'Routes'
        :rtype: pandas.DataFrame or None

        **Test**::

            >>> from preprocessor import Vegetation
            
            >>> veg = Vegetation()

            >>> routes_tbl = veg.get_du_route(update=True, verbose=True)
            Updating "Routes.pickle" at "data\\vegetation\\database\\tables" ... Done.

            >>> routes_tbl.tail()
                                            RouteAlias  ...                 Route
            DUName                                      ...
            Stafford MDU - HS0               LNW South  ...  London North Western
            Swindon MDU - HV2    WESTERN Thames Valley  ...               Western
            Tottenham MDU - HT8                 ANGLIA  ...                Anglia
            Woking MDU - HW3                    WESSEX  ...                Wessex
            York MDU - HG7                         LNE  ...        North and East
            [5 rows x 3 columns]
        """

        table_name = 'Routes'
        path_to_pickle = self.cdd_tables(table_name + ".pickle")

        if os.path.isfile(path_to_pickle) and not update:
            routes = load_pickle(path_to_pickle)

        else:
            try:
                # (Note that 'Routes' table contains information about Delivery Units)
                routes = self.read_table(table_name, index_col=self.get_primary_key(table_name),
                                         save_as=save_original_as,
                                         update=update)

                # Replace values in (index) column 'DUName'
                routes.index = routes.index.to_series().replace(
                    {'Lanc&Cumbria MDU - HR1': 'Lancashire & Cumbria MDU - HR1',
                     'S/wel& Dud MDU - HS7': 'Sandwell & Dudley MDU - HS7'})
                # Replace values in column 'DUNameGIS'
                routes.DUNameGIS.replace({'IMDM  Lanc&Cumbria': 'IMDM Lancashire & Cumbria'},
                                         inplace=True)
                # Update route names
                update_nr_route_names(routes, route_col_name='Route')

                save_pickle(routes, path_to_pickle, verbose=verbose)

            except Exception as e:
                print("Failed to get \"{}\". {}.".format(table_name, e))
                routes = None

        return routes

    def get_s8data(self, update=False, save_original_as=None, verbose=False):
        """
        Get data of the table 'S8Data'.

        :param update: whether to check on update and proceed to update the package data,
            defaults to ``False``
        :type update: bool
        :param save_original_as: file extension for saving the original data, defaults to ``None``
        :type save_original_as: str or None
        :param verbose: whether to print relevant information in console as the function runs,
            defaults to ``False``
        :type verbose: bool or int
        :return: data of the table 'S8Data'
        :rtype: pandas.DataFrame or None

        **Test**::

            >>> from preprocessor import Vegetation
            
            >>> veg = Vegetation()

            >>> s8data_tbl = veg.get_s8data(update=True, verbose=True)
            Updating "S8Data.pickle" at "data\\vegetation\\database\\tables" ... Done.

            >>> s8data_tbl.tail()
                        IncidentNumber           StartDate  ...       Cost                 Route
            IncidentID                                      ...
            6323113             437231 2014-03-28 14:46:00  ...  125086.73  London North Western
            6323221             437608 2014-03-28 17:52:00  ...     136.28  London North Western
            6323807             439651 2014-03-29 17:00:00  ...     310.26              Scotland
            6324514             442500 2014-03-31 09:57:00  ...     193.92              Scotland
            6324647             442184 2014-03-31 02:39:00  ...    1500.24               Western
            [5 rows x 14 columns]
        """

        table_name = 'S8Data'
        path_to_pickle = self.cdd_tables(table_name + ".pickle")

        if os.path.isfile(path_to_pickle) and not update:
            s8data = load_pickle(path_to_pickle)

        else:
            try:
                s8data = self.read_table(table_name, index_col=self.get_primary_key(table_name),
                                         save_as=save_original_as,
                                         update=update)

                update_nr_route_names(s8data, route_col_name='Route')

                save_pickle(s8data, path_to_pickle, verbose=verbose)

            except Exception as e:
                print("Failed to get \"{}\". {}.".format(table_name, e))
                s8data = None

        return s8data

    def get_tree_age_class(self, update=False, save_original_as=None, verbose=False):
        """
        Get data of the table 'TreeAgeClass'.

        :param update: whether to check on update and proceed to update the package data,
            defaults to ``False``
        :type update: bool
        :param save_original_as: file extension for saving the original data, defaults to ``None``
        :type save_original_as: str or None
        :param verbose: whether to print relevant information in console as the function runs,
            defaults to ``False``
        :type verbose: bool or int
        :return: data of the table 'TreeAgeClass'
        :rtype: pandas.DataFrame or None

        **Test**::

            >>> from preprocessor import Vegetation
            
            >>> veg = Vegetation()

            >>> tree_age_class_tbl = veg.get_tree_age_class(update=True, verbose=True)
            Updating "TreeAgeClass.pickle" at "data\\vegetation\\database\\tables" ... Done.

            >>> tree_age_class_tbl.tail()
        """

        table_name = 'TreeAgeClass'
        path_to_pickle = self.cdd_tables(table_name + ".pickle")

        if os.path.isfile(path_to_pickle) and not update:
            tree_age_class = load_pickle(path_to_pickle)

        else:
            try:
                tree_age_class = self.read_table(table_name, index_col=self.get_primary_key(table_name),
                                                 save_as=save_original_as, update=update)

                save_pickle(tree_age_class, path_to_pickle, verbose=verbose)

            except Exception as e:
                print("Failed to get \"{}\". {}.".format(table_name, e))
                tree_age_class = None

        return tree_age_class

    def get_tree_size_class(self, update=False, save_original_as=None, verbose=False):
        """
        Get data of the table 'TreeSizeClass'.

        :param update: whether to check on update and proceed to update the package data,
            defaults to ``False``
        :type update: bool
        :param save_original_as: file extension for saving the original data, defaults to ``None``
        :type save_original_as: str or None
        :param verbose: whether to print relevant information in console as the function runs,
            defaults to ``False``
        :type verbose: bool or int
        :return: data of the table 'TreeSizeClass'
        :rtype: pandas.DataFrame or None

        **Test**::

            >>> from preprocessor import Vegetation
            
            >>> veg = Vegetation()

            >>> tree_size_class_tbl = veg.get_tree_size_class(update=True, verbose=True)
            Updating "TreeSizeClass.pickle" at "data\\vegetation\\database\\tables" ... Done.

            >>> tree_size_class_tbl.tail()
        """

        table_name = 'TreeSizeClass'
        path_to_pickle = self.cdd_tables(table_name + ".pickle")

        if os.path.isfile(path_to_pickle) and not update:
            tree_size_class = load_pickle(path_to_pickle)

        else:
            try:
                tree_size_class = self.read_table(table_name, index_col=self.get_primary_key(table_name),
                                                  save_as=save_original_as, update=update)

                save_pickle(tree_size_class, path_to_pickle, verbose=verbose)

            except Exception as e:
                print("Failed to get \"{}\". {}.".format(table_name, e))
                tree_size_class = None

        return tree_size_class

    def get_tree_type(self, update=False, save_original_as=None, verbose=False):
        """
        Get data of the table 'TreeType'.

        :param update: whether to check on update and proceed to update the package data,
            defaults to ``False``
        :type update: bool
        :param save_original_as: file extension for saving the original data, defaults to ``None``
        :type save_original_as: str or None
        :param verbose: whether to print relevant information in console as the function runs,
            defaults to ``False``
        :type verbose: bool or int
        :return: data of the table 'TreeType'
        :rtype: pandas.DataFrame or None

        **Test**::

            >>> from preprocessor import Vegetation
            
            >>> veg = Vegetation()

            >>> tree_type_tbl = veg.get_tree_type(update=True, verbose=True)
            Updating "TreeType.pickle" at "data\\vegetation\\database\\tables" ... Done.

            >>> tree_type_tbl.tail()
        """

        table_name = 'TreeType'
        path_to_pickle = self.cdd_tables(table_name + ".pickle")

        if os.path.isfile(path_to_pickle) and not update:
            tree_type = load_pickle(path_to_pickle)

        else:
            try:
                tree_type = self.read_table(table_name, index_col=self.get_primary_key(table_name),
                                            save_as=save_original_as,
                                            update=update)

                save_pickle(tree_type, path_to_pickle, verbose=verbose)

            except Exception as e:
                print("Failed to get \"{}\". {}.".format(table_name, e))
                tree_type = None

        return tree_type

    def get_felling_type(self, update=False, save_original_as=None, verbose=False):
        """
        Get data of the table 'FellingType'.

        :param update: whether to check on update and proceed to update the package data,
            defaults to ``False``
        :type update: bool
        :param save_original_as: file extension for saving the original data, defaults to ``None``
        :type save_original_as: str or None
        :param verbose: whether to print relevant information in console as the function runs,
            defaults to ``False``
        :type verbose: bool or int
        :return: data of the table 'FellingType'
        :rtype: pandas.DataFrame or None

        **Test**::

            >>> from preprocessor import Vegetation
            
            >>> veg = Vegetation()

            >>> felling_type_tbl = veg.get_felling_type(update=True, verbose=True)
            Updating "FellingType.pickle" at "data\\vegetation\\database\\tables" ... Done.

            >>> felling_type_tbl.tail()
        """

        table_name = 'FellingType'
        path_to_pickle = self.cdd_tables(table_name + ".pickle")

        if os.path.isfile(path_to_pickle) and not update:
            felling_type = load_pickle(path_to_pickle)

        else:
            try:
                felling_type = self.read_table(table_name, index_col=self.get_primary_key(table_name),
                                               save_as=save_original_as,
                                               update=update)

                save_pickle(felling_type, path_to_pickle, verbose=verbose)

            except Exception as e:
                print("Failed to get \"{}\". {}.".format(table_name, e))
                felling_type = None

        return felling_type

    def get_area_work_type(self, update=False, save_original_as=None, verbose=False):
        """
        Get data of the table 'AreaWorkType'.

        :param update: whether to check on update and proceed to update the package data,
            defaults to ``False``
        :type update: bool
        :param save_original_as: file extension for saving the original data, defaults to ``None``
        :type save_original_as: str or None
        :param verbose: whether to print relevant information in console as the function runs,
            defaults to ``False``
        :type verbose: bool or int
        :return: data of the table 'AreaWorkType'
        :rtype: pandas.DataFrame or None

        **Test**::

            >>> from preprocessor import Vegetation
            
            >>> veg = Vegetation()

            >>> area_work_type_tbl = veg.get_area_work_type(update=True, verbose=True)
            Updating "AreaWorkType.pickle" at "data\\vegetation\\database\\tables" ... Done.

            >>> area_work_type_tbl.tail()
        """

        table_name = 'AreaWorkType'
        path_to_pickle = self.cdd_tables(table_name + ".pickle")

        if os.path.isfile(path_to_pickle) and not update:
            area_work_type = load_pickle(path_to_pickle)

        else:
            try:
                area_work_type = self.read_table(table_name,
                                                 index_col=self.get_primary_key('AreaWorkType'),
                                                 save_as=save_original_as, update=update)

                save_pickle(area_work_type, path_to_pickle, verbose=verbose)

            except Exception as e:
                print("Failed to get \"{}\". {}.".format(table_name, e))
                area_work_type = None

        return area_work_type

    def get_service_detail(self, update=False, save_original_as=None, verbose=False):
        """
        Get data of the table 'ServiceDetail'.

        :param update: whether to check on update and proceed to update the package data,
            defaults to ``False``
        :type update: bool
        :param save_original_as: file extension for saving the original data, defaults to ``None``
        :type save_original_as: str or None
        :param verbose: whether to print relevant information in console as the function runs,
            defaults to ``False``
        :type verbose: bool or int
        :return: data of the table 'ServiceDetail'
        :rtype: pandas.DataFrame or None

        **Test**::

            >>> from preprocessor import Vegetation
            
            >>> veg = Vegetation()

            >>> service_detail_tbl = veg.get_service_detail(update=True, verbose=True)
            Updating "ServiceDetail.pickle" at "data\\vegetation\\database\\tables" ... Done.

            >>> service_detail_tbl.tail()
                     ServiceGroup TOC OrigTiploc  ...   STP  DangerousGoods  Comments
            Service                               ...
            50313768     57620170  DB    YORKYSF  ...  True           False     False
            50313769     57620170  DB    YORKYSF  ...  True           False     False
            50313770     57620170  DB    YORKYSF  ...  True           False     False
            50313771     57620170  DB    YORKYSF  ...  True           False     False
            50313772     72030304  WB    YOVILJN  ...  True           False     False
            [5 rows x 8 columns]
        """

        table_name = 'ServiceDetail'
        path_to_pickle = self.cdd_tables(table_name + ".pickle")

        if os.path.isfile(path_to_pickle) and not update:
            service_detail = load_pickle(path_to_pickle)

        else:
            try:
                service_detail = self.read_table(table_name,
                                                 index_col=self.get_primary_key('ServiceDetail'),
                                                 save_as=save_original_as, update=update)

                save_pickle(service_detail, path_to_pickle, verbose=verbose)

            except Exception as e:
                print("Failed to get \"{}\". {}.".format(table_name, e))
                service_detail = None

        return service_detail

    def get_service_path(self, update=False, save_original_as=None, verbose=False):
        """
        Get data of the table 'ServicePath'.

        :param update: whether to check on update and proceed to update the package data,
            defaults to ``False``
        :type update: bool
        :param save_original_as: file extension for saving the original data, defaults to ``None``
        :type save_original_as: str or None
        :param verbose: whether to print relevant information in console as the function runs,
            defaults to ``False``
        :type verbose: bool or int
        :return: data of the table 'ServicePath'
        :rtype: pandas.DataFrame or None

        **Test**::

            >>> from preprocessor import Vegetation
            
            >>> veg = Vegetation()

            >>> service_path_tbl = veg.get_service_path(update=True, verbose=True)
            Updating "ServicePath.pickle" at "data\\vegetation\\database\\tables" ... Done.

            >>> service_path_tbl.tail()
                                        BytConfLvl
            Service  Seq ZoneID Path
            50313772 90  SE     552428           7
                     100 SE     551164           7
                     110 SE     500955           7
                     120 SE     503959           7
                     130 SE     531253           7
        """

        table_name = 'ServicePath'
        path_to_pickle = self.cdd_tables(table_name + ".pickle")

        if os.path.isfile(path_to_pickle) and not update:
            service_path = load_pickle(path_to_pickle)

        else:
            try:
                service_path = self.read_table(table_name, index_col=self.get_primary_key(table_name),
                                               save_as=save_original_as,
                                               update=update)

                save_pickle(service_path, path_to_pickle, verbose=verbose)

            except Exception as e:
                print("Failed to get \"{}\". {}.".format(table_name, e))
                service_path = None

        return service_path

    def get_supplier(self, update=False, save_original_as=None, verbose=False):
        """
        Get data of the table 'Supplier'.

        :param update: whether to check on update and proceed to update the package data,
            defaults to ``False``
        :type update: bool
        :param save_original_as: file extension for saving the original data, defaults to ``None``
        :type save_original_as: str or None
        :param verbose: whether to print relevant information in console as the function runs,
            defaults to ``False``
        :type verbose: bool or int
        :return: data of the table 'Supplier'
        :rtype: pandas.DataFrame or None

        **Test**::

            >>> from preprocessor import Vegetation
            
            >>> veg = Vegetation()

            >>> supplier_tbl = veg.get_supplier(update=True, verbose=True)
            Updating "Supplier.pickle" at "data\\vegetation\\database\\tables" ... Done.

            >>> supplier_tbl.tail()
        """

        table_name = 'Supplier'
        path_to_pickle = self.cdd_tables(table_name + ".pickle")

        if os.path.isfile(path_to_pickle) and not update:
            supplier = load_pickle(path_to_pickle)

        else:
            try:
                supplier = self.read_table(table_name, index_col=self.get_primary_key(table_name),
                                           save_as=save_original_as,
                                           update=update)

                save_pickle(supplier, path_to_pickle, verbose=verbose)

            except Exception as e:
                print("Failed to get \"{}\". {}.".format(table_name, e))
                supplier = None

        return supplier

    def get_supplier_costs(self, update=False, save_original_as=None, verbose=False):
        """
        Get data of the table 'SupplierCosts'.

        :param update: whether to check on update and proceed to update the package data,
            defaults to ``False``
        :type update: bool
        :param save_original_as: file extension for saving the original data, defaults to ``None``
        :type save_original_as: str or None
        :param verbose: whether to print relevant information in console as the function runs,
            defaults to ``False``
        :type verbose: bool or int
        :return: data of the table 'SupplierCosts'
        :rtype: pandas.DataFrame or None

        **Test**::

            >>> from preprocessor import Vegetation
            
            >>> veg = Vegetation()

            >>> supplier_costs_tbl = veg.get_supplier_costs(update=True, verbose=True)
            Updating "SupplierCosts.pickle" at "data\\vegetation\\database\\tables" ... Done.

            >>> supplier_costs_tbl.tail()
        """

        table_name = 'SupplierCosts'
        path_to_pickle = self.cdd_tables(table_name + ".pickle")

        if os.path.isfile(path_to_pickle) and not update:
            supplier_costs = load_pickle(path_to_pickle)

        else:
            try:
                supplier_costs = self.read_table(table_name, index_col=None, save_as=save_original_as,
                                                 update=update)

                update_nr_route_names(supplier_costs, route_col_name='Route')
                supplier_costs.set_index(self.get_primary_key(table_name), inplace=True)

                save_pickle(supplier_costs, path_to_pickle, verbose=verbose)

            except Exception as e:
                print("Failed to get \"{}\". {}.".format(table_name, e))
                supplier_costs = None

        return supplier_costs

    def get_supplier_costs_area(self, update=False, save_original_as=None, verbose=False):
        """
        Get data of the table 'SupplierCostsArea'.

        :param update: whether to check on update and proceed to update the package data,
            defaults to ``False``
        :type update: bool
        :param save_original_as: file extension for saving the original data, defaults to ``None``
        :type save_original_as: str or None
        :param verbose: whether to print relevant information in console as the function runs,
            defaults to ``False``
        :type verbose: bool or int
        :return: data of the table 'SupplierCostsArea'
        :rtype: pandas.DataFrame or None

        **Test**::

            >>> from preprocessor import Vegetation
            
            >>> veg = Vegetation()

            >>> costs_area_tbl = veg.get_supplier_costs_area(update=True, verbose=True)
            Updating "SupplierCostsArea.pickle" at "data\\vegetation\\database\\tables" ... Done.

            >>> costs_area_tbl.tail()
        """

        table_name = 'SupplierCostsArea'
        path_to_pickle = self.cdd_tables(table_name + ".pickle")

        if os.path.isfile(path_to_pickle) and not update:
            costs_area = load_pickle(path_to_pickle)

        else:
            try:
                costs_area = self.read_table(table_name, index_col=None, save_as=save_original_as,
                                             update=update)

                update_nr_route_names(costs_area, route_col_name='Route')
                costs_area.set_index(self.get_primary_key(table_name), inplace=True)

                save_pickle(costs_area, path_to_pickle, verbose=verbose)

            except Exception as e:
                print("Failed to get \"{}\". {}.".format(table_name, e))
                costs_area = None

        return costs_area

    def get_supplier_cost_simple(self, update=False, save_original_as=None, verbose=False):
        """
        Get data of the table 'SupplierCostsSimple'.

        :param update: whether to check on update and proceed to update the package data,
            defaults to ``False``
        :type update: bool
        :param save_original_as: file extension for saving the original data, defaults to ``None``
        :type save_original_as: str or None
        :param verbose: whether to print relevant information in console as the function runs,
            defaults to ``False``
        :type verbose: bool or int
        :return: data of the table 'SupplierCostsSimple'
        :rtype: pandas.DataFrame or None

        **Test**::

            >>> from preprocessor import Vegetation
            
            >>> veg = Vegetation()

            >>> costs_simple_tbl = veg.get_supplier_cost_simple(update=True, verbose=True)
            Updating "SupplierCostsSimple.pickle" at "data\\vegetation\\database\\tables" ... Done.

            >>> costs_simple_tbl.tail()
        """

        table_name = 'SupplierCostsSimple'
        path_to_pickle = self.cdd_tables(table_name + ".pickle")

        if os.path.isfile(path_to_pickle) and not update:
            costs_simple = load_pickle(path_to_pickle)

        else:
            try:
                costs_simple = self.read_table(table_name, index_col=None, save_as=save_original_as,
                                               update=update)

                update_nr_route_names(costs_simple, route_col_name='Route')
                costs_simple.set_index(self.get_primary_key(table_name), inplace=True)

                save_pickle(costs_simple, path_to_pickle, verbose=verbose)

            except Exception as e:
                print("Failed to get \"{}\". {}.".format(table_name, e))
                costs_simple = None

        return costs_simple

    def get_tree_action_fractions(self, update=False, save_original_as=None, verbose=False):
        """
        Get data of the table 'TreeActionFractions'.

        :param update: whether to check on update and proceed to update the package data,
            defaults to ``False``
        :type update: bool
        :param save_original_as: file extension for saving the original data, defaults to ``None``
        :type save_original_as: str or None
        :param verbose: whether to print relevant information in console as the function runs,
            defaults to ``False``
        :type verbose: bool or int
        :return: data of the table 'TreeActionFractions'
        :rtype: pandas.DataFrame or None

        **Test**::

            >>> from preprocessor import Vegetation
            
            >>> veg = Vegetation()

            >>> tree_action_fractions_tbl = veg.get_tree_action_fractions(update=True, verbose=True)
            Updating "TreeActionFractions.pickle" at "data\\vegetation\\database\\tables" ... Done.

            >>> tree_action_fractions_tbl.tail()
        """

        table_name = 'TreeActionFractions'
        path_to_pickle = self.cdd_tables(table_name + ".pickle")

        if os.path.isfile(path_to_pickle) and not update:
            tree_action_fractions = load_pickle(path_to_pickle)

        else:
            try:
                tree_action_fractions = self.read_table(table_name,
                                                        index_col=self.get_primary_key(table_name),
                                                        save_as=save_original_as, update=update)

                save_pickle(tree_action_fractions, path_to_pickle, verbose=verbose)

            except Exception as e:
                print("Failed to get \"{}\". {}.".format(table_name, e))
                tree_action_fractions = None

        return tree_action_fractions

    def get_veg_surv_type_class(self, update=False, save_original_as=None, verbose=False):
        """
        Get data of the table 'VegSurvTypeClass'.

        :param update: whether to check on update and proceed to update the package data,
            defaults to ``False``
        :type update: bool
        :param save_original_as: file extension for saving the original data, defaults to ``None``
        :type save_original_as: str or None
        :param verbose: whether to print relevant information in console as the function runs,
            defaults to ``False``
        :type verbose: bool or int
        :return: data of the table 'VegSurvTypeClass'
        :rtype: pandas.DataFrame or None

        **Test**::

            >>> from preprocessor import Vegetation
            
            >>> veg = Vegetation()

            >>> veg_surv_type_class_tbl = veg.get_veg_surv_type_class(update=True, verbose=True)
            Updating "VegSurvTypeClass.pickle" at "data\\vegetation\\database\\tables" ... Done.

            >>> veg_surv_type_class_tbl.tail()
        """

        table_name = 'VegSurvTypeClass'
        path_to_pickle = self.cdd_tables(table_name + ".pickle")

        if os.path.isfile(path_to_pickle) and not update:
            veg_surv_type_class = load_pickle(path_to_pickle)

        else:
            try:
                veg_surv_type_class = self.read_table(table_name,
                                                      index_col=self.get_primary_key(table_name),
                                                      save_as=save_original_as, update=update)

                save_pickle(veg_surv_type_class, path_to_pickle, verbose=verbose)

            except Exception as e:
                print("Failed to get \"{}\". {}.".format(table_name, e))
                veg_surv_type_class = None

        return veg_surv_type_class

    def get_wb_factors(self, update=False, save_original_as=None, verbose=False):
        """
        Get data of the table 'WBFactors'.

        :param update: whether to check on update and proceed to update the package data,
            defaults to ``False``
        :type update: bool
        :param save_original_as: file extension for saving the original data, defaults to ``None``
        :type save_original_as: str or None
        :param verbose: whether to print relevant information in console as the function runs,
            defaults to ``False``
        :type verbose: bool or int
        :return: data of the table 'WBFactors'
        :rtype: pandas.DataFrame or None

        **Test**::

            >>> from preprocessor import Vegetation
            
            >>> veg = Vegetation()

            >>> wb_factors_tbl = veg.get_wb_factors(update=True, verbose=True)
            Updating "WBFactors.pickle" at "data\\vegetation\\database\\tables" ... Done.

            >>> wb_factors_tbl.tail()
        """

        table_name = 'WBFactors'
        path_to_pickle = self.cdd_tables(table_name + ".pickle")

        if os.path.isfile(path_to_pickle) and not update:
            wb_factors = load_pickle(path_to_pickle)

        else:
            try:
                wb_factors = self.read_table(table_name, index_col=self.get_primary_key(table_name),
                                             save_as=save_original_as,
                                             update=update)

                save_pickle(wb_factors, path_to_pickle, verbose=verbose)

            except Exception as e:
                print("Failed to get \"{}\". {}.".format(table_name, e))
                wb_factors = None

        return wb_factors

    def get_weed_spray(self, update=False, save_original_as=None, verbose=False):
        """
        Get data of the table 'Weedspray'.

        :param update: whether to check on update and proceed to update the package data,
            defaults to ``False``
        :type update: bool
        :param save_original_as: file extension for saving the original data, defaults to ``None``
        :type save_original_as: str or None
        :param verbose: whether to print relevant information in console as the function runs,
            defaults to ``False``
        :type verbose: bool or int
        :return: data of the table 'Weedspray'
        :rtype: pandas.DataFrame or None

        **Test**::

            >>> from preprocessor import Vegetation
            
            >>> veg = Vegetation()

            >>> weed_spray_tbl = veg.get_weed_spray(update=True, verbose=True)
            Updating "Weedspray.pickle" at "data\\vegetation\\database\\tables" ... Done.

            >>> weed_spray_tbl.tail()
        """

        table_name = 'Weedspray'
        path_to_pickle = self.cdd_tables(table_name + ".pickle")

        if os.path.isfile(path_to_pickle) and not update:
            weed_spray = load_pickle(path_to_pickle)

        else:
            try:
                weed_spray = self.read_table(table_name, index_col=None, save_as=save_original_as,
                                             update=update)

                update_nr_route_names(weed_spray, route_col_name='Route')
                weed_spray.set_index('RouteAlias', inplace=True)

                save_pickle(weed_spray, path_to_pickle, verbose=verbose)

            except Exception as e:
                print("Failed to get \"{}\". {}.".format(table_name, e))
                weed_spray = None

        return weed_spray

    def get_work_hours(self, update=False, save_original_as=None, verbose=False):
        """
        Get data of the table 'WorkHours'.

        :param update: whether to check on update and proceed to update the package data,
            defaults to ``False``
        :type update: bool
        :param save_original_as: file extension for saving the original data, defaults to ``None``
        :type save_original_as: str or None
        :param verbose: whether to print relevant information in console as the function runs,
            defaults to ``False``
        :type verbose: bool or int
        :return: data of the table 'WorkHours'
        :rtype: pandas.DataFrame or None

        **Test**::

            >>> from preprocessor import Vegetation
            
            >>> veg = Vegetation()

            >>> work_hours_tbl = veg.get_work_hours(update=True, verbose=True)
            Updating "WorkHours.pickle" at "data\\vegetation\\database\\tables" ... Done.

            >>> work_hours_tbl.tail()
        """

        table_name = 'WorkHours'
        path_to_pickle = self.cdd_tables(table_name + ".pickle")

        if os.path.isfile(path_to_pickle) and not update:
            work_hours = load_pickle(path_to_pickle)

        else:
            try:
                work_hours = self.read_table(table_name, index_col=self.get_primary_key(table_name),
                                             save_as=save_original_as,
                                             update=update)

                save_pickle(work_hours, path_to_pickle, verbose=verbose)

            except Exception as e:
                print("Failed to get \"{}\". {}.".format(table_name, e))
                work_hours = None

        return work_hours

    def get_furlong_data(self, set_index=False, pseudo_amendment=True, update=False,
                         save_original_as=None, verbose=False):
        """
        Get data of the table 'FurlongData'.

        :param set_index: whether to set an index column, defaults to ``False``
        :type set_index: bool
        :param pseudo_amendment: whether to make an amendment with external data,
            defaults to ``True``
        :type pseudo_amendment: bool
        :param update: whether to check on update and proceed to update the package data,
            defaults to ``False``
        :type update: bool
        :param save_original_as: file extension for saving the original data, defaults to ``None``
        :type save_original_as: str or None
        :param verbose: whether to print relevant information in console as the function runs,
            defaults to ``False``
        :type verbose: bool or int
        :return: data of the table 'FurlongData'
        :rtype: pandas.DataFrame or None

        .. note::

            Equipment Class: VL ('VEGETATION - 1/8 MILE SECTION')
            1/8 mile = 220 yards = 1 furlong

        **Test**::

            >>> from preprocessor import Vegetation
            
            >>> veg = Vegetation()

            >>> furlong_data_tbl = veg.get_furlong_data(update=True, verbose=True)
            Updating "FurlongData.pickle" at "data\\vegetation\\database\\tables" ... Done.

            >>> furlong_data_tbl.tail()
                   ELR         DU RouteAlias  ...  CuttingDepth FurlongID           Route
            75242  YMS  IMDM York        LNE  ...             3      9759  North and East
            75243  YMS  IMDM York        LNE  ...             3      9760  North and East
            75244  YMS  IMDM York        LNE  ...             2     21842  North and East
            75245  YMS  IMDM York        LNE  ...             3     30447  North and East
            75246  YMS  IMDM York        LNE  ...             2     14098  North and East
            [5 rows x 40 columns]
        """

        table_name = 'FurlongData'
        path_to_pickle = self.cdd_tables(table_name + ("-indexed" if set_index else "") + ".pickle")

        if os.path.isfile(path_to_pickle) and not update:
            furlong_data = load_pickle(path_to_pickle)

        else:
            try:
                furlong_data = self.read_table(table_name, index_col=None, coerce_float=False,
                                               save_as=save_original_as,
                                               update=update)

                # Re-format mileage data
                furlong_data[['StartMileage', 'EndMileage']] = furlong_data[
                    ['StartMileage', 'EndMileage']].applymap(nr_mileage_num_to_str)

                # Rename columns
                renamed_cols_dict = {
                    'TEF307601': 'MainSpeciesScore',
                    'TEF307602': 'TreeSizeScore',
                    'TEF307603': 'SurroundingLandScore',
                    'TEF307604': 'DistanceFromRailScore',
                    'TEF307605': 'OtherVegScore',
                    'TEF307606': 'TopographyScore',
                    'TEF307607': 'AtmosphereScore',
                    'TEF307608': 'TreeDensityScore'}
                furlong_data.rename(columns=renamed_cols_dict, inplace=True)
                # Edit the 'TEF' columns
                furlong_data.OtherVegScore.replace({-1: 0}, inplace=True)
                renamed_cols = list(renamed_cols_dict.values())
                furlong_data[renamed_cols] = furlong_data[renamed_cols].applymap(
                    lambda x: 0 if np.isnan(x) else x + 1)
                # Re-format date of measure
                furlong_data.DateOfMeasure = furlong_data.DateOfMeasure.map(
                    lambda x: datetime.datetime.strptime(x, '%d/%m/%Y %H:%M'))
                # Edit route data
                update_nr_route_names(furlong_data, route_col_name='Route')

                if set_index:
                    furlong_data.set_index(self.get_primary_key(table_name), inplace=True)

                # Make amendment to "CoverPercent" data for which the total is not 0 or 100?
                if pseudo_amendment:
                    # Find columns relating to "CoverPercent..."
                    cp_cols = [x for x in furlong_data.columns if re.match('^CoverPercent[A-Z]', x)]

                    temp = furlong_data[cp_cols].sum(1)
                    if not temp.empty:

                        # For all zero 'CoverPercent...'
                        cpo_col = 'CoverPercentOther'
                        furlong_data.loc[temp[temp == 0].index, cpo_col] = 100.0

                        # For all non-100 'CoverPercent...'
                        idx = temp[~temp.isin([0.0, 100.0])].index

                        nonzero_cols = furlong_data.loc[idx, cp_cols].apply(lambda x: x != 0.0).apply(
                            lambda x: list(pd.Index(cp_cols)[x.values]), axis=1)

                        errors = pd.Series(100.0 - temp[idx])

                        for i in idx:
                            features = nonzero_cols[i].copy()
                            if len(features) == 1:
                                feature = features[0]
                                if feature == cpo_col:
                                    furlong_data.loc[[i], cpo_col] = 100.0
                                else:
                                    if errors.loc[i] > 0:
                                        furlong_data.loc[[i], cpo_col] = np.sum([
                                            furlong_data.loc[i, cpo_col], errors.loc[i]])
                                    else:  # errors.loc[i] < 0
                                        furlong_data[feature].loc[[i]] = np.sum([
                                            furlong_data[feature].loc[i], errors.loc[i]])
                            else:  # len(nonzero_cols[i]) > 1
                                if cpo_col in features:
                                    err = np.sum([furlong_data.loc[i, cpo_col], errors.loc[i]])
                                    if err >= 0.0:
                                        furlong_data.loc[[i], cpo_col] = err
                                    else:
                                        features.remove(cpo_col)
                                        furlong_data.loc[[i], cpo_col] = 0.0
                                        if len(features) == 1:
                                            feature = features[0]
                                            furlong_data.loc[[i], feature] = np.sum(
                                                [furlong_data.loc[i, feature], err])
                                        else:
                                            err = np.divide(err, len(features))
                                            furlong_data.loc[i, features] += err
                                else:
                                    err = np.divide(errors.loc[i], len(features))
                                    furlong_data.loc[i, features] += err

                save_pickle(furlong_data, path_to_pickle, verbose=verbose)

            except Exception as e:
                print("Failed to get \"{}\". {}.".format(table_name, e))
                furlong_data = None

        return furlong_data

    def get_furlong_location(self, relevant_columns_only=True, update=False, save_original_as=None,
                             verbose=False):
        """
        Get data of the table 'FurlongLocation'.

        :param relevant_columns_only: whether to return only the columns relevant to the project,
            defaults to ``True``
        :type relevant_columns_only: bool
        :param update: whether to check on update and proceed to update the package data,
            defaults to ``False``
        :type update: bool
        :param save_original_as: file extension for saving the original data, defaults to ``None``
        :type save_original_as: str or None
        :param verbose: whether to print relevant information in console as the function runs,
            defaults to ``False``
        :type verbose: bool or int
        :return: data of the table 'FurlongLocation'
        :rtype: pandas.DataFrame or None

        .. note::

            One set of ELR and mileage may have multiple 'FurlongID's.

        **Test**::

            >>> from preprocessor import Vegetation
            
            >>> veg = Vegetation()

            >>> furlong_location_tbl = veg.get_furlong_location(update=True, verbose=True)
            Updating "FurlongLocation-cut.pickle" at "data\\vegetation\\database\\tables" ... Done.

            >>> furlong_location_tbl.tail()
                        Route RouteAlias           DU  ... EndMileage Electrified HazardOnly
            FurlongID                                  ...
            77013      Wessex     Wessex  IMDM Woking  ...   723.0660           0          1
            77014      Wessex     Wessex  IMDM Woking  ...   723.0660           0          1
            77015      Wessex     Wessex  IMDM Woking  ...    33.0146           0          1
            77016      Wessex     Wessex  IMDM Woking  ...    33.0170           0          1
            77017      Wessex     Wessex  IMDM Woking  ...    33.0170           0          1
            [5 rows x 8 columns]
        """

        table_name = 'FurlongLocation'
        path_to_pickle = self.cdd_tables(
            table_name + ("-cut" if relevant_columns_only else "") + ".pickle")

        if os.path.isfile(path_to_pickle) and not update:
            furlong_location = load_pickle(path_to_pickle)

        else:
            try:
                # Read data from database
                furlong_location = self.read_table(table_name,
                                                   index_col=self.get_primary_key(table_name),
                                                   save_as=save_original_as, update=update)
                # Re-format mileage data
                furlong_location[['StartMileage', 'EndMileage']] = \
                    furlong_location[['StartMileage', 'EndMileage']].applymap(nr_mileage_num_to_str)

                # Replace boolean values with binary values
                furlong_location[['Electrified', 'HazardOnly']] = \
                    furlong_location[['Electrified', 'HazardOnly']].applymap(int)
                # Replace Route names
                update_nr_route_names(furlong_location, route_col_name='Route')

                # Select useful columns only?
                if relevant_columns_only:
                    furlong_location = furlong_location[
                        ['Route', 'RouteAlias', 'DU', 'ELR', 'StartMileage', 'EndMileage',
                         'Electrified', 'HazardOnly']]

                save_pickle(furlong_location, path_to_pickle, verbose=verbose)

            except Exception as e:
                print("Failed to get \"{}\". {}.".format(table_name, e))
                furlong_location = None

        return furlong_location

    def get_hazard_tree(self, set_index=False, update=False, save_original_as=None, verbose=False):
        """
        Get data of the table 'HazardTree'.

        :param set_index: whether to set an index column, defaults to ``False``
        :type set_index: bool
        :param update: whether to check on update and proceed to update the package data,
            defaults to ``False``
        :type update: bool
        :param save_original_as: file extension for saving the original data, defaults to ``None``
        :type save_original_as: str or None
        :param verbose: whether to print relevant information in console as the function runs,
            defaults to ``False``
        :type verbose: bool or int
        :return: data of the table 'HazardTree'
        :rtype: pandas.DataFrame or None

        .. note::

            Error data exists in 'FurlongID'. They could be cancelled out when the 'hazard_tree' data
            set is merged with other data sets on the 'FurlongID'. Errors also exist in 'Easting'
            and 'Northing' columns.

        **Test**::

            >>> from preprocessor import Vegetation
            
            >>> veg = Vegetation()

            >>> hazard_tree_tbl = veg.get_hazard_tree(update=True, verbose=True)
            Updating "HazardTree.pickle" at "data\\vegetation\\database\\tables" ... Done.

            >>> hazard_tree_tbl = veg.get_hazard_tree(set_index=True, update=True, verbose=True)
            Updating "HazardTree-indexed.pickle" at "data\\vegetation\\database\\tables" ... Done.

        """

        table_name = 'HazardTree'
        path_to_pickle = self.cdd_tables(table_name + ("-indexed" if set_index else "") + ".pickle")

        if os.path.isfile(path_to_pickle) and not update:
            hazard_tree = load_pickle(path_to_pickle)

        else:
            try:
                hazard_tree = self.read_table(table_name, index_col=None, save_as=save_original_as,
                                              update=update)
                # Re-format mileage data
                hazard_tree.Mileage = hazard_tree.Mileage.apply(nr_mileage_num_to_str)

                # Edit the original data
                hazard_tree.drop(['Treesurvey', 'Treetunnel'], axis=1, inplace=True)
                hazard_tree.dropna(subset=['Northing', 'Easting'], inplace=True)
                hazard_tree.Treespecies.replace({'': 'No data'}, inplace=True)

                # Update route data
                update_nr_route_names(hazard_tree, route_col_name='Route')

                # Integrate information from several features in a DataFrame
                def sum_up_selected_features(data, selected_features, new_feature):
                    """
                    :param data: original data frame
                    :type data: pandas.DataFrame
                    :param selected_features: list of columns names
                    :type selected_features: list
                    :param new_feature: new column name
                    :type new_feature: str
                    :return: integrated data
                    :rtype: pandas.DataFrame
                    """
                    data.replace({True: 1, False: 0}, inplace=True)
                    data[new_feature] = data[selected_features].fillna(0).apply(sum, axis=1)
                    data.drop(selected_features, axis=1, inplace=True)

                # Integrate TEF: Failure scores
                failure_scores = ['TEF30770' + str(i) for i in range(1, 6)]
                sum_up_selected_features(hazard_tree, failure_scores, new_feature='Failure_Score')
                # Integrate TEF: Target scores
                target_scores = ['TEF3077%02d' % i for i in range(6, 12)]
                sum_up_selected_features(hazard_tree, target_scores, new_feature='Target_Score')
                # Integrate TEF: Impact scores
                impact_scores = ['TEF3077' + str(i) for i in range(12, 16)]
                sum_up_selected_features(hazard_tree, impact_scores, new_feature='Impact_Score')
                # Rename the rest of TEF
                work_req = ['TEF3077' + str(i) for i in range(17, 27)]
                work_req_desc = [
                    'WorkReq_ExpertInspection',
                    'WorkReq_LocalisedPruning',
                    'WorkReq_GeneralPruning',
                    'WorkReq_CrownRemoval',
                    'WorkReq_StumpRemoval',
                    'WorkReq_TreeRemoval',
                    'WorkReq_TargetManagement',
                    'WorkReq_FurtherInvestigation',
                    'WorkReq_LimbRemoval',
                    'WorkReq_InstallSupport']
                hazard_tree.rename(columns=dict(zip(work_req, work_req_desc)), inplace=True)

                # Note the feasibility of the the following operation is not guaranteed:
                hazard_tree[work_req_desc] = hazard_tree[work_req_desc].fillna(value=0)

                # Rearrange DataFrame index
                hazard_tree.index = range(len(hazard_tree))

                # Add two columns of Latitudes and Longitudes corresponding to the Easting and Northing
                hazard_tree['Longitude'], hazard_tree['Latitude'] = osgb36_to_wgs84(
                    hazard_tree.Easting.values, hazard_tree.Northing.values)

                save_pickle(hazard_tree, path_to_pickle, verbose=verbose)

                if set_index:
                    hazard_tree.set_index(self.get_primary_key(table_name), inplace=True)

            except Exception as e:
                print("Failed to get \"{}\". {}.".format(table_name, e))
                hazard_tree = None

        return hazard_tree

    def update_vegetation_table_pickles(self, update=True, verbose=True):
        """
        Update the local pickle files for all tables.

        :param update: whether to check on update and proceed to update the package data,
            defaults to ``True``
        :type update: bool
        :param verbose: whether to print relevant information in console as the function runs,
            defaults to ``True``
        :type verbose: bool or int

        **Test**::

            >>> from preprocessor import Vegetation
            
            >>> veg = Vegetation()

            >>> veg.update_vegetation_table_pickles(update=True, verbose=True)
        """

        if confirmed("To update the pickles of the NR_Vegetation Table data?"):

            _ = self.get_adverse_wind(update, verbose=verbose)
            _ = self.get_area_work_type(update, verbose=verbose)
            _ = self.get_cutting_angle_class(update, verbose=verbose)
            _ = self.get_cutting_depth_class(update, verbose=verbose)
            _ = self.get_du_list(index=False, update=update, verbose=verbose)
            _ = self.get_du_list(index=True, update=update, verbose=verbose)
            _ = self.get_felling_type(update, verbose=verbose)

            _ = self.get_furlong_data(set_index=False, pseudo_amendment=True, update=update,
                                      verbose=verbose)
            _ = self.get_furlong_location(relevant_columns_only=False, update=update, verbose=verbose)
            _ = self.get_furlong_location(relevant_columns_only=True, update=update, verbose=verbose)
            _ = self.get_hazard_tree(set_index=False, update=update, verbose=verbose)

            _ = self.get_path_route(update, verbose=verbose)
            _ = self.get_du_route(update, verbose=verbose)
            _ = self.get_s8data(update, verbose=verbose)
            _ = self.get_service_detail(update, verbose=verbose)
            _ = self.get_service_path(update, verbose=verbose)
            _ = self.get_supplier(update, verbose=verbose)
            _ = self.get_supplier_costs(update, verbose=verbose)
            _ = self.get_supplier_costs_area(update, verbose=verbose)
            _ = self.get_supplier_cost_simple(update, verbose=verbose)
            _ = self.get_tree_action_fractions(update, verbose=verbose)
            _ = self.get_tree_age_class(update, verbose=verbose)
            _ = self.get_tree_size_class(update, verbose=verbose)
            _ = self.get_tree_type(update, verbose=verbose)
            _ = self.get_veg_surv_type_class(update, verbose=verbose)
            _ = self.get_wb_factors(update, verbose=verbose)
            _ = self.get_weed_spray(update, verbose=verbose)
            _ = self.get_work_hours(update, verbose=verbose)

            if verbose:
                print("\nUpdate finished.")

    # == Get views based on the NR_Vegetation data ====================================================

    def view_vegetation_coverage_per_furlong(self, route_name=None, update=False, pickle_it=True,
                                             verbose=False):
        """
        Get a view of data of vegetation coverage per furlong (75247, 45).

        :param route_name: name of a Route; if ``None`` (default), all Routes
        :type route_name: str or None
        :param update: whether to check on update and proceed to update the package data,
            defaults to ``True``
        :type update: bool
        :param pickle_it: whether to save the view as a pickle file, defaults to ``True``
        :type pickle_it: bool
        :param verbose: whether to print relevant information in console as the function runs,
            defaults to ``False``
        :type verbose: bool or int
        :return: data of vegetation coverage per furlong
        :rtype: pandas.DataFrame or None

        **Test**::

            >>> from preprocessor import Vegetation
            
            >>> veg = Vegetation()

            >>> fvc = veg.view_vegetation_coverage_per_furlong(update=True, pickle_it=True, verbose=True)
            Updating "vegetation-coverage-per-furlong.pickle" at "data\\vegetation\\...\\views" ... Done.

            >>> fvc.tail()
                   ELR         DU  ... Description_CuttingDepth  TreeNumber
            75242  YMS  IMDM York  ...        NA-Not Applicable           2
            75243  YMS  IMDM York  ...        NA-Not Applicable           1
            75244  YMS  IMDM York  ...        MD-Medium (2-10m)           4
            75245  YMS  IMDM York  ...        NA-Not Applicable           5
            75246  YMS  IMDM York  ...        MD-Medium (2-10m)          66
            [5 rows x 45 columns]
        """

        path_to_pickle = self.cdd_views(make_filename("vegetation-coverage-per-furlong", route_name))

        if os.path.isfile(path_to_pickle) and not update:
            furlong_vegetation_coverage = load_pickle(path_to_pickle)

        else:
            try:
                furlong_data = self.get_furlong_data()  # (75247, 40)
                furlong_location = self.get_furlong_location()  # Set 'FurlongID' to index (77017, 8)
                cutting_angle_class = self.get_cutting_angle_class()  # (5, 1)
                cutting_depth_class = self.get_cutting_depth_class()  # (5, 1)
                # Merge the data that has been obtained
                furlong_vegetation_coverage = furlong_data. \
                    join(furlong_location,  # (75247, 48)
                         on='FurlongID', how='inner', lsuffix='', rsuffix='_FurlongLocation'). \
                    join(cutting_angle_class,  # (75247, 49)
                         on='CuttingAngle', how='inner'). \
                    join(cutting_depth_class,  # (75247, 50)
                         on='CuttingDepth', how='inner', lsuffix='_CuttingAngle',
                         rsuffix='_CuttingDepth')

                if route_name is not None:
                    route_name = find_similar_str(route_name, self.get_du_route().Route)
                    furlong_vegetation_coverage = furlong_vegetation_coverage[
                        furlong_vegetation_coverage.Route == route_name]

                # The total number of trees on both sides
                furlong_vegetation_coverage['TreeNumber'] = \
                    furlong_vegetation_coverage[['TreeNumberUp', 'TreeNumberDown']].sum(1)

                # Edit the merged data
                furlong_vegetation_coverage.drop(
                    labels=[f for f in furlong_vegetation_coverage.columns if
                            re.match('.*_FurlongLocation$', f)],
                    axis=1, inplace=True)  # (75247, 45)

                # Rearrange
                furlong_vegetation_coverage.sort_values(by='StructuredPlantNumber', inplace=True)
                furlong_vegetation_coverage.index = range(len(furlong_vegetation_coverage))

                if pickle_it:
                    save_pickle(furlong_vegetation_coverage, path_to_pickle, verbose=verbose)

            except Exception as e:
                print(
                    "Failed to fetch the information of vegetation coverage per furlong. {}".format(e))
                furlong_vegetation_coverage = None

        return furlong_vegetation_coverage

    def view_hazardous_trees(self, route_name=None, update=False, pickle_it=True, verbose=False):
        """
        get a view of data of hazardous tress (22180, 66)

        :param route_name: name of a Route; if ``None`` (default), all Routes
        :type route_name: str or None
        :param update: whether to check on update and proceed to update the package data,
            defaults to ``True``
        :type update: bool
        :param pickle_it: whether to save the view as a pickle file, defaults to ``True``
        :type pickle_it: bool
        :param verbose: whether to print relevant information in console as the function runs,
            defaults to ``False``
        :type verbose: bool or int
        :return: data of hazardous tress
        :rtype: pandas.DataFrame or None

        **Test**::

            >>> from preprocessor import Vegetation
            
            >>> veg = Vegetation()

            >>> ht = veg.view_hazardous_trees(update=True, pickle_it=True, verbose=True)
            Updating "hazardous-trees.pickle" at "data\\vegetation\\database\\views" ... Done.
            >>> ht.tail()
                               DU   ELR  ... Description_TreeAgeClass  Description_TreeSizeClass
            22175    IMDM Swindon   BFB  ...               VT-Veteran         VL-Very Large (VL)
            22176    IMDM Swindon  MLN1  ...               VT-Veteran         VL-Very Large (VL)
            22177    IMDM Swindon   THA  ...               VT-Veteran         VL-Very Large (VL)
            22178  IMDM Tottenham   BGK  ...               VT-Veteran         VL-Very Large (VL)
            22179     IMDM Woking   BKE  ...               VT-Veteran         VL-Very Large (VL)
            [5 rows x 66 columns]

            >>> ht = veg.view_hazardous_trees('Anglia', update=True, pickle_it=True, verbose=True)
            Updating "hazardous-trees-Anglia.pickle" at "data\\vegetation\\database\\views" ... Done.
            >>> ht.tail()
                              DU   ELR  ... Description_TreeAgeClass  Description_TreeSizeClass
            2483    IMDM Ipswich  WHC2  ...               VT-Veteran         VL-Very Large (VL)
            2484    IMDM Ipswich  WHC2  ...               VT-Veteran         VL-Very Large (VL)
            2485    IMDM Romford  LTN1  ...               VT-Veteran         VL-Very Large (VL)
            2486    IMDM Romford   TLL  ...               VT-Veteran         VL-Very Large (VL)
            2487  IMDM Tottenham   BGK  ...               VT-Veteran         VL-Very Large (VL)
            [5 rows x 66 columns]
        """

        path_to_pickle = self.cdd_views(make_filename("hazardous-trees", route_name))

        if os.path.isfile(path_to_pickle) and not update:
            hazardous_trees_data = load_pickle(path_to_pickle)

        else:
            try:
                hazard_tree = self.get_hazard_tree()  # (23950, 60) 1770 with FurlongID being -1
                furlong_location = self.get_furlong_location()  # (77017, 8)
                tree_age_class = self.get_tree_age_class()  # (7, 1)
                tree_size_class = self.get_tree_size_class()  # (5, 1)

                hazardous_trees_data = hazard_tree. \
                    join(furlong_location,  # (22180, 68)
                         on='FurlongID', how='inner', lsuffix='', rsuffix='_FurlongLocation'). \
                    join(tree_age_class,  # (22180, 69)
                         on='TreeAgeCatID', how='inner'). \
                    join(tree_size_class,  # (22180, 70)
                         on='TreeSizeCatID', how='inner', lsuffix='_TreeAgeClass',
                         rsuffix='_TreeSizeClass'). \
                    drop(labels=['Route_FurlongLocation', 'DU_FurlongLocation', 'ELR_FurlongLocation'],
                         axis=1)

                if route_name is not None:
                    route_name = find_similar_str(route_name, self.get_du_route().Route)
                    hazardous_trees_data = hazardous_trees_data.loc[
                        hazardous_trees_data.Route == route_name]

                # Edit the merged data
                hazardous_trees_data.drop(
                    [f for f in hazardous_trees_data.columns if re.match('.*_FurlongLocation$', f)][:3],
                    axis=1, inplace=True)  # (22180, 66)
                hazardous_trees_data.index = range(len(hazardous_trees_data))  # Rearrange index

                hazardous_trees_data.rename(columns={'StartMileage': 'Furlong_StartMileage',
                                                     'EndMileage': 'Furlong_EndMileage',
                                                     'Electrified': 'Furlong_Electrified',
                                                     'HazardOnly': 'Furlong_HazardOnly'}, inplace=True)

                if pickle_it:
                    save_pickle(hazardous_trees_data, path_to_pickle, verbose=verbose)

            except Exception as e:
                print("Failed to fetch the information of hazardous trees. {}".format(e))
                hazardous_trees_data = None

        return hazardous_trees_data

    def view_vegetation_condition_per_furlong(self, route_name=None, update=False, pickle_it=True,
                                              verbose=False):
        """
        get a view of vegetation data combined with information of hazardous trees (75247, 58).

        :param route_name: name of a Route; if ``None`` (default), all Routes
        :type route_name: str or None
        :param update: whether to check on update and proceed to update the package data,
            defaults to ``True``
        :type update: bool
        :param pickle_it: whether to save the view as a pickle file, defaults to ``True``
        :type pickle_it: bool
        :param verbose: whether to print relevant information in console as the function runs,
            defaults to ``False``
        :type verbose: bool or int
        :return: vegetation data combined with information of hazardous trees
        :rtype: pandas.DataFrame or None

        **Test**::

            >>> from preprocessor import Vegetation
            
            >>> veg = Vegetation()

            >>> fv = veg.view_vegetation_condition_per_furlong(update=True, pickle_it=True, verbose=True)
            Updating "vegetation-condition-per-furlong.pickle" at "data\\...\\views" ... Done.

            >>> fv.tail()
                   ELR         DU  ... HazardTreeprox3py_min  HazardTreeprox3py_max
            75242  YMS  IMDM York  ...                   NaN                    NaN
            75243  YMS  IMDM York  ...                   NaN                    NaN
            75244  YMS  IMDM York  ...                   NaN                    NaN
            75245  YMS  IMDM York  ...                   NaN                    NaN
            75246  YMS  IMDM York  ...                   NaN                    NaN
            [5 rows x 58 columns]
        """

        path_to_pickle = self.cdd_views(make_filename("vegetation-condition-per-furlong", route_name))

        if os.path.isfile(path_to_pickle) and not update:
            furlong_vegetation_data = load_pickle(path_to_pickle)

        else:
            try:
                hazardous_trees_data = self.view_hazardous_trees()  # (22180, 66)

                group_cols = ['ELR', 'DU', 'Route', 'Furlong_StartMileage', 'Furlong_EndMileage']
                furlong_hazard_tree = hazardous_trees_data.groupby(group_cols).aggregate({
                    # 'AssetNumber': np.count_nonzero,
                    'Haztreeid': np.count_nonzero,
                    'TreeheightM': [lambda x: tuple(x), min, max],
                    'TreediameterM': [lambda x: tuple(x), min, max],
                    'TreeproxrailM': [lambda x: tuple(x), min, max],
                    'Treeprox3py': [lambda x: tuple(x), min, max]})
                # (11320, 13)

                furlong_hazard_tree.columns = ['_'.join(x).strip() for x in furlong_hazard_tree.columns]
                furlong_hazard_tree.rename(columns={'Haztreeid_count_nonzero': 'TreeNumber'},
                                           inplace=True)
                furlong_hazard_tree.columns = ['Hazard' + x.strip('_<lambda_0>') for x in
                                               furlong_hazard_tree.columns]

                #
                furlong_vegetation_coverage = self.view_vegetation_coverage_per_furlong()  # (75247, 45)

                # Processing ...
                furlong_vegetation_data = furlong_vegetation_coverage.join(
                    furlong_hazard_tree, on=['ELR', 'DU', 'Route', 'StartMileage', 'EndMileage'],
                    how='left')
                furlong_vegetation_data.sort_values('StructuredPlantNumber', inplace=True)  # (75247, 58)

                if route_name is not None:
                    route_name = find_similar_str(route_name, self.get_du_route().Route)
                    furlong_vegetation_data = hazardous_trees_data.loc[
                        furlong_vegetation_data.Route == route_name]
                    furlong_vegetation_data.index = range(len(furlong_vegetation_data))

                if pickle_it:
                    save_pickle(furlong_vegetation_data, path_to_pickle, verbose=verbose)

            except Exception as e:
                print(
                    "Failed to fetch the information of vegetation condition per furlong. {}".format(e))
                furlong_vegetation_data = None

        return furlong_vegetation_data

    def view_nr_vegetation_furlong_data(self, update=False, pickle_it=True, verbose=False):
        """
        Get a view of ELR and mileage data of furlong locations.

        :param update: whether to check on update and proceed to update the package data,
            defaults to ``True``
        :type update: bool
        :param pickle_it: whether to save the view as a pickle file, defaults to ``True``
        :type pickle_it: bool
        :param verbose: whether to print relevant information in console as the function runs,
            defaults to ``False``
        :type verbose: bool or int
        :return: vegetation data combined with information of hazardous trees
        :rtype: pandas.DataFrame or None

        **Test**::

            >>> from preprocessor import Vegetation
            
            >>> veg = Vegetation()

            >>> nr_vf = veg.view_nr_vegetation_furlong_data(update=True, pickle_it=True, verbose=True)
            Updating "vegetation-furlong-data.pickle" at "data\\vegetation\\database\\views" ... Done.

            >>> nr_vf.tail()
                       ELR         DU  ... StartMileage_num  EndMileage_num
            FurlongID                  ...
            9759       YMS  IMDM York  ...           41.088         41.1100
            9760       YMS  IMDM York  ...           41.110         41.1320
            21842      YMS  IMDM York  ...           41.132         41.1540
            30447      YMS  IMDM York  ...           41.154         42.0000
            14098      YMS  IMDM York  ...           42.000         42.0145
            [5 rows x 59 columns]
        """

        path_to_pickle = self.cdd_views("vegetation-furlong-data.pickle")

        if os.path.isfile(path_to_pickle) and not update:
            nr_vegetation_furlong_data = load_pickle(path_to_pickle)

        else:
            try:
                # Get the data of furlong location
                nr_vegetation_furlong_data = self.view_vegetation_condition_per_furlong()
                nr_vegetation_furlong_data.set_index('FurlongID', inplace=True)
                nr_vegetation_furlong_data.sort_index(inplace=True)

                # Column names of mileage data (as string)
                str_mileage_colnames = ['StartMileage', 'EndMileage']
                # Column names of ELR and mileage data (as string)
                elr_mileage_colnames = ['ELR'] + str_mileage_colnames

                nr_vegetation_furlong_data.drop_duplicates(elr_mileage_colnames, inplace=True)
                empty_start_mileage_idx = nr_vegetation_furlong_data[
                    nr_vegetation_furlong_data.StartMileage == ''].index
                nr_vegetation_furlong_data.loc[empty_start_mileage_idx, 'StartMileage'] = [
                    nr_vegetation_furlong_data.StructuredPlantNumber.loc[i][11:17] for i in
                    empty_start_mileage_idx]

                # Create two new columns of mileage data (as float)
                num_mileage_colnames = ['StartMileage_num', 'EndMileage_num']
                nr_vegetation_furlong_data[num_mileage_colnames] = nr_vegetation_furlong_data[
                    str_mileage_colnames].applymap(nr_mileage_str_to_num)

                # Sort the furlong data by ELR and mileage
                nr_vegetation_furlong_data.sort_values(['ELR'] + num_mileage_colnames, inplace=True)

                if pickle_it:
                    save_pickle(nr_vegetation_furlong_data, path_to_pickle, verbose=verbose)

            except Exception as e:
                print("Failed to fetch ELR & mileage data of furlong locations. {}".format(e))
                nr_vegetation_furlong_data = None

        return nr_vegetation_furlong_data

    def update_vegetation_view_pickles(self, route_name=None, update=True, pickle_it=True, verbose=True):
        """
        Update the local pickle files for all essential views.

        :param route_name: name of a Route; if ``None`` (default), all Routes
        :type route_name: str or None
        :param update: whether to check on update and proceed to update the package data,
            defaults to ``True``
        :type update: bool
        :param pickle_it: whether to save the view as a pickle file, defaults to ``True``
        :type pickle_it: bool
        :param verbose: whether to print relevant information in console as the function runs,
            defaults to ``False``
        :type verbose: bool or int

        **Test**::

            >>> from preprocessor import Vegetation
            
            >>> veg = Vegetation()

            >>> veg.update_vegetation_view_pickles(update=True, pickle_it=True, verbose=True)
        """

        if confirmed("To update the View pickles of the NR_Vegetation data?"):

            _ = self.view_hazardous_trees(route_name=route_name, update=update, pickle_it=pickle_it,
                                          verbose=verbose)
            _ = self.view_hazardous_trees(route_name='Anglia', update=update, pickle_it=pickle_it,
                                          verbose=verbose)

            _ = self.view_vegetation_condition_per_furlong(route_name=route_name, update=update,
                                                           pickle_it=pickle_it, verbose=verbose)

            _ = self.view_vegetation_coverage_per_furlong(route_name=route_name, update=update,
                                                          pickle_it=pickle_it, verbose=verbose)

            _ = self.view_nr_vegetation_furlong_data(update=update, pickle_it=pickle_it, verbose=verbose)

            if verbose:
                print("\nUpdate finished.")
