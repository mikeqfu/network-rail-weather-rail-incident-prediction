""" Utilities - Helper functions """


import functools
import itertools
import operator
import os
import urllib.parse

import numpy as np
import pandas as pd
import pyodbc
import shapely.wkt
import sqlalchemy
from pyhelpers.dir import cd, cdd
from pyhelpers.store import load_json, save
from pyhelpers.text import find_similar_str


# == Change directories ===============================================================================

def cdd_exploration(*sub_dir, mkdir=False):
    """
    Change directory to "data\\exploration" and sub-directories / a file.

    :param sub_dir: name of directory or names of directories (and/or a filename)
    :type sub_dir: str
    :param mkdir: whether to create a directory, defaults to ``False``
    :type mkdir: bool
    :return: full path to "data\\exploration" and sub-directories / a file
    :rtype: str
    """
    
    path = cdd("exploration", *sub_dir, mkdir=mkdir)
    
    return path


def cdd_metex(*sub_dir, mkdir=False):
    """
    Change directory to "data\\metex" and sub-directories / a file.

    :param sub_dir: name of directory or names of directories (and/or a filename)
    :type sub_dir: str
    :param mkdir: whether to create a directory, defaults to ``False``
    :type mkdir: bool
    :return: full path to "data\\metex" and sub-directories / a file
    :rtype: str
    """

    path = cdd("metex", *sub_dir, mkdir=mkdir)
    
    return path


def cdd_network(*sub_dir, mkdir=False):
    """
    Change directory to "data\\network" and sub-directories / a file.

    :param sub_dir: name of directory or names of directories (and/or a filename)
    :type sub_dir: str
    :param mkdir: whether to create a directory, defaults to ``False``
    :type mkdir: bool
    :return: full path to "data\\network" and sub-directories / a file
    :rtype: str
    """

    path = cdd("network", *sub_dir, mkdir=mkdir)
    
    return path


def cdd_railway_codes(*sub_dir, mkdir=False):
    """
    Change directory to "data\\network\\railway codes" and sub-directories / a file.

    :param sub_dir: name of directory or names of directories (and/or a filename)
    :type sub_dir: str
    :param mkdir: whether to create a directory, defaults to ``False``
    :type mkdir: bool
    :return: full path to "data\\network\\railway codes" and sub-directories / a file
    :rtype: str
    """

    path = cdd_network("railway codes", *sub_dir, mkdir=mkdir)
    
    return path


def cdd_vegetation(*sub_dir, mkdir=False):
    """
    Change directory to "data\\vegetation" and sub-directories / a file.

    :param sub_dir: name of directory or names of directories (and/or a filename)
    :type sub_dir: str
    :param mkdir: whether to create a directory, defaults to ``False``
    :type mkdir: bool
    :return: full path to "data\\vegetation" and sub-directories / a file
    :rtype: str
    """

    path = cdd("vegetation", *sub_dir, mkdir=mkdir)
    
    return path


def cdd_weather(*sub_dir, mkdir=False):
    """
    Change directory to "data\\weather" and sub-directories / a file.

    :param sub_dir: name of directory or names of directories (and/or a filename)
    :type sub_dir: str
    :param mkdir: whether to create a directory, defaults to ``False``
    :type mkdir: bool
    :return: full path to "data\\weather" and sub-directories / a file
    :rtype: str
    """

    path = cdd("weather", *sub_dir, mkdir=mkdir)
    
    return path


def cd_models(*sub_dir, mkdir=False):
    """
    Change directory to "models" and sub-directories / a file.

    :param sub_dir: name of directory or names of directories (and/or a filename)
    :type sub_dir: str
    :param mkdir: whether to create a directory, defaults to ``False``
    :type mkdir: bool
    :return: full path to "models" and sub-directories / a file
    :rtype: str
    """

    path = cd("models", *sub_dir, mkdir=mkdir)

    return path


# == Utilities for communicating with a MS SQL server =================================================

def use_windows_authentication():
    """
    Windows authentication for reading data from the databases.

    The trusted_connection setting indicates whether to use Windows Authentication Mode for login
    validation or not. 'Trusted_Connection=yes' specifies the user used to establish this
    connection. In Microsoft implementations, this user account is a Windows user account.

    :return: whether to use Windows authentication
    :rtype: str
    """

    win_str = 'Trusted_Connection=yes;'

    return win_str


def specify_database_driver():
    """
    Specify an ODBC driver.

    .. note::

        Microsoft have written and distributed multiple ODBC drivers for SQL Server:

            - {SQL Server} - released with SQL Server 2000
            - {SQL Native Client} - released with SQL Server 2005 (also known as version 9.0)
            - {SQL Server Native Client 10.0} - released with SQL Server 2008
            - {SQL Server Native Client 11.0} - released with SQL Server 2012
            - {ODBC Driver 11 for SQL Server} - supports SQL Server 2005 through 2014
            - {ODBC Driver 13 for SQL Server} - supports SQL Server 2005 through 2016
            - {ODBC Driver 13.1 for SQL Server} - supports SQL Server 2008 through 2016
            - {ODBC Driver 17 for SQL Server} - supports SQL Server 2008 through 2017

        The "SQL Server Native Client ..." and earlier drivers are deprecated and should not
        be used for new development.

    .. seealso::

        https://github.com/mkleehammer/pyodbc/wiki/Connecting-to-SQL-Server-from-Windows
    """

    dri_str = 'DRIVER={ODBC Driver 17 for SQL Server};'

    return dri_str


def specify_server_name():
    """
    Specify database server name.

    .. note::

        Alternative methods to get the computer name:

        .. code-block: python

            # method 1:
            import platform
            platform.node()

            # method 2:
            import socket
            socket.gethostname()
    """

    server_name = os.environ['COMPUTERNAME']
    # server_name += '\\SQLEXPRESS'
    ser_str = 'SERVER={};'.format(server_name)

    return ser_str


def specify_database_name(database_name):
    """
    Specify database name.

    :param database_name: name of a database
    :type database_name: str
    """

    dbn_str = 'DATABASE={};'.format(database_name)

    return dbn_str


def create_mssql_connectable_engine(database_name):
    """
    Create a SQLAlchemy connectable engine to MS SQL Server.

    Connect string format: 'mssql+pyodbc://<username>:<password>@<dsn_name>'

    :param database_name: name of a database
    :type database_name: str
    :return: a SQLAlchemy connectable engine to MS SQL Server
    :rtype: sqlalchemy.engine.Engine

    .. note::

        To get connected to the database (e.g. NR_METEX_* and NR_Vegetation_*):

        - Use pyodbc (or pypyodbc)
            connect_string = \
                'driver={drivername};server=servername;database=databaseName;uid=username;pwd=password'
            conn = pyodbc.connect(connect_string)  # equivalent to: pypyodbc.connect(connect_string)
        - Use SQLAlchemy connectable,
            conn_string = 'mssql+pyodbc:///?odbc_connect=%s' % quote_plus(connect_string)
            engine = sqlalchemy.create_engine(conn_string)
            conn = engine.connect()
    """

    conn_str = \
        specify_database_driver() + \
        specify_server_name() + \
        specify_database_name(database_name) + \
        use_windows_authentication()
    db_engine = sqlalchemy.create_engine(
        'mssql+pyodbc:///?odbc_connect=%s' % urllib.parse.quote_plus(conn_str))
    return db_engine


def establish_mssql_connection(database_name, mode=None):
    """
    Establish a SQLAlchemy connection to MS SQL Server.

    :param database_name: name of a database
    :type database_name: str
    :param mode: defaults to ``None``
    :type mode: str or None
    :return: a SQLAlchemy connection to MS SQL Server
    :rtype: sqlalchemy.engine.Connection
    """

    if not mode:  # (default)
        db_engine = create_mssql_connectable_engine(database_name)
        db_conn = db_engine.connect()

    else:  # i.e. to use directly 'pyodbc'
        conn_str = use_windows_authentication() + specify_database_driver() + specify_server_name()
        db_conn = pyodbc.connect(conn_str, database=database_name)

    return db_conn


def create_mssql_db_cursor(database_name):
    """
    Create a pyodbc cursor.

    :param database_name: name of a database
    :type database_name: str
    :return: a pyodbc cursor
    :rtype: pyodbc.Cursor
    """

    db_conn = establish_mssql_connection(database_name, mode='pyodbc')
    db_cursor = db_conn.cursor()

    return db_cursor


def read_table_by_name(database_name, table_name, schema_name='dbo', col_names=None, chunk_size=None,
                       index_col=None, save_as=None, data_dir=None, **kwargs):
    """
    Get all data from a given table in a specific database.

    :param database_name: name of a database
    :type database_name: str
    :param table_name: name of a queried table from the given database
    :type table_name: str
    :param schema_name: defaults to ``'dbo'``
    :type schema_name: str
    :param col_names: defaults to ``None``
    :type col_names: list, None
    :param index_col: defaults to ``None``
    :type index_col: str or list or None
    :param chunk_size: defaults to ``None``
    :type chunk_size: int, None
    :param save_as: defaults to ``None``
    :type save_as: str or None
    :param data_dir: defaults to ``None``
    :type data_dir: str or None
    :param kwargs: optional parameters of
        `pandas.read_sql
        <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_sql.html>`_
    :return: data of the queried table
    :rtype: pandas.DataFrame
    """

    # Connect to the queried database
    db_conn = establish_mssql_connection(database_name)
    # Create a pandas.DataFrame of the queried table_name
    table_data = pd.read_sql_table(table_name=table_name, con=db_conn, schema=schema_name,
                                   columns=col_names, index_col=index_col, chunksize=chunk_size,
                                   **kwargs)
    # Disconnect the database
    db_conn.close()

    if save_as:
        path_to_file = os.path.join(
            os.path.realpath(data_dir if data_dir else ''), table_name + save_as)
        save(table_data, path_to_file)

    # Return the data frame of the queried table
    return table_data


def read_table_by_query(database_name, table_name, schema_name='dbo', col_names=None, index_col=None,
                        chunk_size=None, save_as=None, data_dir=None, **kwargs):
    """
    Get data from a table in a specific database by SQL query.

    :param database_name: name of a database
    :type database_name: str
    :param table_name: name of a queried table from the given database
    :type table_name: str
    :param schema_name: defaults to ``'dbo'``
    :type schema_name: str
    :param col_names: defaults to ``None``
    :type col_names: iterable, None
    :param index_col: defaults to ``None``
    :type index_col: str or list or None
    :param chunk_size: defaults to ``None``
    :type chunk_size: int, None
    :param save_as: defaults to ``None``
    :type save_as: str or None
    :param data_dir: defaults to ``None``
    :type data_dir: str or None
    :param kwargs: optional parameters of
        `pandas.read_sql
        <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_sql.html>`_
    :return: data of the queried table
    :rtype: pandas.DataFrame
    """

    if col_names is not None:
        assert isinstance(col_names, (list, tuple)) and all(isinstance(x, str) for x in col_names)
    if save_as:
        assert isinstance(save_as, str) and save_as in (".pickle", ".csv", ".xlsx", ".txt")
    if data_dir:
        assert isinstance(save_as, str)

    # Connect to the queried database
    assert isinstance(database_name, str)
    db_conn = establish_mssql_connection(database_name)

    # Check if there is column of 'geometry' type
    assert isinstance(table_name, str)
    sql_query_geom_col = "SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS " \
                         "WHERE TABLE_NAME='{}' AND DATA_TYPE='geometry'".format(table_name)
    geom_col_res = db_conn.execute(sql_query_geom_col).fetchall()
    geom_col_names = list(itertools.chain.from_iterable(geom_col_res)) if geom_col_res else []

    # Get a list of column names, excluding the 'geometry' one if it exists
    table_col_names = [x for x in get_table_column_names(database_name, table_name)
                       if x not in geom_col_names]

    # Specify SQL query - read all non-geom columns
    selected_col_names = [x for x in col_names if x not in geom_col_names] if col_names \
        else table_col_names
    sql_query = 'SELECT {} FROM {}."{}"'.format(
        ', '.join('"' + tbl_col_name + '"' for tbl_col_name in selected_col_names),
        schema_name, table_name)

    # Read the queried table_name into a pandas.DataFrame
    table_data = pd.read_sql(
        sql=sql_query, con=db_conn, columns=col_names, index_col=index_col, chunksize=chunk_size,
        **kwargs)

    if chunk_size:
        table_data = pd.concat([pd.DataFrame(tbl_dat) for tbl_dat in table_data], ignore_index=True)

    # Read geom column(s)
    if geom_col_names:
        if len(geom_col_names) == 1:
            geom_sql_query = 'SELECT {} FROM {}."{}"'.format(
                '"{}".STAsText()'.format(geom_col_names[0]), schema_name, table_name)
        else:
            geom_sql_query = 'SELECT {} FROM {}."{}"'.format(
                ', '.join('"' + x + '".STAsText()' for x in geom_col_names),
                schema_name, table_name)

        # Read geom data chunks into a pandas.DataFrame
        geom_data = pd.read_sql(geom_sql_query, db_conn, chunksize=chunk_size, **kwargs)

        if chunk_size:
            geom_data = pd.concat(
                [pd.DataFrame(geom_dat).applymap(shapely.wkt.loads) for geom_dat in geom_data],
                ignore_index=True)
        geom_data.columns = geom_col_names
        #
        table_data = table_data.join(geom_data)

    # Disconnect the database
    db_conn.close()

    if save_as:
        path_to_file = os.path.join(
            os.path.realpath(data_dir if data_dir else ''), table_name + save_as)
        save(table_data, path_to_file)

    return table_data


def save_table_by_chunk(database_name, table_name, schema_name='dbo', col_names=None, index_col=None,
                        chunk_size=1000000, save_as=".pickle", data_dir=None, **kwargs):
    """
    Save a table chunk-wise from a database.

    :param database_name: name of a database to query
    :type database_name: str
    :param table_name: name of a queried table from the given database
    :type table_name: str
    :param schema_name: defaults to ``'dbo'``
    :type schema_name: str
    :param col_names: e.g. a list of column names to retrieve; ``None`` (default) for all columns
    :type col_names: list, None
    :param index_col: defaults to ``None``
    :type index_col: str or list or None
    :param chunk_size: defaults to ``None``
    :type chunk_size: int, None
    :param save_as: defaults to ``None``
    :type save_as: str or None
    :param data_dir: defaults to ``None``
    :type data_dir: str or None
    :param kwargs: optional parameters of
        `pandas.read_sql
        <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_sql.html>`_
    :return: data of the queried table
    :rtype: pandas.DataFrame
    """

    assert isinstance(save_as, str) and save_as in (".pickle", ".csv", ".xlsx", ".txt")
    if col_names is not None:
        assert isinstance(col_names, (list, tuple)) and all(isinstance(x, str) for x in col_names)
    if data_dir:
        assert isinstance(save_as, str)

    # Connect to the queried database
    assert isinstance(database_name, str)
    db_conn = establish_mssql_connection(database_name)

    # Check if there is column of 'geometry' type
    assert isinstance(table_name, str)
    sql_query_geom_col = "SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS " \
                         "WHERE TABLE_NAME='{}' AND DATA_TYPE='geometry'".format(table_name)
    geom_col_res = db_conn.execute(sql_query_geom_col).fetchall()
    geom_col_names = list(itertools.chain.from_iterable(geom_col_res)) if geom_col_res else []

    # Get a list of column names, excluding the 'geometry' one if it exists
    table_col_names = [x for x in get_table_column_names(database_name, table_name)
                       if x not in geom_col_names]

    # Specify SQL query - read all of the selected non-geom columns
    selected_col_names = [x for x in col_names if x not in geom_col_names] if col_names \
        else table_col_names

    sql_query = 'SELECT {} FROM {}."{}"'.format(
        ', '.join('"' + tbl_col_name + '"' for tbl_col_name in selected_col_names),
        schema_name, table_name)

    dat_dir = os.path.realpath(data_dir if data_dir else 'temp_dat')
    if not geom_col_names:
        # Read the queried table_name into a pandas.DataFrame
        table_data = pd.read_sql(sql=sql_query, con=db_conn, columns=col_names, index_col=index_col,
                                 chunksize=chunk_size, **kwargs)
        for tbl_id, tbl_dat in enumerate(table_data):
            path_to_file = os.path.join(dat_dir, table_name + "_{}".format(tbl_id + 1) + save_as)
            save(tbl_dat, path_to_file, sheet_name="Sheet_{}".format(tbl_id + 1))
    else:
        # Read the queried table_name into a pd.DataFrame
        table_data = pd.read_sql(sql=sql_query, con=db_conn, columns=col_names, index_col=index_col,
                                 chunksize=chunk_size, **kwargs)
        tbl_chunks = [tbl_dat for tbl_dat in table_data]

        if len(geom_col_names) == 1:
            geom_sql_query = 'SELECT {} FROM {}."{}"'.format(
                '"{}".STAsText()'.format(geom_col_names[0]), schema_name, table_name)
        else:
            geom_sql_query = 'SELECT {} FROM {}."{}"'.format(
                ', '.join('"' + x + '".STAsText()' for x in geom_col_names),
                schema_name, table_name)
        geom_data = pd.read_sql(geom_sql_query, db_conn, chunksize=chunk_size, **kwargs)
        geom_chunks = [geom_dat.applymap(shapely.wkt.loads) for geom_dat in geom_data]

        counter = 0
        # noinspection PyTypeChecker
        for tbl_dat, geom_dat in zip(tbl_chunks, geom_chunks):
            path_to_file = os.path.join(dat_dir, table_name + "_{}".format(counter + 1) + save_as)
            save(tbl_dat.join(geom_dat), path_to_file, sheet_name="Sheet_{}".format(counter + 1))
            counter += 1

    # Disconnect the database
    db_conn.close()


def get_table_names(database_name, schema_name='dbo', table_type='TABLE'):
    """
    Get a list of table names in a database.

    This function gets a list of names of tables in a database, given a specific table type.
    The table types could include 'TABLE', 'VIEW', 'SYSTEM TABLE', 'ALIAS', 'GLOBAL TEMPORARY',
    'SYNONYM', 'LOCAL TEMPORARY', or a data source-specific type name.

    :param database_name: name of the database queried (also the catalog name)
    :type database_name: str
    :param schema_name: name of schema, e.g. ``'dbo'`` (default), ``'sys'``
    :type schema_name: str
    :param table_type: table type, defaults to ``'TABLE'``
    :type table_type: str
    :return: a list of names of the tables in the queried database
    :rtype: list
    """

    # Create a cursor with a direct connection to the queried database
    db_cursor = create_mssql_db_cursor(database_name)
    # Get a results set of tables defined in the data source
    table_list = [t.table_name for t in db_cursor.tables(schema=schema_name, tableType=table_type)]
    # Close the connection
    db_cursor.close()
    # Return a list of the names of table names
    return table_list


def get_table_column_names(database_name, table_name, schema_name='dbo'):
    """
    Get a list of column names of a given table in a database.

    :param database_name: name of a database
    :type database_name: str
    :param table_name: name of a queried table from the given database
    :type table_name: str
    :param schema_name: defaults to ``'dbo'``
    :type schema_name: str
    :return: a list of column names
    :rtype: list
    """

    db_cursor = create_mssql_db_cursor(database_name)
    col_names = [x.column_name for x in db_cursor.columns(table_name, schema=schema_name)]
    db_cursor.close()
    return col_names


def get_table_primary_keys(database_name, table_name=None, schema_name='dbo', table_type='TABLE'):
    """
    Get the primary keys of each table in a database.

    :param database_name: name of a database
    :type database_name: str
    :param table_name: name of a queried table from the given database, defaults to ``None``
    :type table_name: str or None
    :param schema_name: defaults to ``'dbo'``
    :type schema_name: str
    :param table_type: table type, defaults to ``'TABLE'``
    :type table_type: str
    :return: a list of primary keys
    :rtype: list
    """

    try:
        db_cursor = create_mssql_db_cursor(database_name)
        # Get all table names
        table_names = [table.table_name
                       for table in db_cursor.tables(schema=schema_name, tableType=table_type)]
        # Get primary keys for each table
        tbl_pks = [{k.table_name: k.column_name} for tbl_name in table_names
                   for k in db_cursor.primaryKeys(tbl_name)]
        # Close the cursor
        db_cursor.close()
        # ( Each element of 'tbl_pks' (as a dict) is in the format of {'table_name':'primary key'} )
        tbl_names_set = functools.reduce(operator.or_, (set(d.keys()) for d in tbl_pks), set())
        # Find all primary keys for each table
        tbl_pk_dict = dict((tbl, [d[tbl] for d in tbl_pks if tbl in d]) for tbl in tbl_names_set)
        result_pks = tbl_pk_dict[table_name] if table_name else tbl_pk_dict
    except KeyError:  # Most likely that the table (i.e. 'table_name') does not have any primary key
        result_pks = None
    return result_pks


# == Misc =============================================================================================

def make_filename(name, route_name=None, weather_category=None, *suffixes, sep="-", save_as=".pickle"):
    """
    Make a filename as appropriate.

    :param name: base name, defaults to ``None``
    :type name: str or None
    :param route_name: name of a Route, defaults to ``None``
    :type route_name: str or list or None
    :param weather_category: weather category, defaults to ``None``
    :type weather_category: str or list or None
    :param suffixes: extra suffixes to the filename
    :type suffixes: int or str or None
    :param sep: a separator in the filename, defaults to ``"-"``
    :type sep: str or None
    :param save_as: file extension, defaults to ``".pickle"``
    :type save_as: str
    :return: a filename
    :rtype: str

    **Test**::

        >>> from utils import make_filename

        name = "test"  # None
        sep = "-"
        save_as = ".pickle"

        route_name = None
        weather_category = None
        make_filename(base_name, route_name, weather_category)
        # test.pickle

        route_name = None
        weather_category = 'Heat'
        make_filename(None, route_name, weather_category, "test1")
        # test1.pickle

        make_filename(name, route_name, weather_category, "test1", "test2")
        # test-test1-test2.pickle

        make_filename(name, 'Anglia', weather_category, "test2")
        # test-Anglia-test2.pickle

        make_filename(name, 'North and East', 'Heat', "test1", "test2")
        # test-North_and_East-Heat-test1-test2.pickle
    """

    base_name = "" if name is None else name

    if route_name is None:
        route_name_ = ""
    else:
        rts = list(set(load_json(cdd_network("routes", "name-changes.json")).values()))
        route_name_ = "_".join(
            [find_similar_str(x, rts).replace(" ", "") for x in
             ([route_name] if isinstance(route_name, str) else list(route_name))])
        if base_name != "":
            route_name_ = sep + route_name_

    if weather_category is None:
        weather_category_ = ""
    else:
        wcs = load_json(cdd_weather("weather-categories.json"))['WeatherCategory']
        weather_category_ = "_".join(
            [find_similar_str(x, wcs).replace(" ", "") for x in
             ([weather_category] if isinstance(weather_category, str) else list(weather_category))])
        if base_name != "":
            weather_category_ = sep + weather_category_

    if base_name + route_name_ + weather_category_ == "":
        base_name = "data"

    if suffixes:
        extra_suffixes = [suffixes] if isinstance(suffixes, str) else suffixes
        suffix_ = ["{}".format(s) for s in extra_suffixes if s]
        try:
            suffix = sep + sep.join(suffix_) if len(suffix_) > 1 else sep + suffix_[0]
        except IndexError:
            suffix = ""
        filename = base_name + route_name_ + weather_category_ + suffix + save_as

    else:
        filename = base_name + route_name_ + weather_category_ + save_as

    return filename


def get_subset(data_frame, route_name=None, weather_category=None, rearrange_index=False):
    """
    Subset of a data set for the given Route and weather category.

    :param data_frame: a data frame (that contains 'Route' and 'WeatherCategory')
    :type data_frame: pandas.DataFrame, None
    :param route_name: name of a Route, defaults to ``None``
    :type route_name: str or list or None
    :param weather_category: weather category, defaults to ``None``
    :type weather_category: str or list or None
    :param rearrange_index: whether to rearrange the index of the subset, defaults to ``False``
    :type rearrange_index: bool
    :return: a subset of the ``data_frame`` for the given ``route_name`` and ``weather_category``
    :rtype: pandas.DataFrame, None
    """

    if data_frame is not None:
        assert isinstance(data_frame, pd.DataFrame) and not data_frame.empty
        data_subset = data_frame.copy(deep=True)

        if route_name:
            try:  # assert 'Route' in data_subset.columns
                data_subset.Route = data_subset.Route.astype(str)
                route_lookup = list(set(data_subset.Route))
                route_name_ = [
                    find_similar_str(x, route_lookup)
                    for x in ([route_name] if isinstance(route_name, str) else list(route_name))]
                data_subset = data_subset[data_subset.Route.isin(route_name_)]
            except AttributeError:
                print("Couldn't slice the data by 'Route'. "
                      "The attribute may not exist in the DataFrame.")
                pass

        if weather_category:
            try:  # assert 'WeatherCategory' in data_subset.columns
                data_subset.WeatherCategory = data_subset.WeatherCategory.astype(str)
                weather_category_lookup = list(set(data_subset.WeatherCategory))
                weather_category_ = [
                    find_similar_str(x, weather_category_lookup)
                    for x in ([weather_category] if isinstance(weather_category, str)
                              else list(weather_category))
                ]
                data_subset = data_subset[data_subset.WeatherCategory.isin(weather_category_)]
            except AttributeError:
                print("Couldn't slice the data by 'WeatherCategory'. "
                      "The attribute may not exist in the DataFrame.")
                pass

        if rearrange_index:
            data_subset.index = range(len(data_subset))  # data_subset.reset_index(inplace=True)

    else:
        data_subset = None

    return data_subset


def remove_list_duplicates(lst):
    """
    Remove duplicates in a list.

    :param lst: a list
    :type lst: list
    :return: a list without duplicated items
    :rtype: list
    """

    output = []
    temp = set()
    for item in lst:
        if item not in temp:
            output.append(item)
            temp.add(item)
    del temp
    return output


def remove_list_duplicated_lists(lst_lst):
    """
    Make each item in a list be unique, where the item is also a list.

    :param lst_lst: a list of lists
    :type lst_lst: list
    :return: a list of lists with each item-list being unique
    :rtype: list
    """

    output = []
    temp = set()
    for lst in lst_lst:
        if any(item not in temp for item in lst):
            # lst[0] not in temp and lst[1] not in temp:
            output.append(lst)
            for item in lst:
                temp.add(item)  # e.g. temp.add(lst[0]); temp.add(lst[1])
    del temp
    return output


def get_index_of_dict_in_list(lst_of_dict, key, val):
    """
    Get the index of a dictionary in a list by a given (key, value).

    :param lst_of_dict: a list of dictionaries
    :param key: key of the queried dictionary
    :param val: value of the queried dictionary
    :return:

    **Test**::

        lst_of_dict = [{'a': 1}, {'b': 2}, {'c': 3}]
        key = 'b'
        value = 2

        get_index_of_dict_in_list(lst_of_dict, key, value)
    """

    return next(idx for (idx, d) in enumerate(lst_of_dict) if d.get(key) == val)


def merge_two_dicts(dict1, dict2):
    """
    Given two dicts, merge them into a new dict as a shallow copy.

    :param dict1: a dictionary
    :type dict1: dict
    :param dict2: another dictionary
    :type dict2: dict
    :return: a merged dictionary of ``dict1`` and ``dict2``
    :rtype: dict
    """

    new_dict = dict1.copy()
    new_dict.update(dict2)
    return new_dict


def merge_dicts(*dictionaries):
    """
    Given any number of dicts, shallow copy and merge into a new dict,
    precedence goes to key val pairs in latter dicts.

    :param dictionaries: one or a sequence of dictionaries
    :type dictionaries: dict
    :return: a merged dictionary
    :rtype: dict
    """

    new_dict = {}
    for d in dictionaries:
        new_dict.update(d)
    return new_dict


def reset_double_indexes(data_frame):
    """
    Reset double indexes.

    :param data_frame:
    :return:
    """

    levels = list(data_frame.columns)
    column_names = []
    for i in range(len(levels)):
        col_name = levels[i][0] + '_' + levels[i][1]
        column_names += [col_name]
    data_frame.columns = column_names
    data_frame.reset_index(inplace=True)
    return data_frame


def percentile(n):
    """
    Calculate the n-th percentile.
    """

    def np_percentile(x):
        return np.percentile(x, n)

    np_percentile.__name__ = 'percentile_%s' % n
    return np_percentile


def update_nr_route_names(data_set, route_col_name='Route'):
    """
    Update names of NR Routes.

    :param data_set: a given data frame that contains 'Route' column
    :type data_set: pandas.DataFrame
    :param route_col_name: name of the column for 'Route', defaults to ``'Route'``
    :return: updated data frame
    :rtype: pandas.DataFrame
    """

    assert isinstance(data_set, pd.DataFrame)
    assert route_col_name in data_set.columns
    route_names_changes = load_json(cdd_network("Routes", "name-changes.json"))
    new_route_col_name = route_col_name + 'Alias'
    data_set.rename(columns={route_col_name: new_route_col_name}, inplace=True)
    data_set[route_col_name] = data_set[new_route_col_name].replace(route_names_changes)


def get_coefficients(model, feature_names=None):
    """
    Get regression model coefficients presented as a data frame.

    :param model: an instance of regression model
    :type model: sklearn class
    :param feature_names: name of features (i.e. column names of input data);
        if ``None`` (default), all features
    :return: data frame for model coefficients
    :rtype: pandas.DataFrame
    """

    if feature_names is None:
        feature_names = ['feature_%d' % i for i in range(len(model.coef_))]
    feature_names = ['(intercept)'] + feature_names

    intercept = model.intercept_
    coefficients = model.coef_
    coefficients = [intercept] + list(coefficients)

    coef_dat = pd.DataFrame({'coefficients': coefficients}, index=feature_names)

    return coef_dat
