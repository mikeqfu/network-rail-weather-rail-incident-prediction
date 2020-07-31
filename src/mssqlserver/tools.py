""" Tools for communicating with the databases of NR_METEX_* and NR_Vegetation_*.

In order to connect to the database you use the connect method of the Connection object. To get connected:

- Using pyodbc (or pypyodbc)
    connect_string = 'driver={drivername};server=servername;database=databaseName;uid=username;pwd=password'
    conn = pyodbc.connect(connect_string)  # equivalent to: pypyodbc.connect(connect_string)
- Using SQLAlchemy connectable,
    conn_string = 'mssql+pyodbc:///?odbc_connect=%s' % quote_plus(connect_string)
    engine = sqlalchemy.create_engine(conn_string)
    conn = engine.connect()
"""

import functools
import itertools
import operator
import os
import urllib.parse

import pandas as pd
import pyodbc
import shapely.wkt
import sqlalchemy
from pyhelpers.store import save


# == Functions to establish a connection to a database server =========================================

def use_windows_authentication():
    """
    Windows authentication for reading data from the databases.

    The trusted_connection setting indicates whether to use Windows Authentication Mode for login validation or not.
    'Trusted_Connection=yes' specifies the user used to establish this connection. In Microsoft implementations,
    this user account is a Windows user account.

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

        The "SQL Server Native Client ..." and earlier drivers are deprecated and should not be used for
        new development.

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
    """

    conn_str = \
        specify_database_driver() + \
        specify_server_name() + \
        specify_database_name(database_name) + \
        use_windows_authentication()
    db_engine = sqlalchemy.create_engine('mssql+pyodbc:///?odbc_connect=%s' % urllib.parse.quote_plus(conn_str))
    return db_engine


def establish_mssql_connection(database_name, mode=None):
    """
    Establish a SQLAlchemy connection to MS SQL Server.

    :param database_name: name of a database
    :type database_name: str
    :param mode: defaults to ``None``
    :type mode: str, None
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


# == Functions to retrieve information ================================================================

def get_table_names(database_name, schema_name='dbo', table_type='TABLE'):
    """
    Get a list of table names in a database.

    This function gets a list of names of tables in a database, given a specific table type.
    The table types could include 'TABLE', 'VIEW', 'SYSTEM TABLE', 'ALIAS', 'GLOBAL TEMPORARY', 'SYNONYM',
    'LOCAL TEMPORARY', or a data source-specific type name.

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
    :type table_name: str, None
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
        table_names = [table.table_name for table in db_cursor.tables(schema=schema_name, tableType=table_type)]
        # Get primary keys for each table
        tbl_pks = [{k.table_name: k.column_name} for tbl_name in table_names for k in db_cursor.primaryKeys(tbl_name)]
        # Close the cursor
        db_cursor.close()
        # ( Each element of 'tbl_pks' (as a dict) is in the format of {'table_name': 'primary key'} )
        tbl_names_set = functools.reduce(operator.or_, (set(d.keys()) for d in tbl_pks), set())
        # Find all primary keys for each table
        tbl_pk_dict = dict((tbl, [d[tbl] for d in tbl_pks if tbl in d]) for tbl in tbl_names_set)
        result_pks = tbl_pk_dict[table_name] if table_name else tbl_pk_dict
    except KeyError:  # Most likely that the table (i.e. 'table_name') does not have any primary key
        result_pks = None
    return result_pks


# == Functions to read table data =====================================================================

def read_table_by_name(database_name, table_name, schema_name='dbo', col_names=None, chunk_size=None, index_col=None,
                       save_as=None, data_dir=None, **kwargs):
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
    :type index_col: str, list, None
    :param chunk_size: defaults to ``None``
    :type chunk_size: int, None
    :param save_as: defaults to ``None``
    :type save_as: str, None
    :param data_dir: defaults to ``None``
    :type data_dir: str, None
    :param kwargs: optional parameters of
        `pandas.read_sql <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_sql.html>`_
    :return: data of the queried table
    :rtype: pandas.DataFrame
    """

    # Connect to the queried database
    db_conn = establish_mssql_connection(database_name)
    # Create a pandas.DataFrame of the queried table_name
    table_data = pd.read_sql_table(table_name=table_name, con=db_conn, schema=schema_name, columns=col_names,
                                   index_col=index_col, chunksize=chunk_size, **kwargs)
    # Disconnect the database
    db_conn.close()

    if save_as:
        path_to_file = os.path.join(os.path.realpath(data_dir if data_dir else ''), table_name + save_as)
        save(table_data, path_to_file)

    # Return the data frame of the queried table
    return table_data


def read_table_by_query(database_name, table_name, schema_name='dbo', col_names=None, index_col=None, chunk_size=None,
                        save_as=None, data_dir=None, **kwargs):
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
    :type index_col: str, list, None
    :param chunk_size: defaults to ``None``
    :type chunk_size: int, None
    :param save_as: defaults to ``None``
    :type save_as: str, None
    :param data_dir: defaults to ``None``
    :type data_dir: str, None
    :param kwargs: optional parameters of
        `pandas.read_sql <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_sql.html>`_
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
    table_col_names = [x for x in get_table_column_names(database_name, table_name) if x not in geom_col_names]

    # Specify SQL query - read all non-geom columns
    selected_col_names = [x for x in col_names if x not in geom_col_names] if col_names else table_col_names
    sql_query = 'SELECT {} FROM {}."{}"'.format(
        ', '.join('"' + tbl_col_name + '"' for tbl_col_name in selected_col_names), schema_name, table_name)
    # Read the queried table_name into a pandas.DataFrame
    table_data = pd.read_sql(sql=sql_query, con=db_conn, columns=col_names, index_col=index_col, chunksize=chunk_size,
                             **kwargs)
    if chunk_size:
        table_data = pd.concat([pd.DataFrame(tbl_dat) for tbl_dat in table_data], ignore_index=True)

    # Read geom column(s)
    if geom_col_names:
        if len(geom_col_names) == 1:
            geom_sql_query = 'SELECT "{}".STAsText() FROM {}."{}"'.format(geom_col_names[0], schema_name, table_name)
        else:
            geom_sql_query = 'SELECT {} FROM {}."{}"'.format(
                ', '.join('"' + x + '".STAsText()' for x in geom_col_names), schema_name, table_name)
        # Read geom data chunks into a pandas.DataFrame
        geom_data = pd.read_sql(geom_sql_query, db_conn, chunksize=chunk_size, **kwargs)
        if chunk_size:
            geom_data = pd.concat([pd.DataFrame(geom_dat).applymap(shapely.wkt.loads) for geom_dat in geom_data],
                                  ignore_index=True)
        geom_data.columns = geom_col_names
        #
        table_data = table_data.join(geom_data)

    # Disconnect the database
    db_conn.close()

    if save_as:
        path_to_file = os.path.join(os.path.realpath(data_dir if data_dir else ''), table_name + save_as)
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
    :type index_col: str, list, None
    :param chunk_size: defaults to ``None``
    :type chunk_size: int, None
    :param save_as: defaults to ``None``
    :type save_as: str, None
    :param data_dir: defaults to ``None``
    :type data_dir: str, None
    :param kwargs: optional parameters of
        `pandas.read_sql <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_sql.html>`_
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
    table_col_names = [x for x in get_table_column_names(database_name, table_name) if x not in geom_col_names]

    # Specify SQL query - read all of the selected non-geom columns
    selected_col_names = [x for x in col_names if x not in geom_col_names] if col_names else table_col_names

    sql_query = 'SELECT {} FROM {}."{}"'.format(
        ', '.join('"' + tbl_col_name + '"' for tbl_col_name in selected_col_names), schema_name, table_name)

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
            geom_sql_query = 'SELECT "{}".STAsText() FROM {}."{}"'.format(geom_col_names[0], schema_name, table_name)
        else:
            geom_sql_query = 'SELECT {} FROM {}."{}"'.format(
                ', '.join('"' + x + '".STAsText()' for x in geom_col_names), schema_name, table_name)
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
