"""

In order to connect to the database you use the connect method of the Connection object. To get connected:

- Using pyodbc (or pypyodbc)
    connect_string = 'driver={drivername};server=servername;database=databaseName;uid=username;pwd=password'
    conn = pyodbc.connect(connect_string)  # equivalent to: pypyodbc.connect(connect_string)
- Using SQLAlchemy connectable,
    conn_string = 'mssql+pyodbc:///?odbc_connect=%s' % quote_plus(connect_string)
    engine = sqlalchemy.create_engine(conn_string)
    conn = engine.connect()

"""

import collections.abc
import functools
import itertools
import operator
import os
import urllib.parse

import pandas as pd
import pyhelpers.store
import pyodbc
import shapely.wkt
import sqlalchemy

# ====================================================================================================================
""" Establish a connection to a database server """


# Windows authentication for reading data from the databases
def use_windows_authentication():
    """
    :return:

    The trusted_connection setting indicates whether to use Windows Authentication Mode for login validation or not.
    'Trusted_Connection=yes' specifies the user used to establish this connection. In Microsoft implementations,
    this user account is a Windows user account.
    """
    win_str = 'Trusted_Connection=yes;'
    return win_str


# Database driver
def specify_database_driver():
    """
    :return:

    Ref: https://github.com/mkleehammer/pyodbc/wiki/Connecting-to-SQL-Server-from-Windows

    Using an ODBC driver

    Microsoft have written and distributed multiple ODBC drivers for SQL Server:

    {SQL Server} - released with SQL Server 2000
    {SQL Native Client} - released with SQL Server 2005 (also known as version 9.0)
    {SQL Server Native Client 10.0} - released with SQL Server 2008
    {SQL Server Native Client 11.0} - released with SQL Server 2012
    {ODBC Driver 11 for SQL Server} - supports SQL Server 2005 through 2014
    {ODBC Driver 13 for SQL Server} - supports SQL Server 2005 through 2016
    {ODBC Driver 13.1 for SQL Server} - supports SQL Server 2008 through 2016
    {ODBC Driver 17 for SQL Server} - supports SQL Server 2008 through 2017

    Note that the "SQL Server Native Client ..." and earlier drivers are deprecated and should not be used for new
    development.
    """
    dri_str = 'DRIVER={ODBC Driver 17 for SQL Server};'
    return dri_str


# Database server name
def specify_server_name():
    """
    :return:

    Alternative methods to get the computer name:
    import platform; platform.node(). Or, import socket; socket.gethostname()
    """
    server_name = os.environ['COMPUTERNAME']
    # server_name += '\\SQLEXPRESS'
    ser_str = 'SERVER={};'.format(server_name)
    return ser_str


# Database name
def specify_database_name(database_name):
    """
    :param database_name:
    :return:
    """
    dbn_str = 'DATABASE={};'.format(database_name)
    return dbn_str


# Create a SQLAlchemy connectable engine to MS SQL Server
def create_mssql_connectable_engine(database_name):
    """
    :param database_name: [str]
    :return: [sqlalchemy.engine.base.Engine]

    Connect string format: 'mssql+pyodbc://<username>:<password>@<dsn_name>'
    """
    conn_str = specify_database_driver() + specify_server_name() + specify_database_name(database_name) + \
        use_windows_authentication()
    db_engine = sqlalchemy.create_engine('mssql+pyodbc:///?odbc_connect=%s' % urllib.parse.quote_plus(conn_str))
    return db_engine


# Establish a SQLAlchemy connection to MS SQL Server
def establish_mssql_connection(database_name, mode=None):
    """
    :param database_name: [str]
    :param mode: [str]
    :return:
    """
    if not mode:  # (default)
        db_engine = create_mssql_connectable_engine(database_name)
        db_conn = db_engine.connect()
    else:  # i.e. to use directly 'pyodbc'
        conn_str = use_windows_authentication() + specify_database_driver() + specify_server_name()
        db_conn = pyodbc.connect(conn_str, database=database_name)
    return db_conn


# Create a pyodbc cursor
def create_mssql_db_cursor(database_name):
    """
    :param database_name:
    :return:
    """
    db_conn = establish_mssql_connection(database_name, mode='pyodbc')
    db_cursor = db_conn.cursor()
    return db_cursor


# ====================================================================================================================
""" Retrieve information """


# Get a list of table names in a database
def get_table_names(database_name, schema_name='dbo', table_type='TABLE'):
    """
    This function gets a list of names of tables in a database, given a specific table type.
    The table types could include 'TABLE', 'VIEW', 'SYSTEM TABLE', 'ALIAS', 'GLOBAL TEMPORARY', 'SYNONYM',
    'LOCAL TEMPORARY', or a data source-specific type name

    :param database_name: [str] name of the database queried (also the catalog name)
    :param schema_name: [str] name of schema, e.g. 'dbo', 'sys'
    :param table_type: [str] table type
    :return: [list] a list of names of the tables in the queried database
    """
    # Create a cursor with a direct connection to the queried database
    db_cursor = create_mssql_db_cursor(database_name)
    # Get a results set of tables defined in the data source
    table_list = [t.table_name for t in db_cursor.tables(schema=schema_name, tableType=table_type)]
    # Close the connection
    db_cursor.close()
    # Return a list of the names of table names
    return table_list


# Get a list of column names of a given table in a database
def get_table_column_names(database_name, table_name, schema_name='dbo'):
    """
    :param database_name: [str] name of a database
    :param table_name: [str] name of a queried table from the given database
    :param schema_name: [str]
    :return: [list] a list of column names
    """
    db_cursor = create_mssql_db_cursor(database_name)
    col_names = [x.column_name for x in db_cursor.columns(table_name, schema=schema_name)]
    db_cursor.close()
    return col_names


# Get the primary keys of each table in a database
def get_table_primary_keys(database_name, table_name=None, schema_name='dbo', table_type='TABLE'):
    """
    :param database_name: [str] name of a database
    :param table_name: [str] name of a queried table from the given database
    :param schema_name: [str]
    :param table_type: [str] table type
    :return: [dict] {table_name: primary keys}
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


# ====================================================================================================================
""" Read table data """


# Get all data from a given table in a specific database (Function is limited.)
def read_table_by_name(database_name, table_name, schema_name='dbo', col_names=None, parse_dates=None, chunk_size=None,
                       index_col=None, coerce_float=True, save_as=None, data_dir=None):
    """
    :param database_name: [str] name of a database
    :param table_name: [str] name of a queried table from the given database
    :param schema_name: [str] default 'dbo'
    :param col_names:
    :param parse_dates:
    :param chunk_size:
    :param index_col:
    :param coerce_float:
    :param save_as:
    :param data_dir:
    :return: [pandas.DataFrame] the queried table as a DataFrame
    """
    # Connect to the queried database
    db_conn = establish_mssql_connection(database_name)
    # Create a pandas.DataFrame of the queried table_name
    table_data = pd.read_sql_table(table_name=table_name, con=db_conn, schema=schema_name, columns=col_names,
                                   chunksize=chunk_size, index_col=index_col, parse_dates=parse_dates,
                                   coerce_float=coerce_float)
    # Disconnect the database
    db_conn.close()

    if save_as:
        path_to_file = os.path.join(os.path.realpath(data_dir if data_dir else ''), table_name + save_as)
        pyhelpers.store.save(table_data, path_to_file)

    # Return the data frame of the queried table
    return table_data


# Get data from a table in a specific database by SQL query ==========================================================
def read_table_by_query(database_name, table_name, schema_name='dbo', col_names=None, parse_dates=None, chunk_size=None,
                        index_col=None, coerce_float=None, save_as=None, data_dir=None):
    """
    :param database_name: [str] name of a database
    :param table_name: [str] name of a table within the database
    :param schema_name: [str] e.g. 'dbo'
    :param col_names: [iterable or None(default)] e.g. a list of column names to retrieve; 'None' for all columns
    :param parse_dates: [bool; None(default)]
    :param chunk_size: [int; None(default)]
    :param index_col:
    :param coerce_float:
    :param save_as: [str or None(default)]
    :param data_dir: [str or None(default)]
    :return: [pandas.DataFrame] the queried data as a DataFrame
    """
    if col_names:
        assert isinstance(col_names, collections.abc.Iterable) and all(isinstance(x, str) for x in col_names)
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
    table_data = pd.read_sql(sql=sql_query, con=db_conn, columns=col_names, parse_dates=parse_dates,
                             chunksize=chunk_size, index_col=index_col, coerce_float=coerce_float)
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
        geom_data = pd.read_sql(geom_sql_query, db_conn, chunksize=chunk_size)
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
        pyhelpers.store.save(table_data, path_to_file)

    return table_data


# Read a table chunk-wise from a database ============================================================================
def save_table_by_chunk(database_name, table_name, schema_name='dbo', col_names=None, parse_dates=None,
                        chunk_size=1000000, index_col=None, coerce_float=None, save_as=".pickle", data_dir=None):
    """
    :param database_name: [str] name of a database to query
    :param table_name: [str] name of a queried table from the given database
    :param schema_name: [str]
    :param col_names:
    :param parse_dates: [list] or [dict], default: None
    :param chunk_size: [int] size of a single chunk to read
    :param index_col: [str] name of a column set to be the index
    :param coerce_float:
    :param save_as: [str]
    :param data_dir: [NoneType] or [str]
    """
    assert isinstance(save_as, str) and save_as in (".pickle", ".csv", ".xlsx", ".txt")
    if col_names:
        assert isinstance(col_names, collections.abc.Iterable) and all(isinstance(x, str) for x in col_names)
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
        table_data = pd.read_sql(sql=sql_query, con=db_conn, columns=col_names, parse_dates=parse_dates,
                                 chunksize=chunk_size, index_col=index_col, coerce_float=coerce_float)
        for tbl_id, tbl_dat in enumerate(table_data):
            path_to_file = os.path.join(dat_dir, table_name + "_{}".format(tbl_id + 1) + save_as)
            pyhelpers.store.save(tbl_dat, path_to_file, sheet_name="Sheet_{}".format(tbl_id + 1))
    else:
        # Read the queried table_name into a pandas.DataFrame
        table_data = pd.read_sql(sql=sql_query, con=db_conn, columns=col_names, parse_dates=parse_dates,
                                 chunksize=chunk_size, index_col=index_col, coerce_float=coerce_float)
        tbl_chunks = [tbl_dat for tbl_dat in table_data]

        if len(geom_col_names) == 1:
            geom_sql_query = 'SELECT "{}".STAsText() FROM {}."{}"'.format(geom_col_names[0], schema_name, table_name)
        else:
            geom_sql_query = 'SELECT {} FROM {}."{}"'.format(
                ', '.join('"' + x + '".STAsText()' for x in geom_col_names), schema_name, table_name)
        geom_data = pd.read_sql(geom_sql_query, db_conn, chunksize=chunk_size)
        geom_chunks = [geom_dat.applymap(shapely.wkt.loads) for geom_dat in geom_data]

        counter = 0
        for tbl_dat, geom_dat in zip(tbl_chunks, geom_chunks):
            path_to_file = os.path.join(dat_dir, table_name + "_{}".format(counter + 1) + save_as)
            pyhelpers.store.save(tbl_dat.join(geom_dat), path_to_file, sheet_name="Sheet_{}".format(counter + 1))
            counter += 1

    # Disconnect the database
    db_conn.close()
