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

import functools
import operator
import os
import urllib.parse

import pandas as pd
import pyodbc
import sqlalchemy

from utils import cdd, save


# Windows authentication for reading data from the databases =========================================================
def windows_authentication():
    """
    The trusted_connection setting indicates whether to use Windows Authentication Mode for login validation or not.
    'Trusted_Connection=yes' specifies the user used to establish this connection. In Microsoft implementations,
    this user account is a Windows user account.
    """
    return 'Trusted_Connection=yes;'


# Database driver ====================================================================================================
def database_driver():
    """
    DRIVER={SQL Server Native Client 11.0}
    """
    return 'DRIVER={SQL Server};'


# Database server name ===============================================================================================
def database_server():
    """
    Alternative methods to get the computer name

    import platform
    platform.node()

    Or,

    import socket
    socket.gethostname()
    """
    return 'SERVER={};'.format(os.environ['COMPUTERNAME'] + '\\SQLEXPRESS')


# Database name ======================================================================================================
def database_name(db_name):
    return 'DATABASE={};'.format(db_name)


# Create a SQLAlchemy Connectable to MSSQL Server ====================================================================
def sqlalchemy_connectable(db_name):
    """
    Connect string format: 'mssql+pyodbc://<username>:<password>@<dsnname>'
    """
    p_str = windows_authentication() + database_driver() + database_server() + database_name(db_name)
    engine = sqlalchemy.create_engine('mssql+pyodbc:///?odbc_connect=%s' % urllib.parse.quote_plus(p_str))
    return engine.connect()


# Get a list of table names in a database ============================================================================
def get_table_names(db_name, schema='dbo', table_type='TABLE'):
    """
    :param db_name: [str] name of the database queried (also the catalog name)
    :param schema: [str] name of schema, e.g. 'dbo', 'sys'
    :param table_type: [str] table type
    :return: [list] a list of names of the tables in the queried database

    This function gets a list of names of tables in a database, given a specific table type.
    The table types could include 'TABLE', 'VIEW', 'SYSTEM TABLE', 'ALIAS', 'GLOBAL TEMPORARY', 'SYNONYM',
    'LOCAL TEMPORARY', or a data source-specific type name

    """
    # Make a direct connection to the queried database
    p_str = windows_authentication() + database_driver() + database_server()
    conn_db = pyodbc.connect(p_str, database=db_name)
    # Create a cursor
    db_cursor = conn_db.cursor()
    # Get a results set of tables defined in the data source
    table_cursor = db_cursor.tables(schema=schema, tableType=table_type)
    # Return a list of the names of table names
    return [t.table_name for t in table_cursor]


# Get a list of column names of a given table in a database ==========================================================
def get_table_colnames(db_name, table_name):
    """
    :param db_name: [str] name of a database
    :param table_name: [str] name of a queried table from the given database
    :return: [list] a list of column names
    """
    # Make a direct connection to the queried database
    p_str = windows_authentication() + database_driver() + database_server()
    conn_db = pyodbc.connect(p_str, database=db_name)
    db_cursor = conn_db.cursor()
    colnames = [x.column_name for x in db_cursor.columns(table_name)]
    conn_db.close()
    return colnames


# Get the primary keys of each table in a database ===================================================================
def get_table_primary_keys(db_name, table_name=None, schema='dbo', table_type='TABLE'):
    """
    :param db_name: [str] name of a database
    :param table_name: [str] name of a queried table from the given database
    :param schema: [str]
    :param table_type: [str] table type
    :return: [dict] {table_name: primary keys}
    """
    p_str = windows_authentication() + database_driver() + database_server()
    conn_db = pyodbc.connect(p_str, database=db_name)
    db_cursor = conn_db.cursor()
    table_names = [table.table_name for table in db_cursor.tables(schema=schema, tableType=table_type)]
    # Get primary keys for each table
    pri_keys = [{k.table_name: k.column_name} for t in table_names for k in db_cursor.primaryKeys(t)]
    keys = functools.reduce(operator.or_, (set(d.keys()) for d in pri_keys), set())
    # Disconnect the database
    conn_db.close()
    tk_dict = dict((k, [d[k] for d in pri_keys if k in d]) for k in keys)
    if table_name is not None:
        return tk_dict[table_name]
    else:
        return tk_dict


# Get data from a table in a specific database by SQL query ==========================================================
def read_table_by_query(db_name, sql_query):
    """
    :param db_name: [str] name of a database
    :param sql_query: [str] SQL query to get data from the given database
    :return:[pandas.DataFrame] the queried data as a DataFrame
    """
    # Connect to the queried database
    conn_db = sqlalchemy_connectable(db_name)
    # Create a pandas.DataFrame of the queried table_name
    table_data = pd.read_sql_query(sql=sql_query, con=conn_db)
    # Disconnect the database
    conn_db.close()
    # Return the data frame of the queried table
    return table_data


# Get all data from a given table in a specific database =============================================================
def read_table_by_name(db_name, table_name, schema='dbo'):
    """
    :param db_name: [str] name of a database
    :param table_name: [str] name of a queried table from the given database
    :param schema: [str] default 'dbo'
    :return: [pandas.DataFrame] the queried table as a DataFrame
    """
    # Connect to the queried database
    conn_db = sqlalchemy_connectable(db_name)
    # Create a pandas.DataFrame of the queried table_name
    table_data = pd.read_sql_table(table_name=table_name, con=conn_db, schema=schema)
    # Disconnect the database
    conn_db.close()
    # Return the data frame of the queried table
    return table_data


# Get the primary key ================================================================================================
def get_pri_keys(db_name, table_name, schema='dbo', table_type='TABLE'):
    assert table_name is not None, '"table_name" must be specified as a string.'
    return get_table_primary_keys(db_name, table_name, schema, table_type)


# Change directory to "Data\\METEX\\Database" ========================================================================
def cdd_metex_db(*directories):
    path = cdd("METEX", "Database")
    for directory in directories:
        path = os.path.join(path, directory)
    return path


# Change directory to "Data\\Vegetation\\Database" ===================================================================
def cdd_veg_db(*directories):
    path = cdd("Vegetation", "Database")
    for directory in directories:
        path = os.path.join(path, directory)
    return path


# Read a table chunk-wise from a database ============================================================================
def read_table_by_part(db_name, table_name, index_col=None, parse_dates=None, chunk_size=100000,
                       save_as=None, save_by_chunk=False, save_by_value=False):
    """
    :param db_name: [str] name of a database to query
    :param table_name: [str] name of a queried table from the given database
    :param index_col: [str] name of a column set to be the index
    :param parse_dates: [list] or [dict], default: None
    :param chunk_size: [int] size of a single chunk to read
    :param save_as: [NoneType] or [str]
    :param save_by_chunk: [str]
    :param save_by_value: [str] ext of file which the queried df is saved as
    :return:[pandas.DataFrame] the queried data as a DataFrame

    This function is used to read a table chunk-wise from a database,
    when the table might be too large for pd.read_sql to process once.

    """
    # Make a direct connection to the queried database
    conn_metex = sqlalchemy_connectable(db_name)
    # Specify the SQL query that gets all data of a given table
    sql_query = "SELECT * FROM dbo.{}".format(table_name)
    # Generate a query iterator
    chunks = pd.read_sql_query(sql_query, conn_metex, index_col, parse_dates=parse_dates, chunksize=chunk_size)
    # Retrieve the full DataFrame
    if index_col is not None:
        table_data = pd.DataFrame(pd.concat(chunks, axis=0))
        table_data.sort_index(ascending=True, inplace=True)
    else:
        table_data = pd.DataFrame(pd.concat(chunks, axis=0, ignore_index=True))

    # Save the full DataFrame and any DataFrame chunks?
    if save_as:
        # (Note .xlsx file can handle a maximum of 1,048,576 observations.)
        save(table_data, cdd_metex_db("Tables_original", table_name + save_as))

    if save_by_chunk:
        chunk_path = cdd_metex_db("Tables_original", table_name + '_by_chunk')
        os.mkdir(chunk_path)
        if save_by_chunk is True:
            for chunk_id, chunk_data in enumerate(chunks):
                save(chunk_data, os.path.join(chunk_path, 'chunk{}'.format(chunk_id) + '.pickle'))
        elif save_by_chunk == '.xlsx':
            # The DataFrame will be saved as .xlsx file, with each worksheet taking in one "chunk" of the data
            excel_writer = pd.ExcelWriter(os.path.join(chunk_path, table_name + '_by_chunk.xlsx'), engine='xlsxwriter')
            for chunk_id, chunk_data in enumerate(chunks):
                if index_col is not None:
                    chunk_data.reset_index(inplace=True)
                chunk_data.to_excel(excel_writer, sheet_name='chunk{}'.format(chunk_id), index=False)
            excel_writer.save()

    if save_by_value:  # This requires that the corresponding column is set to be the index
        value_path = cdd_metex_db("Tables_original", table_name + '_by_value')
        os.mkdir(value_path)
        if isinstance(save_by_value, list) or isinstance(save_by_value, tuple):  # e.g. save_by_value = (13844, 13852)
            for v in save_by_value:
                save(table_data.ix[v], os.path.join(value_path, '{}'.format(v) + '.pickle'))
        elif save_by_value == '.xlsx':
            excel_writer = pd.ExcelWriter(cdd_metex_db(table_name + '_by_value.xlsx'), engine='xlsxwriter')
            for v in set(table_data.index):
                table_data.ix[v].to_excel(excel_writer, sheet_name=str(v), index=False)
            excel_writer.save()

    return table_data


# Read tables available in Database ==================================================================================
def read_metex_table(table_name, schema='dbo', index_col=None, route=None, weather=None, save_as=None, update=False):
    """
    :param table_name: [str] name of a queried table from the Database
    :param schema: [str] 'dbo', as default
    :param index_col: [str] name of a column that is set to be the index
    :param route: [str] name of the specific Route
    :param weather: [str] name of the specific weather category
    :param save_as: [str]
    :param update:
    :return: [pandas.DataFrame] the queried data as a DataFrame
    """
    table = schema + '.' + table_name
    # Connect to the queried database
    conn_metex = sqlalchemy_connectable(db_name='NR_METEX')
    # Specify possible scenarios:
    if not route and not weather:
        sql_query = "SELECT * FROM {}".format(table)  # Get all data of a given table
    elif route and not weather:
        sql_query = "SELECT * FROM {} WHERE Route = '{}'".format(table, route)  # given Route
    elif route is None and weather is not None:
        sql_query = "SELECT * FROM {} WHERE WeatherCategory = '{}'".format(table, weather)  # given weather category
    else:
        # Get all data of a table, given Route and weather category e.g. data about wind-related events on Anglia Route
        sql_query = "SELECT * FROM {} WHERE Route = '{}' AND WeatherCategory = '{}'".format(table, route, weather)
    # Create a pandas.DataFrame of the queried table
    table_data = pd.read_sql_query(sql=sql_query, con=conn_metex, index_col=index_col)
    # Disconnect the database
    conn_metex.close()
    if save_as:
        path_to_file = cdd_metex_db("Tables_original", table_name + save_as)
        if not os.path.isfile(path_to_file) or update:
            save(table_data, cdd_metex_db("Tables_original", table_name + save_as))
    return table_data


# Read tables available in NR_VEG database ===========================================================================
def read_veg_table(table_name, schema='dbo', index_col=None, route=None, save_as=None, update=False):
    """
    :param table_name: [str]
    :param schema: [str]
    :param index_col: [str] or None
    :param route: [str] or None
    :param save_as: [str] or None
    :param update: [bool]
    :return: [pandas.DataFrame]
    """
    table = schema + '.' + table_name
    # Make a direct connection to the queried database
    conn_veg = sqlalchemy_connectable(db_name='NR_VEG')
    if route is None:
        sql_query = "SELECT * FROM {}".format(table)  # Get all data of a given table
    else:
        sql_query = "SELECT * FROM {} WHERE Route = '{}'".format(table, route)  # given a specific Route
    # Create a pandas.DataFrame of the queried table
    data = pd.read_sql(sql=sql_query, con=conn_veg, index_col=index_col)
    # Disconnect the database
    conn_veg.close()
    # Save the DataFrame as a worksheet locally?
    if save_as:
        path_to_file = cdd_veg_db("Tables_original", table_name + save_as)
        if not os.path.isfile(path_to_file) or update:
            save(data, path_to_file)
    # Return the data frame of the queried table
    return data
