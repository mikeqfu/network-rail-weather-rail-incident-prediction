"""
Communicate with a PostgreSQL/MSSQL server.
"""

import copy
import gc
import json
import os
import re
import sys

import numpy as np
import pandas as pd
from pyhelpers._cache import _print_failure_message
from pyhelpers.dbms import MSSQL, PostgreSQL
from pyhelpers.dirs import cd, cdd
from pyhelpers.ops import confirmed
from sqlalchemy.dialects.postgresql import BYTEA


class WxRailIncidentsPred(PostgreSQL):
    """
    A class for communicating with the project's database in a `PostgreSQL`_ server.

    .. _`PostgreSQL`: https://www.postgresql.org/
    """

    def __init__(self, host=None, port=None, username=None, password=None,
                 database_name='NR_WxRailIncidentsPred', **kwargs):
        """
        Constructor method, which creates proxy object allowing us to access methods of
        the base class `pyhelpers.dbms.PostgreSQL
        <https://pyhelpers.readthedocs.io/en/latest/_generated/pyhelpers.dbms.PostgreSQL.html>`_

        :param host: The database host; defaults to ``None``.
        :type host: str | None
        :param port: The database port; defaults to ``None``.
        :type port: int | None
        :param username: The database username; defaults to ``None``.
        :type username: str | None
        :param password: The database password; defaults to ``None``.
        :type password: str | int | None
        :param database_name: The name of the database; defaults to ``NR_WxRailIncidentsPred``.
        :type database_name: str
        :param kwargs: [Optional] parameters of the class `pyhelpers.dbms.PostgreSQL`_.

        .. _`pyhelpers.dbms.PostgreSQL`:
            https://pyhelpers.readthedocs.io/en/latest/_generated/pyhelpers.dbms.PostgreSQL.html

        **Examples**::

            >>> from src.utils import WxRailIncidentsPred
            >>> db_instance = WxRailIncidentsPred()
            Password (postgres@localhost:5432): ***
            Connecting <username>:***@<host>:<port>/NR_WxRailIncidentsPred ... Successfully.
            >>> db_instance.database_name
            'NR_WxRailIncidentsPred'
        """

        if host not in {'localhost', '127.0.0.1'}:
            try:  # Load credentials from the .credentials file
                with open(os.path.join(".credentials"), "r") as f:
                    credentials = json.load(f)

                kwargs.update(credentials)
                super().__init__(**kwargs)

            except FileNotFoundError:
                print("The credential file does not exist.")

                credentials = {
                    'host': host,
                    'port': port,
                    'username': username,
                    'password': password,
                    'database_name': database_name,
                }
                kwargs.update(credentials)
                super().__init__(**kwargs)


class Handler:
    """
    A base class for handling data, especially via the subpackage :mod:`~src.preprocessor`.
    """

    #: Name of the data.
    DATA_NAME: str = 'Data'
    #: Pathname of the data directory for this specific class.
    DATA_DIR: str = cdd()
    #: Name of the Database in Microsoft SQL Server.
    MSSQL_DATABASE_NAME: str = ''
    #: Schema name for the original data in the PostgreSQL Database.
    POSTGRES_SCHEMA_NAME: str = copy.copy(MSSQL_DATABASE_NAME)
    #: Schema name for the preprocessed data.
    SCHEMA_NAME: str = POSTGRES_SCHEMA_NAME + '_prep'

    def __init__(self, db_instance=None):
        """
        :param db_instance: An instance of the class `pyhelpers.dbms.PostgreSQL`_,
            connecting to the project database in a PostgreSQL server; defaults to ``None``.
        :type db_instance: WxRailIncidentsPred | None

        :ivar MSSQL | None mssql: An instance of the class `pyhelpers.dbms.MSSQL`_,
            connecting to a Microsoft SQL Server database.
        :ivar PostgreSQL | None db_instance: An instance of the class `pyhelpers.dbms.PostgreSQL`_,
            connecting to the project database in a PostgreSQL server; defaults to ``None``.

        .. _`pyhelpers.dbms.PostgreSQL`:
            https://pyhelpers.readthedocs.io/en/latest/_generated/pyhelpers.dbms.PostgreSQL.html
        .. _`pyhelpers.dbms.MSSQL`:
            https://pyhelpers.readthedocs.io/en/latest/_generated/pyhelpers.dbms.MSSQL.html
        """

        self.mssql = None

        self.db_instance = db_instance

    def cdd(self, *sub_dir, mkdir=False):
        """
        Change directory to data (sub)directories / a file.

        :param sub_dir: Name of directory or names of directories (and/or a filename).
        :type sub_dir: str
        :param mkdir: Whether to create a directory; defaults to ``False``.
        :type mkdir: bool
        :return: Pathname of the data (sub)directories or a specific file.
        :rtype: str
        """

        path = cd(self.DATA_DIR, *sub_dir, mkdir=mkdir)

        return path

    # == Read table data from the Microsoft SQL Database ===========================================

    def read_mssql_table(self, table_name, route_name=None, weather_category=None, **kwargs):
        """
        Read data of a table in the Microsoft SQL Server database of the project.

        :param table_name: Name of a table.
        :type table_name: str
        :param route_name: Name of a Network Rail's Route;
            when ``route_name=None`` (default), it takes all available Routes.
        :type route_name: str | None
        :param weather_category: Weather category;
            when ``weather_category=None`` (default), it takes all available weather categories.
        :type weather_category: str | None
        :param kwargs: [Optional] parameters of the function `pandas.read_sql()`_.
        :return: Data of the queried table stored in the project database.
        :rtype: pandas.DataFrame

        .. _`pandas.read_sql()`:
            https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_sql.html
        """

        if self.mssql is None:
            self.mssql = MSSQL(database_name=self.MSSQL_DATABASE_NAME)

        sql_query = f'SELECT * FROM %s' % f'dbo.[{table_name}]'

        if route_name is not None and weather_category is None:
            sql_query += f" WHERE [Route]='{route_name}'"  # given Route
        elif route_name is None and weather_category is not None:
            sql_query += f" WHERE [WeatherCategory]='{weather_category}'"
        elif route_name and weather_category:
            sql_query += f" WHERE [Route]='{route_name}' AND [WeatherCategory]='{weather_category}'"

        index_col = self.mssql.get_primary_keys(table_name)

        table_data = pd.read_sql(sql_query, con=self.mssql.engine, index_col=index_col, **kwargs)

        return table_data

    # == Write/read table data to/from the PostgreSQL Database =====================================

    @staticmethod
    def specify_table_name(table_name):
        if table_name.upper() == table_name:
            table_name_ = table_name.lower()
        elif table_name.lower() == table_name:
            table_name_ = table_name
        else:
            table_name_ = '_'.join(re.findall(r'[A-Z][^A-Z]*', table_name)).lower()

        return table_name_

    def specify_sql_query(self, table_name, raw=False):
        schema_name = self.POSTGRES_SCHEMA_NAME if raw else self.SCHEMA_NAME
        sql_query = 'SELECT * FROM {}'.format(f'"{schema_name}"."{table_name}"')

        return sql_query

    def dump_preprocessed_data(self, data, table_name, verbose, pkey=None, index=True,
                               if_exists='replace', **kwargs):
        if verbose:
            tbl = f'"{self.SCHEMA_NAME}"."{table_name}"'
            if self.db_instance.table_exists(table_name, self.SCHEMA_NAME):
                msg = "  Updating "
            else:
                msg = "  Importing "
            print(msg + f"{tbl}", end=" ... ")

        import_args = {
            'data': data,
            'table_name': table_name,
            'schema_name': self.SCHEMA_NAME,
            'if_exists': if_exists,
            'index': index,
            'method': self.db_instance.psql_insert_copy,
            'confirmation_required': False,
        }
        try:
            kwargs.update(import_args)
            self.db_instance.import_data(**kwargs)

            if pkey is None:
                primary_keys = self.db_instance.get_primary_keys(
                    table_name, self.POSTGRES_SCHEMA_NAME)
            else:
                primary_keys = pkey.copy()

            if primary_keys:
                self.db_instance.add_primary_keys(
                    primary_keys=primary_keys, table_name=table_name, schema_name=self.SCHEMA_NAME)

            if verbose:
                print("Done.")

        except Exception as e:
            _print_failure_message(e)

    def read_data(self, table_name, ivar_name=None, index_col=None, update=False, verbose=False,
                  ret_data=False, **kwargs):
        if self.db_instance is None:
            self.db_instance = WxRailIncidentsPred()

        if ivar_name is None:
            instance_var_name = self.specify_table_name(table_name)
        else:
            instance_var_name = copy.copy(ivar_name)

        if self.db_instance.table_exists(table_name, self.SCHEMA_NAME) and not update:
            sql_query = self.specify_sql_query(table_name)

            if index_col is None:
                index_column_names = self.db_instance.get_primary_keys(
                    table_name=table_name, schema_name=self.SCHEMA_NAME)
            else:
                index_column_names = copy.copy(index_col)

            kwargs.update({'index_col': index_column_names})
            data = self.db_instance.read_sql_query(sql_query, **kwargs)

            text_cols = [x for x in data.columns if data[x].dtype.name == 'object']
            if bool(text_cols):
                data.loc[:, text_cols] = data[text_cols].fillna('')

            # if data.index.name is None:
            #     data.set_index(index_column_names, inplace=True)

            data.sort_index(inplace=True)

        else:
            data = getattr(self, '_' + instance_var_name)(verbose=verbose)

        self.__setattr__(instance_var_name, data)

        if ret_data:
            return data


def mssql_to_postgresql(mssql_db_name, postgres_db_name, chunk_size=None, excluded_tables=None,
                        file_tables=False, memory_threshold=2., update=False,
                        confirmation_required=True, verbose=True):
    """
    Copy tables of a database from a Microsoft SQL server to a PostgreSQL server.

    :param mssql_db_name: name of a Microsoft SQL (source) Database
    :type mssql_db_name: str
    :param postgres_db_name: name of a PostgreSQL (destination) Database
    :type postgres_db_name: str
    :param chunk_size: number of rows in each batch to be read/written at a time;
        defaults to ``None``.
    :type chunk_size: int | None
    :param excluded_tables: names of Tables that are excluded from the data migration
    :type excluded_tables: list | None
    :param file_tables: whether to include FileTables; defaults to ``False``.
    :type file_tables: bool
    :param memory_threshold: threshold (in GiB) beyond which the data is migrated by partitions;
        defaults to ``2.``
    :type memory_threshold: float | int
    :param update: whether to redo the transfer between the Database servers; defaults to ``False``.
    :type update: bool
    :param confirmation_required: whether asking for confirmation to proceed; defaults to ``True``.
    :type confirmation_required: bool
    :param verbose: Whether to print relevant information; defaults to ``True``.
    :type verbose: bool | int

    **Examples**::

        >>> from src.utils import mssql_to_postgresql
        >>> mssql_to_postgresql(
        ...     mssql_db_name='NR_Vegetation_20141031', postgres_db_name='NR_WxRailIncidentsPred')
        >>> mssql_to_postgresql(
        ...     mssql_db_name='NR_METEx_20150331', postgres_db_name='NR_WxRailIncidentsPred')
        >>> mssql_to_postgresql(
        ...     mssql_db_name='NR_METEx_20190203', postgres_db_name='NR_WxRailIncidentsPred')
    """

    task_msg = f'from "{mssql_db_name}" (MSSQL) to "{postgres_db_name}" (PostgreSQL)'

    if confirmed(f'To copy Tables {task_msg}\n?', confirmation_required=confirmation_required):

        # MSSQL
        mssql = MSSQL(database_name=mssql_db_name)
        mssql_table_names = mssql.get_table_names()

        # PostgreSQL
        postgres = WxRailIncidentsPred(database_name=postgres_db_name)
        postgres_schema = copy.copy(mssql_db_name)

        if verbose:
            if confirmation_required:
                print("Processing Tables ... ")
            else:
                print(f"Copying Tables {task_msg} ... ")

        excl_tbl_names = [] if excluded_tables is None else copy.copy(excluded_tables)

        if not file_tables:
            file_table_names = mssql.get_file_tables()
            excl_tbl_names += file_table_names

        mssql_table_names = [x for x in mssql_table_names if x not in excl_tbl_names]

        table_counter, table_total = 1, len(mssql_table_names)
        error_log = {}
        for table_name in mssql_table_names:
            counter_msg = f"({table_counter}/{table_total})"

            postgres_table_exists = postgres.table_exists(table_name, schema_name=postgres_schema)

            if not postgres_table_exists or update:
                try:
                    if verbose:
                        postgresql_tbl = f'"{postgres_schema}"."{table_name}"'
                        if postgres_table_exists:
                            msg = f"Updating {postgresql_tbl}"
                        else:
                            msg = f"Copying [{table_name}] to {postgresql_tbl}"
                        print(f'\t{counter_msg} ' + msg, end=" ... ")

                    row_count = mssql.get_row_count(table_name)

                    if row_count >= 1000000:

                        if chunk_size is None:
                            chunk_size = 1000000

                    source_data = mssql.read_table(table_name=table_name, chunk_size=chunk_size)

                    col_type = {}

                    check_dtypes = ['hierarchyid', 'varbinary']
                    check_dtypes_rslt = mssql.has_dtypes(table_name, dtypes=check_dtypes)

                    for dtype, if_exists, col_names in check_dtypes_rslt:
                        if if_exists:
                            if dtype == 'hierarchyid':
                                source_data.loc[:, col_names] = source_data[col_names].applymap(
                                    lambda x: str(x).replace('\\', '\\\\'))
                            col_type.update(dict(zip(col_names, [BYTEA] * len(col_names))))

                    memory_usage = sys.getsizeof(source_data) / 1024 ** 3
                    if memory_usage > memory_threshold:
                        source_data = np.array_split(source_data, memory_usage // memory_threshold)

                        i = 0
                        while i < len(source_data):
                            postgres.import_data(
                                source_data[i], table_name=table_name, schema_name=postgres_schema,
                                if_exists='replace' if i == 0 else 'append',
                                method=postgres.psql_insert_copy, chunk_size=chunk_size,
                                col_type=col_type, confirmation_required=False, verbose=False)

                            gc.collect()
                            i += 1

                    else:
                        postgres.import_data(
                            source_data, table_name=table_name, schema_name=postgres_schema,
                            if_exists='replace', method=postgres.psql_insert_copy,
                            chunk_size=chunk_size, col_type=col_type, confirmation_required=False,
                            verbose=False)

                    # Get primary keys from MSSQL
                    primary_keys = mssql.get_primary_keys(table_name, table_type='TABLE')

                    # Specify primary keys in PostgreSQL
                    postgres_pkey = postgres.get_primary_keys(
                        table_name, schema_name=postgres_schema)
                    if not postgres_pkey:
                        postgres.add_primary_keys(primary_keys, table_name, postgres_schema)

                    del source_data
                    gc.collect()

                    if verbose:
                        print("Done.")

                except Exception as e:
                    print("Failed.")

                    error_log.update({table_name: "{}".format(e)})

            else:
                if verbose:
                    postgresql_tbl = f'"{postgres_schema}"."{table_name}"'
                    print(f"\t{counter_msg} {postgresql_tbl} already exists.")

            table_counter += 1

        if verbose:
            print(f"\tProcess finished.")

        if bool(error_log):
            return error_log


def py2_etlalchemy_migrate(mssql_db_name, postgres_db_name, postgres_pwd=None, python2_exe=None,
                           confirmation_required=True, **kwargs):
    """
    Use etlalchemy (in Python 2) to migrate a Database from MSSQL to PostgreSQL.

    :param mssql_db_name: Name of the source database.
    :type mssql_db_name: str
    :param postgres_db_name: Name of the destination database.
    :type postgres_db_name: str
    :param postgres_pwd: Password to the PostgreSQL server.
    :type postgres_pwd: int | str | None
    :param python2_exe: Pathname of Python 2 executable.
    :type python2_exe: str | None
    :param confirmation_required:
    :type confirmation_required: bool

    **Examples**::

        >>> from src.utils import py2_etlalchemy_migrate
        >>> py2_etlalchemy_migrate('NR_Vegetation_20141031', 'NR_WxRailIncidentsPred')
    """

    cfm_msg = f'To copy tables from "{mssql_db_name}" (MSSQL) to "{postgres_db_name}" (PostgreSQL)'
    if confirmed(cfm_msg + "\n?", confirmation_required=confirmation_required):

        import urllib.parse
        import execnet.multi

        if python2_exe is None:
            python2_exe = "C:\\Program Files\\Python27\\python"

        server_name = os.environ['COMPUTERNAME']
        mssql_str = 'Trusted_Connection=yes;DRIVER={SQL Server};SERVER=%s;DATABASE=%s;' % (
            server_name, mssql_db_name)
        mssql_str = 'mssql+pyodbc:///?odbc_connect=%s' % urllib.parse.quote_plus(mssql_str)

        pgsql_str = 'postgresql+psycopg2://postgres:%s@localhost/%s' % (postgres_pwd, postgres_db_name)

        _ = PostgreSQL(password=postgres_pwd, database_name=postgres_db_name, **kwargs)

        gw = execnet.multi.makegateway("popen//python='%s'" % python2_exe)
        channel = gw.remote_exec(
            """
            import etlalchemy

            mssql_db = etlalchemy.ETLAlchemySource('%s')

            pgsql_db = etlalchemy.ETLAlchemyTarget('%s', drop_database=False)

            pgsql_db.addSource(mssql_db)
            pgsql_db.py2_etlalchemy_migrate()

            channel.send(None)
            """ % (mssql_str, pgsql_str)
        )

        channel.send(None)

        channel.receive()

        channel.close()


"""
(Script in Python 2)

import os
import urllib.parse

import etlalchemy


def migrate_db_mssql_to_postgresql(origin_db, destination_db):
    '''
    Migrate data from Microsoft SQL Server onto PostgreSQL.

    :param origin_db: name of source Database
    :type origin_db: str
    :param destination_db: name of Database, to which the source Database py2_etlalchemy_migrate
    :type destination_db: str
    '''

    def windows_authentication():
        return 'Trusted_Connection=yes;'

    def db_driver():
        return 'DRIVER={SQL Server};'

    def db_server():
        server_name = os.environ['COMPUTERNAME']
        return 'SERVER={};'.format(server_name)

    # Database name
    def database_name(db_name):
        return 'DATABASE={};'.format(db_name)

    mssql_str = windows_authentication() + db_driver() + db_server() + database_name(origin_db)
    mssql_str = 'mssql+pyodbc:///?odbc_connect=%s' % urllib.parse.quote_plus(mssql_str)
    mssql_db = etlalchemy.ETLAlchemySource(mssql_str)

    pgsql_pwd = int(raw_input('Password to connect PostgreSQL: '))
    pgsql_str = 'postgresql+psycopg2://postgres:{}@localhost/{}'.format(pgsql_pwd, destination_db)
    pgsql_db = etlalchemy.ETLAlchemyTarget(pgsql_str, drop_database=True)

    pgsql_db.addSource(mssql_db)
    pgsql_db.py2_etlalchemy_migrate()


if __name__ == '__main__':
    source_db_name = raw_input('Origin Database name: ')
    destination_db_name = raw_input('Destination Database name: ')

    migrate_db_mssql_to_postgresql(source_db_name, destination_db_name)
"""
