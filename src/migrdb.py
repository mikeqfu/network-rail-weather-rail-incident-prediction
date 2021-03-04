"""
Migrate data from Microsoft SQL Server onto PostgreSQL.

"""

import getpass
import os
import urllib.parse

import execnet.multi
import sqlalchemy
import sqlalchemy.engine.url
import sqlalchemy_utils
from pyhelpers.sql import PostgreSQL

from utils import read_table_by_name


def create_postgres_engine_url(database_name='postgres'):
    """
    Create a PostgreSQL engine URL.

    :param database_name: name of a database
    :type database_name: str
    :return: PostgreSQL engine URL
    :rtype: sqlalchemy.engine.url.URL
    """

    postgres_engine_url = sqlalchemy.engine.url.URL(
        drivername='postgresql+psycopg2',
        username=input('PostgreSQL username: '),
        password=getpass.getpass('PostgreSQL password: '),
        host=input('Host name: '),
        port=5432,
        database=database_name)

    return postgres_engine_url


def import_data_to_postgresql(data, db_name, table_name, update=True, verbose=False):
    """
    Copy a single table from MSSQL to PostgreSQL.

    :param data: data frame to be dumped into a PostgreSQL server
    :type data: pandas.DataFrame
    :param db_name: name of a database
    :type db_name: str
    :param table_name: name of a table
    :type table_name: str
    :param update: defaults to ``True``
    :type update: bool
    :param verbose: defaults to ``False``
    :type verbose: bool
    """

    conn_str = create_postgres_engine_url(db_name)
    # Check if the database already exists
    if not sqlalchemy_utils.database_exists(conn_str):
        sqlalchemy_utils.create_database(conn_str)

    conn_engine = sqlalchemy.create_engine(conn_str, isolation_level='AUTOCOMMIT')
    if not conn_engine.dialect.has_table(conn_engine, table_name) or update:
        try:
            # Dump data to database
            if conn_engine.dialect.has_table(conn_engine, table_name):
                print_word = "Updating"
            else:
                print_word = "Importing"

            if verbose:
                print("{} \"{}\" ... ".format(print_word, table_name), end="")

            data.to_sql(table_name, con=conn_engine,
                        if_exists='fail' if not update else 'replace',
                        index=False)

            print("Successfully.") if verbose else ""

            del data

        except Exception as e:
            print("Failed. {}".format(e))
    else:
        if verbose:
            print("\"{}\" already exists in \"{}\".".format(table_name, db_name))


def copy_mssql_to_postgresql(origin_db_name, destination_db_name, update=True, verbose=True):
    """
    Copy all tables of a database from MSSQL to PostgreSQL.

    :param origin_db_name: name of the source database
    :type origin_db_name: str
    :param destination_db_name: name of the destination database
    :type destination_db_name: str
    :param update: defaults to ``True``
    :type update: bool
    :param verbose: defaults to ``False``
    :type verbose: bool

    **Examples**::

        >>> from migrdb import copy_mssql_to_postgresql

        >>> copy_mssql_to_postgresql(origin_db_name='NR_VEG', db_name='NR_VEG')

        >>> copy_mssql_to_postgresql(origin_db_name='NR_METEX', db_name='NR_METEX')
    """

    conn_str = create_postgres_engine_url(destination_db_name)
    # Check if the database already exists
    if not sqlalchemy_utils.database_exists(conn_str):
        sqlalchemy_utils.create_database(conn_str)

    conn_engine = sqlalchemy.create_engine(conn_str, isolation_level='AUTOCOMMIT')

    for table_name in conn_engine.table_names(schema='dbo'):
        try:
            source_data = read_table_by_name(origin_db_name, table_name, schema_name='dbo')
        except Exception as e:
            print("Failed to identify the source database \"{}\". {}".format(origin_db_name, e))
            break

        if not conn_engine.dialect.has_table(conn_engine, table_name) or update:
            try:
                # Dump data to database
                if conn_engine.dialect.has_table(conn_engine, table_name):
                    print_word = "Updating"
                else:
                    print_word = "Importing"

                if verbose:
                    print("{} \"{}\" ... ".format(print_word, table_name), end="")

                source_data.to_sql(table_name, conn_engine,
                                   if_exists='fail' if not update else 'replace',
                                   index=False)

                if verbose:
                    print("Successfully.")

                del source_data

            except Exception as e:
                print("Failed to dump \"{}\". {}".format(table_name, e))
                break

        else:
            if verbose:
                print("\"{}\" already exists in \"{}\".".format(table_name, destination_db_name))


def py2_etlalchemy_migrate(source_db_name, destination_db_name, postgres_pwd, python2=None):
    """

    :param source_db_name:
    :param destination_db_name:
    :param postgres_pwd:
    :param python2:
    :return:

    **Test**::

        >>> from migrdb import py2_etlalchemy_migrate

        >>> source_database = 'NR_Vegetation_20141031'  # source_db_name
        >>> destination_database = 'TestVeg'  # destination_db_name
        >>> destination_pwd = 123  # postgres_pwd

        >>> py2_etlalchemy_migrate(source_database, destination_database, destination_pwd)
    """

    if python2 is None:
        python2 = "C:\\Python27\\python"

    server_name = os.environ['COMPUTERNAME']
    mssql_str = 'Trusted_Connection=yes;DRIVER={SQL Server};SERVER=%s;DATABASE=%s;' % (
        server_name, source_db_name)
    mssql_str = 'mssql+pyodbc:///?odbc_connect=%s' % urllib.parse.quote_plus(mssql_str)

    pgsql_str = 'postgresql+psycopg2://postgres:%s@localhost/%s' % (postgres_pwd, destination_db_name)

    postgres = PostgreSQL(host='localhost', port=5432, username='postgres', password=postgres_pwd,
                          database_name='postgres')

    if not postgres.database_exists(destination_db_name):
        postgres.create_database(database_name=destination_db_name)

    gw = execnet.multi.makegateway("popen//python='%s'" % python2)
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
(In Python 2.)

import os
import urllib.parse

import etlalchemy


def migrate_db_mssql_to_postgresql(origin_db, destination_db):
    '''
    Migrate data from Microsoft SQL Server onto PostgreSQL.

    :param origin_db: name of source database
    :type origin_db: str
    :param destination_db: name of database, to which the source database py2_etlalchemy_migrate
    :type destination_db: str
    '''

    def windows_authentication():
        return 'Trusted_Connection=yes;'

    def db_driver():
        return 'DRIVER={SQL Server};'

    def db_server():
        server_name = os.environ['COMPUTERNAME']
        if 'EEE' in server_name:
            server_name += '\\SQLEXPRESS'
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
    source_db_name = raw_input('Origin database name: ')
    destination_db_name = raw_input('Destination database name: ')

    migrate_db_mssql_to_postgresql(source_db_name, destination_db_name)

"""