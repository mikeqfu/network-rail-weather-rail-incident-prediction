""" This file is written and should be run in Python 2 """

import os
import urllib.parse

import etlalchemy


def migrate_mssql_db_to_postgresql(source_database_name, destination_database_name):
    """
    Migrate from SQL Server onto PostgreSQL.

    :param source_database_name: name of source database
    :type source_database_name: str
    :param destination_database_name: name of database, to which the source database migrate
    :type destination_database_name: str
    """

    def windows_authentication():
        return 'Trusted_Connection=yes;'

    def database_driver():
        return 'DRIVER={SQL Server};'

    def database_server():
        server_name = os.environ['COMPUTERNAME']
        if 'EEE' in server_name:
            server_name += '\\SQLEXPRESS'
        return 'SERVER={};'.format(server_name)

    # Database name
    def database_name(db_name):
        return 'DATABASE={};'.format(db_name)

    mssql_str = windows_authentication() + database_driver() + database_server() + database_name(source_database_name)
    mssql_str = 'mssql+pyodbc:///?odbc_connect=%s' % urllib.parse.quote_plus(mssql_str)
    mssql_db = etlalchemy.ETLAlchemySource(mssql_str)

    pgsql_pwd = int(raw_input('Password to connect PostgreSQL: '))
    pgsql_str = 'postgresql+psycopg2://postgres:{}@localhost/{}'.format(pgsql_pwd, destination_database_name)
    pgsql_db = etlalchemy.ETLAlchemyTarget(pgsql_str, drop_database=True)

    pgsql_db.addSource(mssql_db)
    pgsql_db.migrate()


source_db_name = raw_input('Source database name: ')
destination_db_name = raw_input('Destination database name: ')
migrate_mssql_db_to_postgresql(source_db_name, destination_db_name)
