"""
Migrate data from Microsoft SQL Server onto PostgreSQL.

(This module should be used in Python 2.)
"""

import os
import urllib.parse

import etlalchemy


def migrate_db_mssql_to_postgresql(origin_db, destination_db):
    """
    Migrate data from Microsoft SQL Server onto PostgreSQL.

    :param origin_db: name of source database
    :type origin_db: str
    :param destination_db: name of database, to which the source database migrate
    :type destination_db: str
    """

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
    pgsql_db.migrate()


if __name__ == '__main__':
    source_db_name = raw_input('Origin database name: ')
    destination_db_name = raw_input('Destination database name: ')

    migrate_db_mssql_to_postgresql(source_db_name, destination_db_name)
