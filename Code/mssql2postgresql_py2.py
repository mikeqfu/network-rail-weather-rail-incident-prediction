# This file is written and should be run in Python 2 environment

import os
import urllib

import etlalchemy


# Migrate from SQL Server onto PostgreSQL
def migrate_mssql_db_to_postgresql(source_db_name, destination_db_name):
    """
    :param source_db_name: [str] Name of source database
    :param destination_db_name: [str] Name of database, to which the source database migrate
    :return: N/A
    """

    # Windows authentication for reading data from the databases
    def windows_authentication():
        """
        The trusted_connection setting indicates whether to use Windows Authentication Mode for login validation or not.
        'Trusted_Connection=yes' specifies the user used to establish this connection. In Microsoft implementations,
        this user account is a Windows user account.
        """
        return 'Trusted_Connection=yes;'

    # Database driver
    def database_driver():
        """
        DRIVER={SQL Server Native Client 11.0}
        """
        return 'DRIVER={SQL Server};'

    # Database server name
    def database_server():
        """
        Alternative methods to get the computer name

        import platform
        platform.node()

        Or,

        import socket
        socket.gethostname()
        """
        server_name = os.environ['COMPUTERNAME']
        if 'EEE' in server_name:
            server_name += '\\SQLEXPRESS'
        return 'SERVER={};'.format(server_name)

    # Database name
    def database_name(db_name):
        return 'DATABASE={};'.format(db_name)

    mssql_str = windows_authentication() + database_driver() + database_server() + database_name(source_db_name)
    mssql_str = 'mssql+pyodbc:///?odbc_connect=%s' % urllib.quote_plus(mssql_str)
    mssql_db = etlalchemy.ETLAlchemySource(mssql_str)

    pgsql_pwd = int(raw_input('Password to connect PostgreSQL: '))
    pgsql_str = 'postgresql+psycopg2://postgres:{}@localhost/{}'.format(pgsql_pwd, destination_db_name)
    pgsql_db = etlalchemy.ETLAlchemyTarget(pgsql_str, drop_database=True)

    pgsql_db.addSource(mssql_db)
    pgsql_db.migrate()


source_db_name = raw_input('Source database name: ')
destination_db_name = raw_input('Destination database name: ')
migrate_mssql_db_to_postgresql(source_db_name, destination_db_name)
