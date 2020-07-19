import getpass

import sqlalchemy
import sqlalchemy.engine.url
import sqlalchemy_utils

from mssqlserver.tools import read_table_by_name


def create_postgres_engine_url(database_name='postgres'):
    """
    Create a PostgreSQL engine URL

    :param database_name: name of a database
    :type database_name: str
    :return: PostgreSQL engine URL
    :rtype: sqlalchemy.engine.url.URL
    """

    postgres_engine_url = sqlalchemy.engine.url.URL(drivername='postgresql+psycopg2',
                                                    username=input('PostgreSQL username: '),
                                                    password=getpass.getpass('PostgreSQL password: '),
                                                    host=input('Host name: '),
                                                    port=5432,
                                                    database=database_name)
    return postgres_engine_url


def dump_data_to_postgresql(source_data, destination_database_name, destination_table_name, update=True, verbose=False):
    """
    Copy a single table

    :param source_data: data frame to be dumped into a PostgreSQL server
    :type source_data: pandas.DataFrame
    :param destination_database_name: name of a database
    :type destination_database_name: str
    :param destination_table_name: name of a table
    :type destination_table_name: str
    :param update: defaults to ``True``
    :type update: bool
    :param verbose: defaults to ``False``
    :type verbose: bool
    """

    conn_str = create_postgres_engine_url(destination_database_name)
    # Check if the database already exists
    if not sqlalchemy_utils.database_exists(conn_str):
        sqlalchemy_utils.create_database(conn_str)

    conn_engine = sqlalchemy.create_engine(conn_str, isolation_level='AUTOCOMMIT')
    if not conn_engine.dialect.has_table(conn_engine, destination_table_name) or update:
        try:
            # Dump data to database
            print_word = "Updating" if conn_engine.dialect.has_table(conn_engine, destination_table_name) else "Dumping"
            print("{} \"{}\" ... ".format(print_word, destination_table_name), end="") if verbose else None
            source_data.to_sql(destination_table_name, con=conn_engine, if_exists='fail' if not update else 'replace',
                               index=False)
            print("Successfully.") if verbose else ""
            del source_data
        except Exception as e:
            print("Failed. {}".format(e))
    else:
        if verbose:
            print("\"{}\" already exists in \"{}\".".format(destination_table_name, destination_database_name))


def copy_mssql_to_postgresql(source_database_name, destination_database_name, update=True, verbose=True):
    """
    Copy all tables from MSSQL to PostgreSQL

    :param source_database_name: name of the source database
    :type source_database_name: str
    :param destination_database_name: name of the destination database
    :type destination_database_name: str
    :param update: defaults to ``True``
    :type update: bool
    :param verbose: defaults to ``False``
    :type verbose: bool
    """

    conn_str = create_postgres_engine_url(destination_database_name)
    # Check if the database already exists
    if not sqlalchemy_utils.database_exists(conn_str):
        sqlalchemy_utils.create_database(conn_str)

    conn_engine = sqlalchemy.create_engine(conn_str, isolation_level='AUTOCOMMIT')

    for table_name in conn_engine.table_names(schema='dbo'):
        try:
            source_data = read_table_by_name(source_database_name, table_name, schema_name='dbo')
        except Exception as e:
            print("Failed to identify the source database \"{}\". {}".format(source_database_name, e))
            break

        if not conn_engine.dialect.has_table(conn_engine, table_name) or update:
            try:
                # Dump data to database
                print_word = "Updating" if conn_engine.dialect.has_table(conn_engine, table_name) else "Dumping"
                print("{} \"{}\" ... ".format(print_word, table_name), end="") if verbose else ""
                source_data.to_sql(table_name, conn_engine, if_exists='fail' if not update else 'replace', index=False)
                print("Successfully.") if verbose else ""
                del source_data
            except Exception as e:
                print("Failed to dump \"{}\". {}".format(table_name, e))
                break
        else:
            print("\"{}\" already exists in \"{}\".".format(table_name, destination_database_name)) if verbose else ""


# copy_mssql_to_postgresql(source_database_name='NR_VEG', destination_database_name='NR_VEG')
# copy_mssql_to_postgresql(source_database_name='NR_METEX', destination_database_name='NR_METEX')
# copy_mssql_to_postgresql(source_database_name='NR_METEX', destination_database_name='NR_METEX')
