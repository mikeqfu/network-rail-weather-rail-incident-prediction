import getpass

import sqlalchemy
import sqlalchemy.engine.url
import sqlalchemy_utils

from mssql.utils import read_table_by_name


#
def create_postgres_engine_url(db_name='postgres'):
    """
    :param db_name: 
    :return:
    """
    database_info = {'drivername': 'postgresql+psycopg2',
                     'username': input('PostgreSQL username: '),
                     'password': getpass.getpass('PostgreSQL password: '),
                     'host': input('Host name: '),
                     'port': 5432,
                     'database': db_name}
    return sqlalchemy.engine.url.URL(**database_info)


# Copy a single table
def dump_data_to_postgresql(source_data, destination_db_name, destination_table_name, update=True, verbose=False):
    """
    :param source_data:
    :param destination_db_name:
    :param destination_table_name:
    :param update:
    :param verbose:
    :return:
    """
    conn_str = create_postgres_engine_url(destination_db_name)
    # Check if the database already exists
    if not sqlalchemy_utils.database_exists(conn_str):
        sqlalchemy_utils.create_database(conn_str)

    conn_engine = sqlalchemy.create_engine(conn_str, isolation_level='AUTOCOMMIT')
    if not conn_engine.dialect.has_table(conn_engine, destination_table_name) or update:
        try:
            # Dump data to database
            print_word = "Updating" if conn_engine.dialect.has_table(conn_engine, destination_table_name) else "Dumping"
            print("{} \"{}\" ... ".format(print_word, destination_table_name), end="") if verbose else None
            source_data.to_sql(
                destination_table_name, con=conn_engine, if_exists='fail' if not update else 'replace', index=False)
            print("Successfully.") if verbose else None
            del source_data
        except Exception as e:
            print("Failed. {}".format(e))
    else:
        print("\"{}\" already exists in \"{}\".".format(destination_table_name, destination_db_name))


# Copy all tables from MSSQL to PostgreSQL
def copy_mssql_to_postgresql(source_db_name, destination_db_name, update=True, verbose=True):
    """
    :param source_db_name:
    :param destination_db_name:
    :param update:
    :param verbose:
    :return:
    """
    conn_str = create_postgres_engine_url(destination_db_name)
    # Check if the database already exists
    if not sqlalchemy_utils.database_exists(conn_str):
        sqlalchemy_utils.create_database(conn_str)

    conn_engine = sqlalchemy.create_engine(conn_str, isolation_level='AUTOCOMMIT')

    for table_name in conn_engine.table_names(schema='dbo'):
        try:
            source_data = read_table_by_name(source_db_name, table_name, schema_name='dbo')
        except Exception as e:
            print("Failed to identify the source database \"{}\". {}".format(source_db_name, e))
            break

        if not conn_engine.dialect.has_table(conn_engine, table_name) or update:
            try:
                # Dump data to database
                print_word = "Updating" if conn_engine.dialect.has_table(conn_engine, table_name) else "Dumping"
                print("{} \"{}\" ... ".format(print_word, table_name), end="") if verbose else None
                source_data.to_sql(table_name, conn_engine, if_exists='fail' if not update else 'replace', index=False)
                print("Successfully.") if verbose else None
                del source_data
            except Exception as e:
                print("Failed to dump \"{}\". {}".format(table_name, e))
                break
        else:
            print("\"{}\" already exists in \"{}\".".format(table_name, destination_db_name))


# copy_mssql_to_postgresql(source_db_name='NR_VEG', source_table_type='TABLE', destination_db_name='NR_VEG')
# copy_mssql_to_postgresql(source_db_name='NR_METEX', source_table_type='TABLE', destination_db_name='NR_METEX')
# copy_mssql_to_postgresql(source_db_name='NR_METEX', source_table_type='VIEW', destination_db_name='NR_METEX')
