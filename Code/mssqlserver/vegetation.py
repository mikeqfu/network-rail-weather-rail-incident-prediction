""" Read and clean data of NR_Vegetation database """

import os
import re

import pandas as pd
from pyhelpers.geom import osgb36_to_wgs84
from pyhelpers.misc import confirmed
from pyhelpers.settings import pd_preferences
from pyhelpers.store import load_pickle, save, save_pickle
from pyhelpers.text import find_matched_str
from pyrcs.utils import nr_mileage_num_to_str, str_to_num_mileage

from mssqlserver.tools import establish_mssql_connection, get_table_primary_keys
from utils import cdd_vegetation, update_nr_route_names

pd_preferences()

# ====================================================================================================================
""" Change directories """


# Change directory to "Data\\Vegetation\\Database"
def cdd_veg_db(*sub_dir):
    path = cdd_vegetation("Database")
    os.makedirs(path, exist_ok=True)
    for x in sub_dir:
        path = os.path.join(path, x)
    return path


# Change directory to "Data\\Vegetation\\Database\\Tables"
def cdd_veg_db_tables(*sub_dir):
    path = cdd_veg_db("Tables")
    os.makedirs(path, exist_ok=True)
    for directory in sub_dir:
        path = os.path.join(path, directory)
    return path


# Change directory to "Data\\Vegetation\\Database\\Views"
def cdd_veg_db_views(*sub_dir):
    path = cdd_veg_db("Views")
    os.makedirs(path, exist_ok=True)
    for directory in sub_dir:
        path = os.path.join(path, directory)
    return path


# ====================================================================================================================
""" Read table data from the database """


# Read tables available in NR_Vegetation database
def read_veg_table(table_name, index_col=None, route_name=None, coerce_float=True, parse_dates=None, params=None,
                   schema='dbo', save_as=None, update=False):
    """
    :param table_name: [str]
    :param index_col: [str; None (default)]
    :param route_name: [str; None (default)]
    :param coerce_float: [bool; None] (default: True)
    :param parse_dates: [list; dict; None (default)]
    :param params: [list; tuple; dict; None (default)]
    :param schema: [str] (default: 'dbo')
    :param save_as: [str; None (default)]
    :param update: [bool] (default: False)
    :return: [pd.DataFrame]

    Testing e.g.
        table_name = 'AdverseWind'
        index_col = None
        route_name = None
        coerce_float = True
        parse_dates = None
        params = None
        schema = 'dbo'
        save_as = None
        update = False

        read_veg_table(table_name, index_col, route_name, coerce_float, parse_dates, params, schema, save_as, update)
    """
    table = schema + '.' + table_name
    # Make a direct connection to the queried database
    conn_veg = establish_mssql_connection(database_name='NR_Vegetation_20141031')
    if route_name is None:
        sql_query = "SELECT * FROM {}".format(table)  # Get all data of a given table
    else:
        sql_query = "SELECT * FROM {} WHERE Route = '{}'".format(table, route_name)  # given a specific Route
    # Create a pd.DataFrame of the queried table
    data = pd.read_sql(sql=sql_query, con=conn_veg, index_col=index_col, coerce_float=coerce_float,
                       parse_dates=parse_dates, params=params)
    # Disconnect the database
    conn_veg.close()
    # Save the DataFrame as a worksheet locally?
    if save_as:
        path_to_file = cdd_veg_db("Tables_original", table_name + save_as)
        if not os.path.isfile(path_to_file) or update:
            save(data, path_to_file, index=False if index_col is None else True)
    # Return the data frame of the queried table
    return data


# Get primary keys of a table in database 'NR_Vegetation'
def get_veg_table_pk(table_name):
    """
    :param table_name: [str]
    :return: [list]

    Testing e.g.
        table_name = 'AdverseWind'
        get_veg_table_pk(table_name)
    """
    pri_key = get_table_primary_keys(database_name='NR_Vegetation_20141031', table_name=table_name)
    return pri_key


# ====================================================================================================================
""" Get table data """


# Get AdverseWind
def get_adverse_wind(update=False, save_original_as=None, verbose=False):
    """
    :param update: [bool] (default: False)
    :param save_original_as: [str; None (default)]
    :param verbose: [bool] (default: False)
    :return: [pd.DataFrame; None]

    Testing e.g.
        update = True
        save_original_as = None
        verbose = True

        get_adverse_wind(update, save_original_as, verbose)
    """
    table_name = 'AdverseWind'
    path_to_pickle = cdd_veg_db_tables(table_name + ".pickle")
    if os.path.isfile(path_to_pickle) and not update:
        adverse_wind = load_pickle(path_to_pickle)
    else:
        try:
            adverse_wind = read_veg_table(table_name, index_col=None, save_as=save_original_as, update=update)
            update_nr_route_names(adverse_wind, route_col_name='Route')  # Update route names
            adverse_wind = adverse_wind.groupby('Route').agg(list).applymap(lambda x: x if len(x) > 1 else x[0])
            save_pickle(adverse_wind, path_to_pickle, verbose=verbose)
        except Exception as e:
            print("Failed to get \"{}\". {}.".format(table_name, e))
            adverse_wind = None
    return adverse_wind


# Get CuttingAngleClass
def get_cutting_angle_class(update=False, save_original_as=None, verbose=False):
    """
    :param update: [bool] (default: False)
    :param save_original_as: [str; None (default)]
    :param verbose: [bool] (default: False)
    :return: [pd.DataFrame; None]

    Testing e.g.
        update = True
        save_original_as = None
        verbose = True

        get_cutting_angle_class(update, save_original_as, verbose)
    """
    table_name = 'CuttingAngleClass'
    path_to_pickle = cdd_veg_db_tables(table_name + ".pickle")
    if os.path.isfile(path_to_pickle) and not update:
        cutting_angle = load_pickle(path_to_pickle)
    else:
        try:
            cutting_angle = read_veg_table(table_name, index_col=get_veg_table_pk(table_name), save_as=save_original_as,
                                           update=update)
            save_pickle(cutting_angle, path_to_pickle, verbose=verbose)
        except Exception as e:
            print("Failed to get \"{}\". {}.".format(table_name, e))
            cutting_angle = None
    return cutting_angle


# Get CuttingDepthClass
def get_cutting_depth_class(update=False, save_original_as=None, verbose=False):
    """
    :param update: [bool] (default: False)
    :param save_original_as: [str; None (default)]
    :param verbose: [bool] (default: False)
    :return: [pd.DataFrame; None]

    Testing e.g.
        update = True
        save_original_as = None
        verbose = True

        get_cutting_depth_class(update, save_original_as, verbose)
    """
    table_name = 'CuttingDepthClass'
    path_to_pickle = cdd_veg_db_tables(table_name + ".pickle")
    if os.path.isfile(path_to_pickle) and not update:
        cutting_depth = load_pickle(path_to_pickle)
    else:
        try:
            cutting_depth = read_veg_table(table_name, index_col=get_veg_table_pk(table_name), save_as=save_original_as,
                                           update=update)
            save_pickle(cutting_depth, path_to_pickle, verbose=verbose)
        except Exception as e:
            print("Failed to get \"{}\". {}.".format(table_name, e))
            cutting_depth = None
    return cutting_depth


# Get DUList
def get_du_list(index=True, update=False, save_original_as=None, verbose=False):
    """
    :param index: [bool] (default: True)
    :param update: [bool] (default: False)
    :param save_original_as: [str; None (default)]
    :param verbose: [bool] (default: False)
    :return: [pd.DataFrame; None]

    Testing e.g.
        index = True
        update = True
        save_original_as = None
        verbose = True

        get_du_list(index, update, save_original_as, verbose)
        get_du_list(False, update, save_original_as, verbose)
    """
    table_name = 'DUList'
    path_to_pickle = cdd_veg_db_tables(table_name + ("-indexed" if index else "") + ".pickle")
    if os.path.isfile(path_to_pickle) and not update:
        du_list = load_pickle(path_to_pickle)
    else:
        try:
            du_list = read_veg_table(table_name, index_col=get_veg_table_pk(table_name) if index else None,
                                     save_as=save_original_as, update=update)
            save_pickle(du_list, path_to_pickle, verbose=verbose)
        except Exception as e:
            print("Failed to get \"{}\". {}.".format(table_name, e))
            du_list = None
    return du_list


# Get PathRoute
def get_path_route(update=False, save_original_as=None, verbose=False):
    """
    :param update: [bool] (default: False)
    :param save_original_as: [str; None (default)]
    :param verbose: [bool] (default: False)
    :return: [pd.DataFrame; None]

    Testing e.g.
        update = True
        save_original_as = None
        verbose = True

        get_path_route(update, save_original_as, verbose)
    """
    table_name = 'PathRoute'
    path_to_pickle = cdd_veg_db_tables(table_name + ".pickle")
    if os.path.isfile(path_to_pickle) and not update:
        path_route = load_pickle(path_to_pickle)
    else:
        try:
            path_route = read_veg_table(table_name, index_col=get_veg_table_pk(table_name), save_as=save_original_as,
                                        update=update)
            save_pickle(path_route, path_to_pickle, verbose=verbose)
        except Exception as e:
            print("Failed to get \"{}\". {}.".format(table_name, e))
            path_route = None
    return path_route


# Get Routes
def get_du_route(update=False, save_original_as=None, verbose=False):
    """
    :param update: [bool] (default: False)
    :param save_original_as: [str; None (default)]
    :param verbose: [bool] (default: False)
    :return: [pd.DataFrame; None]

    Testing e.g.
        update = True
        save_original_as = None
        verbose = True

        get_du_route(update, save_original_as, verbose)
    """
    table_name = 'Routes'
    path_to_pickle = cdd_veg_db_tables(table_name + ".pickle")
    if os.path.isfile(path_to_pickle) and not update:
        routes = load_pickle(path_to_pickle)
    else:
        try:
            # (Note that 'Routes' table contains information about Delivery Units)
            routes = read_veg_table(table_name, index_col=get_veg_table_pk(table_name), save_as=save_original_as,
                                    update=update)
            # Replace values in (index) column 'DUName'
            routes.index = routes.index.to_series().replace({'Lanc&Cumbria MDU - HR1': 'Lancashire & Cumbria MDU - HR1',
                                                             'S/wel& Dud MDU - HS7': 'Sandwell & Dudley MDU - HS7'})
            # Replace values in column 'DUNameGIS'
            routes.DUNameGIS.replace({'IMDM  Lanc&Cumbria': 'IMDM Lancashire & Cumbria'}, inplace=True)
            # Update route names
            update_nr_route_names(routes, route_col_name='Route')
            save_pickle(routes, path_to_pickle, verbose=verbose)
        except Exception as e:
            print("Failed to get \"{}\". {}.".format(table_name, e))
            routes = None
    return routes


# Get S8Data
def get_s8data(update=False, save_original_as=None, verbose=False):
    """
    :param update: [bool] (default: False)
    :param save_original_as: [str; None (default)]
    :param verbose: [bool] (default: False)
    :return: [pd.DataFrame; None]

    Testing e.g.
        update = True
        save_original_as = None
        verbose = True

        get_s8data(update, save_original_as, verbose)
    """
    table_name = 'S8Data'
    path_to_pickle = cdd_veg_db_tables(table_name + ".pickle")
    if os.path.isfile(path_to_pickle) and not update:
        s8data = load_pickle(path_to_pickle)
    else:
        try:
            s8data = read_veg_table(table_name, index_col=get_veg_table_pk(table_name), save_as=save_original_as,
                                    update=update)
            update_nr_route_names(s8data, route_col_name='Route')
            save_pickle(s8data, path_to_pickle, verbose=verbose)
        except Exception as e:
            print("Failed to get \"{}\". {}.".format(table_name, e))
            s8data = None
    return s8data


# Get TreeAgeClass
def get_tree_age_class(update=False, save_original_as=None, verbose=False):
    """
    :param update: [bool] (default: False)
    :param save_original_as: [str; None (default)]
    :param verbose: [bool] (default: False)
    :return: [pd.DataFrame; None]

    Testing e.g.
        update = True
        save_original_as = None
        verbose = True

        get_tree_age_class(update, save_original_as, verbose)
    """
    table_name = 'TreeAgeClass'
    path_to_pickle = cdd_veg_db_tables(table_name + ".pickle")
    if os.path.isfile(path_to_pickle) and not update:
        tree_age_class = load_pickle(path_to_pickle)
    else:
        try:
            tree_age_class = read_veg_table(table_name, index_col=get_veg_table_pk(table_name),
                                            save_as=save_original_as, update=update)
            save_pickle(tree_age_class, path_to_pickle, verbose=verbose)
        except Exception as e:
            print("Failed to get \"{}\". {}.".format(table_name, e))
            tree_age_class = None
    return tree_age_class


# Get TreeSizeClass
def get_tree_size_class(update=False, save_original_as=None, verbose=False):
    """
    :param update: [bool] (default: False)
    :param save_original_as: [str; None (default)]
    :param verbose: [bool] (default: False)
    :return: [pd.DataFrame; None]

    Testing e.g.
        update = True
        save_original_as = None
        verbose = True

        get_tree_size_class(update, save_original_as, verbose)
    """
    table_name = 'TreeSizeClass'
    path_to_pickle = cdd_veg_db_tables(table_name + ".pickle")
    if os.path.isfile(path_to_pickle) and not update:
        tree_size_class = load_pickle(path_to_pickle)
    else:
        try:
            tree_size_class = read_veg_table(table_name, index_col=get_veg_table_pk(table_name),
                                             save_as=save_original_as, update=update)
            save_pickle(tree_size_class, path_to_pickle, verbose=verbose)
        except Exception as e:
            print("Failed to get \"{}\". {}.".format(table_name, e))
            tree_size_class = None
    return tree_size_class


# Get TreeType
def get_tree_type(update=False, save_original_as=None, verbose=False):
    """
    :param update: [bool] (default: False)
    :param save_original_as: [str; None (default)]
    :param verbose: [bool] (default: False)
    :return: [pd.DataFrame; None]

    Testing e.g.
        update = True
        save_original_as = None
        verbose = True

        get_tree_type(update, save_original_as, verbose)
    """
    table_name = 'TreeType'
    path_to_pickle = cdd_veg_db_tables(table_name + ".pickle")
    if os.path.isfile(path_to_pickle) and not update:
        tree_type = load_pickle(path_to_pickle)
    else:
        try:
            tree_type = read_veg_table(table_name, index_col=get_veg_table_pk(table_name), save_as=save_original_as,
                                       update=update)
            save_pickle(tree_type, path_to_pickle, verbose=verbose)
        except Exception as e:
            print("Failed to get \"{}\". {}.".format(table_name, e))
            tree_type = None
    return tree_type


# Get FellingType
def get_felling_type(update=False, save_original_as=None, verbose=False):
    """
    :param update: [bool] (default: False)
    :param save_original_as: [str; None (default)]
    :param verbose: [bool] (default: False)
    :return: [pd.DataFrame; None]

    Testing e.g.
        update = True
        save_original_as = None
        verbose = True

        get_felling_type(update, save_original_as, verbose)
    """
    table_name = 'FellingType'
    path_to_pickle = cdd_veg_db_tables(table_name + ".pickle")
    if os.path.isfile(path_to_pickle) and not update:
        felling_type = load_pickle(path_to_pickle)
    else:
        try:
            felling_type = read_veg_table(table_name, index_col=get_veg_table_pk(table_name), save_as=save_original_as,
                                          update=update)
            save_pickle(felling_type, path_to_pickle, verbose=verbose)
        except Exception as e:
            print("Failed to get \"{}\". {}.".format(table_name, e))
            felling_type = None
    return felling_type


# Get AreaWorkType
def get_area_work_type(update=False, save_original_as=None, verbose=False):
    """
    :param update: [bool] (default: False)
    :param save_original_as: [str; None (default)]
    :param verbose: [bool] (default: False)
    :return: [pd.DataFrame; None]

    Testing e.g.
        update = True
        save_original_as = None
        verbose = True

        get_area_work_type(update, save_original_as, verbose)
    """
    table_name = 'AreaWorkType'
    path_to_pickle = cdd_veg_db_tables(table_name + ".pickle")
    if os.path.isfile(path_to_pickle) and not update:
        area_work_type = load_pickle(path_to_pickle)
    else:
        try:
            area_work_type = read_veg_table(table_name, index_col=get_veg_table_pk('AreaWorkType'),
                                            save_as=save_original_as, update=update)
            save_pickle(area_work_type, path_to_pickle, verbose=verbose)
        except Exception as e:
            print("Failed to get \"{}\". {}.".format(table_name, e))
            area_work_type = None
    return area_work_type


# Get ServiceDetail
def get_service_detail(update=False, save_original_as=None, verbose=False):
    """
    :param update: [bool] (default: False)
    :param save_original_as: [str; None (default)]
    :param verbose: [bool] (default: False)
    :return: [pd.DataFrame; None]

    Testing e.g.
        update = True
        save_original_as = None
        verbose = True

        get_service_detail(update, save_original_as, verbose)
    """
    table_name = 'ServiceDetail'
    path_to_pickle = cdd_veg_db_tables(table_name + ".pickle")
    if os.path.isfile(path_to_pickle) and not update:
        service_detail = load_pickle(path_to_pickle)
    else:
        try:
            service_detail = read_veg_table(table_name, index_col=get_veg_table_pk('ServiceDetail'),
                                            save_as=save_original_as, update=update)
            save_pickle(service_detail, path_to_pickle, verbose=verbose)
        except Exception as e:
            print("Failed to get \"{}\". {}.".format(table_name, e))
            service_detail = None
    return service_detail


# Get ServicePath
def get_service_path(update=False, save_original_as=None, verbose=False):
    """
    :param update: [bool] (default: False)
    :param save_original_as: [str; None (default)]
    :param verbose: [bool] (default: False)
    :return: [pd.DataFrame; None]

    Testing e.g.
        update = True
        save_original_as = None
        verbose = True

        get_service_path(update, save_original_as, verbose)
    """
    table_name = 'ServicePath'
    path_to_pickle = cdd_veg_db_tables(table_name + ".pickle")
    if os.path.isfile(path_to_pickle) and not update:
        service_path = load_pickle(path_to_pickle)
    else:
        try:
            service_path = read_veg_table(table_name, index_col=get_veg_table_pk(table_name), save_as=save_original_as,
                                          update=update)
            save_pickle(service_path, path_to_pickle, verbose=verbose)
        except Exception as e:
            print("Failed to get \"{}\". {}.".format(table_name, e))
            service_path = None
    return service_path


# Get Supplier
def get_supplier(update=False, save_original_as=None, verbose=False):
    """
    :param update: [bool] (default: False)
    :param save_original_as: [str; None (default)]
    :param verbose: [bool] (default: False)
    :return: [pd.DataFrame; None]

    Testing e.g.
        update = True
        save_original_as = None
        verbose = True

        get_supplier(update, save_original_as, verbose)
    """
    table_name = 'Supplier'
    path_to_pickle = cdd_veg_db_tables(table_name + ".pickle")
    if os.path.isfile(path_to_pickle) and not update:
        supplier = load_pickle(path_to_pickle)
    else:
        try:
            supplier = read_veg_table(table_name, index_col=get_veg_table_pk(table_name), save_as=save_original_as,
                                      update=update)
            save_pickle(supplier, path_to_pickle, verbose=verbose)
        except Exception as e:
            print("Failed to get \"{}\". {}.".format(table_name, e))
            supplier = None
    return supplier


# Get SupplierCosts
def get_supplier_costs(update=False, save_original_as=None, verbose=False):
    """
    :param update: [bool] (default: False)
    :param save_original_as: [str; None (default)]
    :param verbose: [bool] (default: False)
    :return: [pd.DataFrame; None]

    Testing e.g.
        update = True
        save_original_as = None
        verbose = True

        get_supplier_costs(update, save_original_as, verbose)
    """
    table_name = 'SupplierCosts'
    path_to_pickle = cdd_veg_db_tables(table_name + ".pickle")
    if os.path.isfile(path_to_pickle) and not update:
        supplier_costs = load_pickle(path_to_pickle)
    else:
        try:
            supplier_costs = read_veg_table(table_name, index_col=None, save_as=save_original_as, update=update)
            update_nr_route_names(supplier_costs, route_col_name='Route')
            supplier_costs.set_index(get_veg_table_pk(table_name), inplace=True)
            save_pickle(supplier_costs, path_to_pickle, verbose=verbose)
        except Exception as e:
            print("Failed to get \"{}\". {}.".format(table_name, e))
            supplier_costs = None
    return supplier_costs


# Get SupplierCostsArea
def get_supplier_costs_area(update=False, save_original_as=None, verbose=False):
    """
    :param update: [bool] (default: False)
    :param save_original_as: [str; None (default)]
    :param verbose: [bool] (default: False)
    :return: [pd.DataFrame; None]

    Testing e.g.
        update = True
        save_original_as = None
        verbose = True

        get_supplier_costs_area(update, save_original_as, verbose)
    """
    table_name = 'SupplierCostsArea'
    path_to_pickle = cdd_veg_db_tables(table_name + ".pickle")
    if os.path.isfile(path_to_pickle) and not update:
        costs_area = load_pickle(path_to_pickle)
    else:
        try:
            costs_area = read_veg_table(table_name, index_col=None, save_as=save_original_as, update=update)
            update_nr_route_names(costs_area, route_col_name='Route')
            costs_area.set_index(get_veg_table_pk(table_name), inplace=True)
            save_pickle(costs_area, path_to_pickle, verbose=verbose)
        except Exception as e:
            print("Failed to get \"{}\". {}.".format(table_name, e))
            costs_area = None
    return costs_area


# Get SupplierCostsSimple
def get_supplier_cost_simple(update=False, save_original_as=None, verbose=False):
    """
    :param update: [bool] (default: False)
    :param save_original_as: [str; None (default)]
    :param verbose: [bool] (default: False)
    :return: [pd.DataFrame; None]

    Testing e.g.
        update = True
        save_original_as = None
        verbose = True

        get_supplier_cost_simple(update, save_original_as, verbose)
    """
    table_name = 'SupplierCostsSimple'
    path_to_pickle = cdd_veg_db_tables(table_name + ".pickle")
    if os.path.isfile(path_to_pickle) and not update:
        costs_simple = load_pickle(path_to_pickle)
    else:
        try:
            costs_simple = read_veg_table(table_name, index_col=None, save_as=save_original_as, update=update)
            update_nr_route_names(costs_simple, route_col_name='Route')
            costs_simple.set_index(get_veg_table_pk(table_name), inplace=True)
            save_pickle(costs_simple, path_to_pickle, verbose=verbose)
        except Exception as e:
            print("Failed to get \"{}\". {}.".format(table_name, e))
            costs_simple = None
    return costs_simple


# Get TreeActionFractions
def get_tree_action_fractions(update=False, save_original_as=None, verbose=False):
    """
    :param update: [bool] (default: False)
    :param save_original_as: [str; None (default)]
    :param verbose: [bool] (default: False)
    :return: [pd.DataFrame; None]

    Testing e.g.
        update = True
        save_original_as = None
        verbose = True

        get_tree_action_fractions(update, save_original_as, verbose)
    """
    table_name = 'TreeActionFractions'
    path_to_pickle = cdd_veg_db_tables(table_name + ".pickle")
    if os.path.isfile(path_to_pickle) and not update:
        tree_action_fractions = load_pickle(path_to_pickle)
    else:
        try:
            tree_action_fractions = read_veg_table(table_name, index_col=get_veg_table_pk(table_name),
                                                   save_as=save_original_as, update=update)
            save_pickle(tree_action_fractions, path_to_pickle, verbose=verbose)
        except Exception as e:
            print("Failed to get \"{}\". {}.".format(table_name, e))
            tree_action_fractions = None
    return tree_action_fractions


# Get VegSurvTypeClass
def get_veg_surv_type_class(update=False, save_original_as=None, verbose=False):
    """
    :param update: [bool] (default: False)
    :param save_original_as: [str; None (default)]
    :param verbose: [bool] (default: False)
    :return: [pd.DataFrame; None]

    Testing e.g.
        update = True
        save_original_as = None
        verbose = True

        get_veg_surv_type_class(update, save_original_as, verbose)
    """
    table_name = 'VegSurvTypeClass'
    path_to_pickle = cdd_veg_db_tables(table_name + ".pickle")
    if os.path.isfile(path_to_pickle) and not update:
        veg_surv_type_class = load_pickle(path_to_pickle)
    else:
        try:
            veg_surv_type_class = read_veg_table(table_name, index_col=get_veg_table_pk(table_name),
                                                 save_as=save_original_as, update=update)
            save_pickle(veg_surv_type_class, path_to_pickle, verbose=verbose)
        except Exception as e:
            print("Failed to get \"{}\". {}.".format(table_name, e))
            veg_surv_type_class = None
    return veg_surv_type_class


# Get WBFactors
def get_wb_factors(update=False, save_original_as=None, verbose=False):
    """
    :param update: [bool] (default: False)
    :param save_original_as: [str; None (default)]
    :param verbose: [bool] (default: False)
    :return: [pd.DataFrame; None]

    Testing e.g.
        update = True
        save_original_as = None
        verbose = True

        get_wb_factors(update, save_original_as, verbose)
    """
    table_name = 'WBFactors'
    path_to_pickle = cdd_veg_db_tables(table_name + ".pickle")
    if os.path.isfile(path_to_pickle) and not update:
        wb_factors = load_pickle(path_to_pickle)
    else:
        try:
            wb_factors = read_veg_table(table_name, index_col=get_veg_table_pk(table_name), save_as=save_original_as,
                                        update=update)
            save_pickle(wb_factors, path_to_pickle, verbose=verbose)
        except Exception as e:
            print("Failed to get \"{}\". {}.".format(table_name, e))
            wb_factors = None
    return wb_factors


# Get Weedspray
def get_weed_spray(update=False, save_original_as=None, verbose=False):
    """
    :param update: [bool] (default: False)
    :param save_original_as: [str; None (default)]
    :param verbose: [bool] (default: False)
    :return: [pd.DataFrame; None]

    Testing e.g.
        update = True
        save_original_as = None
        verbose = True

        get_weed_spray(update, save_original_as, verbose)
    """
    table_name = 'Weedspray'
    path_to_pickle = cdd_veg_db_tables(table_name + ".pickle")
    if os.path.isfile(path_to_pickle) and not update:
        weed_spray = load_pickle(path_to_pickle)
    else:
        try:
            weed_spray = read_veg_table(table_name, index_col=None, save_as=save_original_as, update=update)
            update_nr_route_names(weed_spray, route_col_name='Route')
            weed_spray.set_index('RouteAlias', inplace=True)
            save_pickle(weed_spray, path_to_pickle, verbose=verbose)
        except Exception as e:
            print("Failed to get \"{}\". {}.".format(table_name, e))
            weed_spray = None
    return weed_spray


# Get WorkHours
def get_work_hours(update=False, save_original_as=None, verbose=False):
    """
    :param update: [bool] (default: False)
    :param save_original_as: [str; None (default)]
    :param verbose: [bool] (default: False)
    :return: [pd.DataFrame; None]

    Testing e.g.
        update = True
        save_original_as = None
        verbose = True

        get_work_hours(update, save_original_as, verbose)
    """
    table_name = 'WorkHours'
    path_to_pickle = cdd_veg_db_tables(table_name + ".pickle")
    if os.path.isfile(path_to_pickle) and not update:
        work_hours = load_pickle(path_to_pickle)
    else:
        try:
            work_hours = read_veg_table(table_name, index_col=get_veg_table_pk(table_name), save_as=save_original_as,
                                        update=update)
            save_pickle(work_hours, path_to_pickle, verbose=verbose)
        except Exception as e:
            print("Failed to get \"{}\". {}.".format(table_name, e))
            work_hours = None
    return work_hours


# Get FurlongData
def get_furlong_data(set_index=False, pseudo_amendment=True, update=False, save_original_as=None, verbose=False):
    """
    :param set_index: [bool] (default: False)
    :param pseudo_amendment: [bool] (default: True)
    :param update: [bool] (default: False)
    :param save_original_as: [str; None (default)]
    :param verbose: [bool] (default: False)
    :return: [pd.DataFrame; None]

    Testing e.g.
        set_index = False
        pseudo_amendment = True
        update = True
        save_original_as = None
        verbose = True

        get_furlong_data(set_index, pseudo_amendment, update, save_original_as, verbose)

    Equipment Class: VL ('VEGETATION - 1/8 MILE SECTION')
    1/8 mile = 220 yards = 1 furlong
    """
    table_name = 'FurlongData'
    path_to_pickle = cdd_veg_db_tables(table_name + ("-indexed" if set_index else "") + ".pickle")

    if os.path.isfile(path_to_pickle) and not update:
        furlong_data = load_pickle(path_to_pickle)
    else:
        try:
            furlong_data = read_veg_table(table_name, index_col=None, coerce_float=False, save_as=save_original_as,
                                          update=update)
            # Re-format mileage data
            furlong_data[['StartMileage', 'EndMileage']] = furlong_data[['StartMileage', 'EndMileage']].applymap(
                nr_mileage_num_to_str)

            # Rename columns
            renamed_cols_dict = {
                'TEF307601': 'MainSpeciesScore',
                'TEF307602': 'TreeSizeScore',
                'TEF307603': 'SurroundingLandScore',
                'TEF307604': 'DistanceFromRailScore',
                'TEF307605': 'OtherVegScore',
                'TEF307606': 'TopographyScore',
                'TEF307607': 'AtmosphereScore',
                'TEF307608': 'TreeDensityScore'}
            furlong_data.rename(columns=renamed_cols_dict, inplace=True)
            # Edit the 'TEF' columns
            furlong_data.OtherVegScore.replace({-1: 0}, inplace=True)
            renamed_cols = list(renamed_cols_dict.values())
            furlong_data[renamed_cols] = furlong_data[renamed_cols].applymap(lambda x: 0 if pd.np.isnan(x) else x + 1)
            # Re-format date of measure
            furlong_data.DateOfMeasure = furlong_data.DateOfMeasure.map(
                lambda x: pd.datetime.strptime(x, '%d/%m/%Y %H:%M'))
            # Edit route data
            update_nr_route_names(furlong_data, route_col_name='Route')

            if set_index:
                furlong_data.set_index(get_veg_table_pk(table_name), inplace=True)

            # Make amendment to "CoverPercent" data for which the total is not 0 or 100?
            if pseudo_amendment:
                # Find columns relating to "CoverPercent..."
                cp_cols = [x for x in furlong_data.columns if re.match('^CoverPercent[A-Z]', x)]

                temp = furlong_data[cp_cols].sum(1)
                if not temp.empty:

                    furlong_data.CoverPercentOther.loc[temp[temp == 0].index] = 100.0  # For all zero 'CoverPercent...'
                    idx = temp[~temp.isin([0.0, 100.0])].index  # For all non-100 'CoverPercent...'

                    nonzero_cols = furlong_data[cp_cols].loc[idx].apply(
                        lambda x: x != 0.0).apply(
                        lambda x: list(pd.Index(cp_cols)[x.values]), axis=1)

                    errors = pd.Series(100.0 - temp[idx])

                    for i in idx:
                        features = nonzero_cols[i].copy()
                        if len(features) == 1:
                            feature = features[0]
                            if feature == 'CoverPercentOther':
                                furlong_data.CoverPercentOther.loc[[i]] = 100.0
                            else:
                                if errors.loc[i] > 0:
                                    furlong_data.CoverPercentOther.loc[[i]] = pd.np.sum([
                                        furlong_data.CoverPercentOther.loc[i], errors.loc[i]])
                                else:  # errors.loc[i] < 0
                                    furlong_data[feature].loc[[i]] = pd.np.sum([
                                        furlong_data[feature].loc[i], errors.loc[i]])
                        else:  # len(nonzero_cols[i]) > 1
                            if 'CoverPercentOther' in features:
                                err = pd.np.sum([furlong_data.CoverPercentOther.loc[i], errors.loc[i]])
                                if err >= 0.0:
                                    furlong_data.CoverPercentOther.loc[[i]] = err
                                else:
                                    features.remove('CoverPercentOther')
                                    furlong_data.CoverPercentOther.loc[[i]] = 0.0
                                    if len(features) == 1:
                                        feature = features[0]
                                        furlong_data[feature].loc[[i]] = pd.np.sum([furlong_data[feature].loc[i], err])
                                    else:
                                        err = pd.np.divide(err, len(features))
                                        furlong_data.loc[i, features] += err
                            else:
                                err = pd.np.divide(errors.loc[i], len(features))
                                furlong_data.loc[i, features] += err

            save_pickle(furlong_data, path_to_pickle, verbose=verbose)

        except Exception as e:
            print("Failed to get \"{}\". {}.".format(table_name, e))
            furlong_data = None

    return furlong_data


# Get FurlongLocation
def get_furlong_location(relevant_columns_only=True, update=False, save_original_as=None, verbose=False):
    """
    :param relevant_columns_only: [bool] (default: True)
    :param update: [bool] (default: False)
    :param save_original_as: [str; None (default)]
    :param verbose: [bool] (default: False)
    :return: [pd.DataFrame; None]

    Testing e.g.
        relevant_columns_only = True
        update = True
        save_original_as = None
        verbose = True

        get_furlong_location(relevant_columns_only, update, save_original_as, verbose)

    Note: One ELR&mileage may have multiple 'FurlongID's.
    """
    table_name = 'FurlongLocation'
    path_to_pickle = cdd_veg_db_tables(table_name + ("-cut" if relevant_columns_only else "") + ".pickle")

    if os.path.isfile(path_to_pickle) and not update:
        furlong_location = load_pickle(path_to_pickle)
    else:
        try:
            # Read data from database
            furlong_location = read_veg_table(table_name, index_col=get_veg_table_pk(table_name),
                                              save_as=save_original_as, update=update)
            # Re-format mileage data
            furlong_location[['StartMileage', 'EndMileage']] = \
                furlong_location[['StartMileage', 'EndMileage']].applymap(nr_mileage_num_to_str)

            # Replace boolean values with binary values
            furlong_location[['Electrified', 'HazardOnly']] = \
                furlong_location[['Electrified', 'HazardOnly']].applymap(int)
            # Replace Route names
            update_nr_route_names(furlong_location, route_col_name='Route')

            # Select useful columns only?
            if relevant_columns_only:
                furlong_location = furlong_location[['Route', 'RouteAlias', 'DU', 'ELR', 'StartMileage', 'EndMileage',
                                                     'Electrified', 'HazardOnly']]

            save_pickle(furlong_location, path_to_pickle, verbose=verbose)

        except Exception as e:
            print("Failed to get \"{}\". {}.".format(table_name, e))
            furlong_location = None

    return furlong_location


# Get HazardTree
def get_hazard_tree(set_index=False, update=False, save_original_as=None, verbose=False):
    """
    :param set_index: [bool] (default: False)
    :param update: [bool] (default: False)
    :param save_original_as: [str; None (default)]
    :param verbose: [bool] (default: False)
    :return: [pd.DataFrame; None]

    Testing e.g.
        set_index = False
        update = True
        save_original_as = None
        verbose = True

        get_hazard_tree(set_index, update, save_original_as, verbose)

    Note that error data exists in 'FurlongID'. They could be cancelled out when the 'hazard_tree' data set is being
    merged with other data sets on the 'FurlongID'. Errors also exist in 'Easting' and 'Northing' columns.
    """
    table_name = 'HazardTree'
    path_to_pickle = cdd_veg_db_tables(table_name + ("-indexed" if set_index else "") + ".pickle")

    if os.path.isfile(path_to_pickle) and not update:
        hazard_tree = load_pickle(path_to_pickle)
    else:
        try:
            hazard_tree = read_veg_table(table_name, index_col=None, save_as=save_original_as, update=update)
            # Re-format mileage data
            hazard_tree.Mileage = hazard_tree.Mileage.apply(nr_mileage_num_to_str)

            # Edit the original data
            hazard_tree.drop(['Treesurvey', 'Treetunnel'], axis=1, inplace=True)
            hazard_tree.dropna(subset=['Northing', 'Easting'], inplace=True)
            hazard_tree.Treespecies.replace({'': 'No data'}, inplace=True)

            # Update route data
            update_nr_route_names(hazard_tree, route_col_name='Route')

            # Integrate information from several features in a DataFrame
            def sum_up_selected_features(data, selected_features, new_feature):
                """
                :param data: [pd.DataFrame] original DataFrame
                :param selected_features: [list] list of columns names
                :param new_feature: [str] new column name
                :return: [pd.DataFrame]
                """
                data.replace({True: 1, False: 0}, inplace=True)
                data[new_feature] = data[selected_features].fillna(0).apply(sum, axis=1)
                data.drop(selected_features, axis=1, inplace=True)

            # Integrate TEF: Failure scores
            failure_scores = ['TEF30770' + str(i) for i in range(1, 6)]
            sum_up_selected_features(hazard_tree, failure_scores, new_feature='Failure_Score')
            # Integrate TEF: Target scores
            target_scores = ['TEF3077%02d' % i for i in range(6, 12)]
            sum_up_selected_features(hazard_tree, target_scores, new_feature='Target_Score')
            # Integrate TEF: Impact scores
            impact_scores = ['TEF3077' + str(i) for i in range(12, 16)]
            sum_up_selected_features(hazard_tree, impact_scores, new_feature='Impact_Score')
            # Rename the rest of TEF
            work_req = ['TEF3077' + str(i) for i in range(17, 27)]
            work_req_desc = [
                'WorkReq_ExpertInspection',
                'WorkReq_LocalisedPruning',
                'WorkReq_GeneralPruning',
                'WorkReq_CrownRemoval',
                'WorkReq_StumpRemoval',
                'WorkReq_TreeRemoval',
                'WorkReq_TargetManagement',
                'WorkReq_FurtherInvestigation',
                'WorkReq_LimbRemoval',
                'WorkReq_InstallSupport']
            hazard_tree.rename(columns=dict(zip(work_req, work_req_desc)), inplace=True)

            # Note the feasibility of the the following operation is not guaranteed:
            hazard_tree[work_req_desc] = hazard_tree[work_req_desc].fillna(value=0)

            # Rearrange DataFrame index
            hazard_tree.index = range(len(hazard_tree))

            # Add two columns of Latitudes and Longitudes corresponding to the Easting and Northing coordinates
            hazard_tree['Longitude'], hazard_tree['Latitude'] = osgb36_to_wgs84(
                hazard_tree.Easting.values, hazard_tree.Northing.values)

            save_pickle(hazard_tree, path_to_pickle, verbose=verbose)

            if set_index:
                hazard_tree.set_index(get_veg_table_pk(table_name), inplace=True)

        except Exception as e:
            print("Failed to get \"{}\". {}.".format(table_name, e))
            hazard_tree = None

    return hazard_tree


# Update the local pickle files for all tables
def update_vegetation_table_pickles(update=True, verbose=True):
    """
    :param update: [bool] (default: True)
    :param verbose: [bool] (default: True)

    Testing e.g.
        update = True
        verbose = True

        update_vegetation_table_pickles(update, verbose)
    """
    if confirmed("To update the pickles of the NR_Vegetation Table data?"):

        _ = get_adverse_wind(update, verbose=verbose)
        _ = get_area_work_type(update, verbose=verbose)
        _ = get_cutting_angle_class(update, verbose=verbose)
        _ = get_cutting_depth_class(update, verbose=verbose)
        _ = get_du_list(index=False, update=update, verbose=verbose)
        _ = get_du_list(index=True, update=update, verbose=verbose)
        _ = get_felling_type(update, verbose=verbose)

        _ = get_furlong_data(set_index=False, pseudo_amendment=True, update=update, verbose=verbose)
        _ = get_furlong_location(relevant_columns_only=False, update=update, verbose=verbose)
        _ = get_furlong_location(relevant_columns_only=True, update=update, verbose=verbose)
        _ = get_hazard_tree(set_index=False, update=update, verbose=verbose)

        _ = get_path_route(update, verbose=verbose)
        _ = get_du_route(update, verbose=verbose)
        _ = get_s8data(update, verbose=verbose)
        _ = get_service_detail(update, verbose=verbose)
        _ = get_service_path(update, verbose=verbose)
        _ = get_supplier(update, verbose=verbose)
        _ = get_supplier_costs(update, verbose=verbose)
        _ = get_supplier_costs_area(update, verbose=verbose)
        _ = get_supplier_cost_simple(update, verbose=verbose)
        _ = get_tree_action_fractions(update, verbose=verbose)
        _ = get_tree_age_class(update, verbose=verbose)
        _ = get_tree_size_class(update, verbose=verbose)
        _ = get_tree_type(update, verbose=verbose)
        _ = get_veg_surv_type_class(update, verbose=verbose)
        _ = get_wb_factors(update, verbose=verbose)
        _ = get_weed_spray(update, verbose=verbose)
        _ = get_work_hours(update, verbose=verbose)

        if verbose:
            print("\nUpdate finished.")


# ====================================================================================================================
""" Get views based on the NR_Vegetation data """


# Create a pickle filename
def make_filename(base_name, route_name, *extra_suffixes, sep="-", save_as=".pickle"):
    """
    :param base_name: [str; None]
    :param route_name: [str; None]
    :param extra_suffixes: [str; None]
    :param sep: [str; None]
    :param save_as: [str] (default: ".pickle")
    :return: [str]
    """
    assert save_as.startswith("."), "\"save_as\" must be extension-like, such as \".pickle\"."
    base_name_ = "" if base_name is None else base_name
    route_name_ = "" if route_name is None \
        else (sep if base_name_ else "") + find_matched_str(route_name, get_du_route().Route).replace(" ", "_")
    if extra_suffixes:
        suffix = ["{}".format(s) for s in extra_suffixes if s]
        suffix = (sep if any(x for x in (base_name_, route_name_)) else "") + sep.join(suffix) \
            if len(suffix) > 1 else sep + suffix[0]
        filename = base_name_ + route_name_ + suffix + save_as
    else:
        filename = base_name_ + route_name_ + save_as
    return filename


# View Vegetation data (75247, 45)
def view_vegetation_coverage_per_furlong(route_name=None, update=False, pickle_it=True, verbose=False):
    """
    :param route_name: [str; None (default)]
    :param update: [bool] (default: False)
    :param pickle_it: [bool] (default: True)
    :param verbose: [bool] (default: False)
    :return: [pd.DataFrame; None]

    Testing e.g.
        route_name = None
        update = True
        pickle_it = True
        verbose = True

        view_vegetation_coverage_per_furlong(route_name, update, pickle_it, verbose)
    """
    path_to_pickle = cdd_veg_db_views(make_filename("vegetation-coverage-per-furlong", route_name))

    if os.path.isfile(path_to_pickle) and not update:
        furlong_vegetation_coverage = load_pickle(path_to_pickle)

    else:
        try:
            furlong_data = get_furlong_data()  # (75247, 40)
            furlong_location = get_furlong_location()  # Set 'FurlongID' to be its index (77017, 8)
            cutting_angle_class = get_cutting_angle_class()  # (5, 1)
            cutting_depth_class = get_cutting_depth_class()  # (5, 1)
            # Merge the data that has been obtained
            furlong_vegetation_coverage = furlong_data. \
                join(furlong_location,  # (75247, 48)
                     on='FurlongID', how='inner', lsuffix='', rsuffix='_FurlongLocation'). \
                join(cutting_angle_class,  # (75247, 49)
                     on='CuttingAngle', how='inner'). \
                join(cutting_depth_class,  # (75247, 50)
                     on='CuttingDepth', how='inner', lsuffix='_CuttingAngle', rsuffix='_CuttingDepth')

            if route_name is not None:
                route_name = find_matched_str(route_name, get_du_route().Route)
                furlong_vegetation_coverage = furlong_vegetation_coverage[
                    furlong_vegetation_coverage.Route == route_name]

            # The total number of trees on both sides
            furlong_vegetation_coverage['TreeNumber'] = \
                furlong_vegetation_coverage[['TreeNumberUp', 'TreeNumberDown']].sum(1)

            # Edit the merged data
            furlong_vegetation_coverage.drop(
                labels=[f for f in furlong_vegetation_coverage.columns if re.match('.*_FurlongLocation$', f)],
                axis=1, inplace=True)  # (75247, 45)

            # Rearrange
            furlong_vegetation_coverage.sort_values(by='StructuredPlantNumber', inplace=True)
            furlong_vegetation_coverage.index = range(len(furlong_vegetation_coverage))

            if pickle_it:
                save_pickle(furlong_vegetation_coverage, path_to_pickle, verbose=verbose)

        except Exception as e:
            print("Failed to fetch the information of vegetation coverage per furlong. {}".format(e))
            furlong_vegetation_coverage = None

    return furlong_vegetation_coverage


# View data of hazardous tress (22180, 66)
def view_hazardous_trees(route_name=None, update=False, pickle_it=True, verbose=False):
    """
    :param route_name: [str; None (default)]
    :param update: [bool] (default: False)
    :param pickle_it: [bool] (default: True)
    :param verbose: [bool] (default: False)
    :return: [pd.DataFrame; None]

    Testing e.g.
        route_name = None
        update = True
        pickle_it = True
        verbose = True

        view_hazardous_trees(route_name, update, pickle_it, verbose)
        view_hazardous_trees('Anglia', update, pickle_it, verbose)
    """
    path_to_pickle = cdd_veg_db_views(make_filename("hazardous-trees", route_name))

    if os.path.isfile(path_to_pickle) and not update:
        hazardous_trees_data = load_pickle(path_to_pickle)

    else:
        try:
            hazard_tree = get_hazard_tree()  # (23950, 60) 1770 with FurlongID being -1
            furlong_location = get_furlong_location()  # (77017, 8)
            tree_age_class = get_tree_age_class()  # (7, 1)
            tree_size_class = get_tree_size_class()  # (5, 1)

            hazardous_trees_data = hazard_tree. \
                join(furlong_location,  # (22180, 68)
                     on='FurlongID', how='inner', lsuffix='', rsuffix='_FurlongLocation'). \
                join(tree_age_class,  # (22180, 69)
                     on='TreeAgeCatID', how='inner'). \
                join(tree_size_class,  # (22180, 70)
                     on='TreeSizeCatID', how='inner', lsuffix='_TreeAgeClass', rsuffix='_TreeSizeClass'). \
                drop(labels=['Route_FurlongLocation', 'DU_FurlongLocation', 'ELR_FurlongLocation'], axis=1)

            if route_name is not None:
                route_name = find_matched_str(route_name, get_du_route().Route)
                hazardous_trees_data = hazardous_trees_data.loc[hazardous_trees_data.Route == route_name]

            # Edit the merged data
            hazardous_trees_data.drop(
                [f for f in hazardous_trees_data.columns if re.match('.*_FurlongLocation$', f)][:3],
                axis=1, inplace=True)  # (22180, 66)
            hazardous_trees_data.index = range(len(hazardous_trees_data))  # Rearrange index

            hazardous_trees_data.rename(columns={'StartMileage': 'Furlong_StartMileage',
                                                 'EndMileage': 'Furlong_EndMileage',
                                                 'Electrified': 'Furlong_Electrified',
                                                 'HazardOnly': 'Furlong_HazardOnly'}, inplace=True)

            if pickle_it:
                save_pickle(hazardous_trees_data, path_to_pickle, verbose=verbose)

        except Exception as e:
            print("Failed to fetch the information of hazardous trees. {}".format(e))
            hazardous_trees_data = None

    return hazardous_trees_data


# View Vegetation data as well as hazardous trees information (75247, 58)
def view_vegetation_condition_per_furlong(route_name=None, update=False, pickle_it=True, verbose=False):
    """
    :param route_name: [str; None (default)]
    :param update: [bool] (default: False)
    :param pickle_it: [bool] (default: True)
    :param verbose: [bool] (default: False)
    :return: [pd.DataFrame; None]

    Testing e.g.
        route_name = None
        update = True
        pickle_it = True
        verbose = True

        view_vegetation_condition_per_furlong(route_name, update, pickle_it, verbose)
    """
    path_to_pickle = cdd_veg_db_views(make_filename("vegetation-condition-per-furlong", route_name))

    if os.path.isfile(path_to_pickle) and not update:
        furlong_vegetation_data = load_pickle(path_to_pickle)

    else:
        try:
            hazardous_trees_data = view_hazardous_trees()  # (22180, 66)

            group_cols = ['ELR', 'DU', 'Route', 'Furlong_StartMileage', 'Furlong_EndMileage']
            furlong_hazard_tree = hazardous_trees_data.groupby(group_cols).aggregate({
                # 'AssetNumber': np.count_nonzero,
                'Haztreeid': pd.np.count_nonzero,
                'TreeheightM': [lambda x: tuple(x), min, max],
                'TreediameterM': [lambda x: tuple(x), min, max],
                'TreeproxrailM': [lambda x: tuple(x), min, max],
                'Treeprox3py': [lambda x: tuple(x), min, max]})  # (11320, 13)

            furlong_hazard_tree.columns = ['_'.join(x).strip() for x in furlong_hazard_tree.columns]
            furlong_hazard_tree.rename(columns={'Haztreeid_count_nonzero': 'TreeNumber'}, inplace=True)
            furlong_hazard_tree.columns = ['Hazard' + x.strip('_<lambda_0>') for x in furlong_hazard_tree.columns]

            #
            furlong_vegetation_coverage = view_vegetation_coverage_per_furlong()  # (75247, 45)

            # Processing ...
            furlong_vegetation_data = furlong_vegetation_coverage.join(
                furlong_hazard_tree, on=['ELR', 'DU', 'Route', 'StartMileage', 'EndMileage'], how='left')
            furlong_vegetation_data.sort_values('StructuredPlantNumber', inplace=True)  # (75247, 58)

            if route_name is not None:
                route_name = find_matched_str(route_name, get_du_route().Route)
                furlong_vegetation_data = hazardous_trees_data.loc[furlong_vegetation_data.Route == route_name]
                furlong_vegetation_data.index = range(len(furlong_vegetation_data))

            if pickle_it:
                save_pickle(furlong_vegetation_data, path_to_pickle, verbose=verbose)

        except Exception as e:
            print("Failed to fetch the information of vegetation condition per furlong. {}".format(e))
            furlong_vegetation_data = None

    return furlong_vegetation_data


# ELR & mileage data of furlong locations
def view_nr_vegetation_furlong_data(update=False, pickle_it=True, verbose=False) -> pd.DataFrame:
    """
    :param update: [bool] (default: False)
    :param pickle_it: [bool] (default: True)
    :param verbose: [bool] (default: False)
    :return: [pd.DataFrame; None]

    Testing e.g.
        update = True
        pickle_it = True
        verbose = True

        view_nr_vegetation_furlong_data(update, pickle_it, verbose)
    """
    path_to_pickle = cdd_veg_db_views("vegetation-furlong-data.pickle")

    if os.path.isfile(path_to_pickle) and not update:
        nr_vegetation_furlong_data = load_pickle(path_to_pickle)

    else:
        try:
            # Get the data of furlong location
            nr_vegetation_furlong_data = view_vegetation_condition_per_furlong()
            nr_vegetation_furlong_data.set_index('FurlongID', inplace=True)
            nr_vegetation_furlong_data.sort_index(inplace=True)

            # Column names of mileage data (as string)
            str_mileage_colnames = ['StartMileage', 'EndMileage']
            # Column names of ELR and mileage data (as string)
            elr_mileage_colnames = ['ELR'] + str_mileage_colnames

            nr_vegetation_furlong_data.drop_duplicates(elr_mileage_colnames, inplace=True)

            # Create two new columns of mileage data (as float)
            num_mileage_colnames = ['StartMileage_num', 'EndMileage_num']
            nr_vegetation_furlong_data[num_mileage_colnames] = nr_vegetation_furlong_data[
                str_mileage_colnames].applymap(str_to_num_mileage)

            # Sort the furlong data by ELR and mileage
            nr_vegetation_furlong_data.sort_values(['ELR'] + num_mileage_colnames, inplace=True)

            if pickle_it:
                save_pickle(nr_vegetation_furlong_data, path_to_pickle, verbose=verbose)

        except Exception as e:
            print("Failed to fetch ELR & mileage data of furlong locations. {}".format(e))
            nr_vegetation_furlong_data = None

    return nr_vegetation_furlong_data


# Update the local pickle files for all essential views
def update_vegetation_view_pickles(route_name=None, update=True, pickle_it=True, verbose=True):
    """
    :param route_name: [str; None (default)]
    :param update: [bool] (default: True)
    :param pickle_it: [bool] (default: True)
    :param verbose: [bool] (default: True)

    Testing e.g.
        route_name = None
        update = True
        pickle_it = True
        verbose = True

        update_vegetation_view_pickles(route_name, update, pickle_it, verbose)
    """
    if confirmed("To update the View pickles of the NR_Vegetation data?"):

        _ = view_hazardous_trees(route_name, update, pickle_it, verbose)
        _ = view_hazardous_trees('Anglia', update, pickle_it, verbose)

        _ = view_vegetation_condition_per_furlong(route_name, update, pickle_it, verbose)

        _ = view_vegetation_coverage_per_furlong(route_name, update, pickle_it, verbose)

        _ = view_nr_vegetation_furlong_data(update, pickle_it, verbose)

        if verbose:
            print("\nUpdate finished.")
