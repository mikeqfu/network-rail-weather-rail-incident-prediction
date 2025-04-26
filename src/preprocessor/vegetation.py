"""
Read and cleanse data of vegetation database.
"""

import numpy as np
from pyhelpers.ops import confirmed
from pyhelpers.text import find_similar_str
from pyrcs.converter import mileage_str_to_num

from src.preprocessor._base import BaseVegetation


# noinspection PyUnresolvedReferences
class Vegetation(BaseVegetation):
    """
    A class for handling the database for vegetation data.
    """

    def __init__(self, db_instance=None):
        """
        :param db_instance: An instance of a class connecting the project's database;
            defaults to ``None``.
        :type db_instance: pyhelpers.dbms.PostgreSQL | None

        :ivar src.utils.WxRailIncidentsPred | None db_instance:
            An instance of a class connecting the project's database,
            e.g. :class:`~src.utils.WxRailIncidentsPred`.

        :ivar pandas.DataFrame | None furlong_data:
        :ivar pandas.DataFrame | None furlong_location:
        :ivar pandas.DataFrame | None hazard_tree:
        :ivar pandas.DataFrame | None routes:
        :ivar pandas.DataFrame | None cutting_angle_class:
        :ivar pandas.DataFrame | None cutting_depth_class:
        :ivar pandas.DataFrame | None tree_age_class:
        :ivar pandas.DataFrame | None tree_size_class:
        :ivar pandas.DataFrame | None vegetation_coverage:
        :ivar pandas.DataFrame | None hazardous_trees:
        :ivar pandas.DataFrame | None vegetation_condition:
        :ivar pandas.DataFrame | None vegetation_condition2:

        **Examples**::

            >>> from src.preprocessor.vegetation import Vegetation
            >>> veg = Vegetation()
            >>> veg.DATA_NAME
            'Vegetation'
            >>> veg.SCHEMA_NAME
            'NR_Vegetation_20141031_prep'
        """

        super().__init__()

        self.db_instance = db_instance

        self.furlong_data = None
        self.furlong_location = None
        self.hazard_tree = None
        self.routes = None
        self.cutting_angle_class = None
        self.cutting_depth_class = None
        self.tree_age_class = None
        self.tree_size_class = None

        self.vegetation_coverage = None
        self.hazardous_trees = None
        self.vegetation_condition = None
        self.vegetation_condition2 = None

    def read_adverse_wind(self, update=False, verbose=False, **kwargs):
        # noinspection PyUnresolvedReferences
        """
        Get data of the table "AdverseWind".

        :param update: Whether to reprocess the original/raw data and update the preprocessed data;
            defaults to ``False``.
        :type update: bool
        :param verbose: Whether to print relevant information in console; defaults to ``False``.
        :type verbose: bool | int
        :param kwargs: [Optional] parameters of the method
            :meth:`src.preprocessor._base.BaseVegetation._read_data`.
        :return: Data of the table "AdverseWind".
        :rtype: pandas.DataFrame

        **Examples**::

            >>> from src.preprocessor.vegetation import Vegetation
            >>> veg = Vegetation()
            >>> veg.read_adverse_wind()
            >>> veg.adverse_wind
        """

        kwargs.update(dict(table_name='AdverseWind', update=update, verbose=verbose))
        self._read_data(**kwargs)

    def read_cutting_angle_class(self, update=False, verbose=False, **kwargs):
        """
        Get data of the table "CuttingAngleClass".

        :param update: Whether to reprocess the original/raw data and update the preprocessed data;
            defaults to ``False``.
        :type update: bool
        :param verbose: Whether to print relevant information in console; defaults to ``False``.
        :type verbose: bool | int
        :param kwargs: [Optional] parameters of the method
            :meth:`src.preprocessor._base.BaseVegetation._read_data`.
        :return: Data of the table "CuttingAngleClass".
        :rtype: pandas.DataFrame | None

        **Examples**::

            >>> from src.preprocessor.vegetation import Vegetation
            >>> veg = Vegetation()
            >>> veg.read_cutting_angle_class()
            >>> veg.cutting_angle_class
        """

        kwargs.update(dict(table_name='CuttingAngleClass', update=update, verbose=verbose))
        self._read_data(**kwargs)

    def read_cutting_depth_class(self, update=False, verbose=False, **kwargs):
        """
        Get data of the table "CuttingDepthClass".

        :param update: Whether to reprocess the original/raw data and update the preprocessed data;
            defaults to ``False``.
        :type update: bool
        :param verbose: Whether to print relevant information in console; defaults to ``False``.
        :type verbose: bool | int
        :param kwargs: [Optional] parameters of the method
            :meth:`src.preprocessor._base.BaseVegetation._read_data`.
        :return: Data of the table "CuttingDepthClass".
        :rtype: pandas.DataFrame | None

        **Examples**::

            >>> from src.preprocessor.vegetation import Vegetation
            >>> veg = Vegetation()
            >>> veg.read_cutting_depth_class()
            >>> veg.cutting_depth_class
        """

        kwargs.update(dict(table_name='CuttingDepthClass', update=update, verbose=verbose))
        self._read_data(**kwargs)

    def read_du_list(self, update=False, verbose=False, **kwargs):
        # noinspection PyUnresolvedReferences
        """
        Get data of the table "DUList".

        :param update: Whether to reprocess the original/raw data and update the preprocessed data;
            defaults to ``False``.
        :type update: bool
        :param verbose: Whether to print relevant information in console; defaults to ``False``.
        :type verbose: bool | int
        :param kwargs: [Optional] parameters of the method
            :meth:`src.preprocessor._base.BaseVegetation._read_data`.
        :return: Data of the table "DUList".
        :rtype: pandas.DataFrame | None

        **Examples**::

            >>> from src.preprocessor.vegetation import Vegetation
            >>> veg = Vegetation()
            >>> veg.read_du_list()
            >>> veg.du_list
        """

        kwargs.update(
            dict(table_name='DUList', ivar_name='du_list', update=update, verbose=verbose))
        self._read_data(**kwargs)

    def read_path_route(self, update=False, verbose=False, **kwargs):
        # noinspection PyUnresolvedReferences
        """
        Get data of the table "PathRoute".

        :param update: Whether to reprocess the original/raw data and update the preprocessed data;
            defaults to ``False``.
        :type update: bool
        :param verbose: Whether to print relevant information in console; defaults to ``False``.
        :type verbose: bool | int
        :param kwargs: [Optional] parameters of the method
            :meth:`src.preprocessor._base.BaseVegetation._read_data`.
        :return: Data of the table "PathRoute".
        :rtype: pandas.DataFrame | None

        **Examples**::

            >>> from src.preprocessor.vegetation import Vegetation
            >>> veg = Vegetation()
            >>> veg.read_path_route()
            >>> veg.path_route
        """

        kwargs.update(
            dict(table_name='PathRoute', true_values=['t', 'true'], false_values=['f', 'false'],
                 keep_default_na=False, update=update, verbose=verbose))
        self._read_data(**kwargs)

    def read_routes(self, update=False, verbose=False, **kwargs):
        """
        Get data of the table "Routes".

        :param update: Whether to reprocess the original/raw data and update the preprocessed data;
            defaults to ``False``.
        :type update: bool
        :param verbose: Whether to print relevant information in console; defaults to ``False``.
        :type verbose: bool | int
        :param kwargs: [Optional] parameters of the method
            :meth:`src.preprocessor._base.BaseVegetation._read_data`.
        :return: Data of the table "Routes".
        :rtype: pandas.DataFrame | None

        **Examples**::

            >>> from src.preprocessor.vegetation import Vegetation
            >>> veg = Vegetation()
            >>> veg.read_routes()
            >>> veg.routes
        """

        kwargs.update(dict(table_name='Routes', update=update, verbose=verbose))
        self._read_data(**kwargs)

    def read_s8data(self, update=False, verbose=False, **kwargs):
        """
        Get data of the table "S8Data".

        :param update: Whether to reprocess the original/raw data and update the preprocessed data;
            defaults to ``False``.
        :type update: bool
        :param verbose: Whether to print relevant information in console; defaults to ``False``.
        :type verbose: bool | int
        :param kwargs: [Optional] parameters of the method
            :meth:`src.preprocessor._base.BaseVegetation._read_data`.
        :return: Data of the table "S8Data".
        :rtype: pandas.DataFrame | None

        **Examples**::

            >>> from src.preprocessor.vegetation import Vegetation
            >>> veg = Vegetation()
            >>> veg.read_s8data()
            >>> veg.s8data
        """

        table_name = 'S8Data'
        kwargs.update(
            dict(table_name=table_name, ivar_name=table_name.lower(), update=update,
                 verbose=verbose))
        self._read_data(**kwargs)

    def read_tree_age_class(self, update=False, verbose=False, **kwargs):
        """
        Get data of the table "TreeAgeClass".

        :param update: Whether to reprocess the original/raw data and update the preprocessed data;
            defaults to ``False``.
        :type update: bool
        :param verbose: Whether to print relevant information in console; defaults to ``False``.
        :type verbose: bool | int
        :param kwargs: [Optional] parameters of the method
            :meth:`src.preprocessor._base.BaseVegetation._read_data`.
        :return: Data of the table "TreeAgeClass".
        :rtype: pandas.DataFrame | None

        **Examples**::

            >>> from src.preprocessor.vegetation import Vegetation
            >>> veg = Vegetation()
            >>> veg.read_tree_age_class()
            >>> veg.tree_age_class
        """

        kwargs.update(dict(table_name='TreeAgeClass', update=update, verbose=verbose))
        self._read_data(**kwargs)

    def read_tree_size_class(self, update=False, verbose=False, **kwargs):
        """
        Get data of the table "TreeSizeClass".

        :param update: Whether to reprocess the original/raw data and update the preprocessed data;
            defaults to ``False``.
        :type update: bool
        :param verbose: Whether to print relevant information in console; defaults to ``False``.
        :type verbose: bool | int
        :param kwargs: [Optional] parameters of the method
            :meth:`src.preprocessor._base.BaseVegetation._read_data`.
        :return: Data of the table "TreeSizeClass".
        :rtype: pandas.DataFrame | None

        **Examples**::

            >>> from src.preprocessor.vegetation import Vegetation
            >>> veg = Vegetation()
            >>> veg.read_tree_size_class()
            >>> veg.tree_size_class
        """

        kwargs.update(dict(table_name='TreeSizeClass', update=update, verbose=verbose))
        self._read_data(**kwargs)

    def read_tree_type(self, update=False, verbose=False, **kwargs):
        """
        Get data of the table "TreeType".

        :param update: Whether to reprocess the original/raw data and update the preprocessed data;
            defaults to ``False``.
        :type update: bool
        :param verbose: Whether to print relevant information in console; defaults to ``False``.
        :type verbose: bool | int
        :param kwargs: [Optional] parameters of the method
            :meth:`src.preprocessor._base.BaseVegetation._read_data`.
        :return: Data of the table "TreeType".
        :rtype: pandas.DataFrame | None

        **Examples**::

            >>> from src.preprocessor.vegetation import Vegetation
            >>> veg = Vegetation()
            >>> veg.read_tree_type()
            >>> veg.tree_type
        """

        kwargs.update(dict(table_name='TreeType', update=update, verbose=verbose))
        self._read_data(**kwargs)

    def read_felling_type(self, update=False, verbose=False, **kwargs):
        """
        Get data of the table "FellingType".

        :param update: Whether to reprocess the original/raw data and update the preprocessed data;
            defaults to ``False``.
        :type update: bool
        :param verbose: Whether to print relevant information in console; defaults to ``False``.
        :type verbose: bool | int
        :param kwargs: [Optional] parameters of the method
            :meth:`src.preprocessor._base.BaseVegetation._read_data`.
        :return: Data of the table "FellingType".
        :rtype: pandas.DataFrame | None

        **Examples**::

            >>> from src.preprocessor.vegetation import Vegetation
            >>> veg = Vegetation()
            >>> veg.read_felling_type()
            >>> veg.felling_type
        """

        kwargs.update(dict(table_name='FellingType', update=update, verbose=verbose))
        self._read_data(**kwargs)

    def read_area_work_type(self, update=False, verbose=False, **kwargs):
        """
        Get data of the table "AreaWorkType".

        :param update: Whether to reprocess the original/raw data and update the preprocessed data;
            defaults to ``False``.
        :type update: bool
        :param verbose: Whether to print relevant information in console; defaults to ``False``.
        :type verbose: bool | int
        :param kwargs: [Optional] parameters of the method
            :meth:`src.preprocessor._base.BaseVegetation._read_data`.
        :return: Data of the table "AreaWorkType".
        :rtype: pandas.DataFrame | None

        **Examples**::

            >>> from src.preprocessor.vegetation import Vegetation
            >>> veg = Vegetation()
            >>> veg.read_area_work_type()
            >>> veg.area_work_type
        """

        kwargs.update(dict(table_name='AreaWorkType', update=update, verbose=verbose))
        self._read_data(**kwargs)

    def read_service_detail(self, update=False, verbose=False, **kwargs):
        """
        Get data of the table "ServiceDetail".

        :param update: Whether to reprocess the original/raw data and update the preprocessed data;
            defaults to ``False``.
        :type update: bool
        :param verbose: Whether to print relevant information in console; defaults to ``False``.
        :type verbose: bool | int
        :param kwargs: [Optional] parameters of the method
            :meth:`src.preprocessor._base.BaseVegetation._read_data`.
        :return: Data of the table "ServiceDetail".
        :rtype: pandas.DataFrame | None

        **Examples**::

            >>> from src.preprocessor.vegetation import Vegetation
            >>> veg = Vegetation()
            >>> veg.read_service_detail()
            >>> veg.service_detail
        """

        kwargs.update(dict(table_name='ServiceDetail', update=update, verbose=verbose))
        self._read_data(**kwargs)

    def read_service_path(self, update=False, verbose=False, **kwargs):
        """
        Get data of the table "ServicePath".

        :param update: Whether to reprocess the original/raw data and update the preprocessed data;
            defaults to ``False``.
        :type update: bool
        :param verbose: Whether to print relevant information in console; defaults to ``False``.
        :type verbose: bool | int
        :param kwargs: [Optional] parameters of the method
            :meth:`src.preprocessor._base.BaseVegetation._read_data`.
        :return: Data of the table "ServicePath".
        :rtype: pandas.DataFrame | None

        **Examples**::

            >>> from src.preprocessor.vegetation import Vegetation
            >>> veg = Vegetation()
            >>> veg.read_service_path()
            >>> veg.service_path
        """

        kwargs.update(dict(table_name='ServicePath', update=update, verbose=verbose))
        self._read_data(**kwargs)

    def read_supplier(self, update=False, verbose=False, **kwargs):
        """
        Get data of the table "Supplier".

        :param update: Whether to reprocess the original/raw data and update the preprocessed data;
            defaults to ``False``.
        :type update: bool
        :param verbose: Whether to print relevant information in console; defaults to ``False``.
        :type verbose: bool | int
        :param kwargs: [Optional] parameters of the method
            :meth:`src.preprocessor._base.BaseVegetation._read_data`.
        :return: Data of the table "Supplier".
        :rtype: pandas.DataFrame | None

        **Examples**::

            >>> from src.preprocessor.vegetation import Vegetation
            >>> veg = Vegetation()
            >>> veg.read_supplier()
            >>> veg.supplier
        """

        kwargs.update(dict(table_name='Supplier', update=update, verbose=verbose))
        self._read_data(**kwargs)

    def read_supplier_costs(self, update=False, verbose=False, **kwargs):
        """
        Get data of the table "SupplierCosts".

        :param update: Whether to reprocess the original/raw data and update the preprocessed data;
            defaults to ``False``.
        :type update: bool
        :param verbose: Whether to print relevant information in console; defaults to ``False``.
        :type verbose: bool | int
        :param kwargs: [Optional] parameters of the method
            :meth:`src.preprocessor._base.BaseVegetation._read_data`.
        :return: Data of the table "SupplierCosts".
        :rtype: pandas.DataFrame | None

        **Examples**::

            >>> from src.preprocessor.vegetation import Vegetation
            >>> veg = Vegetation()
            >>> veg.read_supplier_costs()
            >>> veg.supplier_costs
        """

        kwargs.update(dict(table_name='SupplierCosts', update=update, verbose=verbose))
        self._read_data(**kwargs)

    def read_supplier_costs_area(self, update=False, verbose=False, **kwargs):
        """
        Get data of the table "SupplierCostsArea".

        :param update: Whether to reprocess the original/raw data and update the preprocessed data;
            defaults to ``False``.
        :type update: bool
        :param verbose: Whether to print relevant information in console; defaults to ``False``.
        :type verbose: bool | int
        :param kwargs: [Optional] parameters of the method
            :meth:`src.preprocessor._base.BaseVegetation._read_data`.
        :return: Data of the table "SupplierCostsArea".
        :rtype: pandas.DataFrame | None

        **Examples**::

            >>> from src.preprocessor.vegetation import Vegetation
            >>> veg = Vegetation()
            >>> veg.read_supplier_costs_area()
            >>> veg.supplier_costs_area
        """

        kwargs.update(dict(table_name='SupplierCostsArea', update=update, verbose=verbose))
        self._read_data(**kwargs)

    def read_supplier_costs_simple(self, update=False, verbose=False, **kwargs):
        """
        Get data of the table "SupplierCostsSimple".

        :param update: Whether to reprocess the original/raw data and update the preprocessed data;
            defaults to ``False``.
        :type update: bool
        :param verbose: Whether to print relevant information in console; defaults to ``False``.
        :type verbose: bool | int
        :param kwargs: [Optional] parameters of the method
            :meth:`src.preprocessor._base.BaseVegetation._read_data`.
        :return: Data of the table "SupplierCostsSimple".
        :rtype: pandas.DataFrame | None

        **Examples**::

            >>> from src.preprocessor.vegetation import Vegetation
            >>> veg = Vegetation()
            >>> veg.read_supplier_costs_simple()
            >>> veg.supplier_costs_simple
        """

        kwargs.update(dict(table_name='SupplierCostsSimple', update=update, verbose=verbose))
        self._read_data(**kwargs)

    def read_tree_action_fractions(self, update=False, verbose=False, **kwargs):
        """
        Get data of the table "TreeActionFractions".

        :param update: Whether to reprocess the original/raw data and update the preprocessed data;
            defaults to ``False``.
        :type update: bool
        :param verbose: Whether to print relevant information in console; defaults to ``False``.
        :type verbose: bool | int
        :param kwargs: [Optional] parameters of the method
            :meth:`src.preprocessor._base.BaseVegetation._read_data`.
        :return: Data of the table "TreeActionFractions".
        :rtype: pandas.DataFrame | None

        **Examples**::

            >>> from src.preprocessor.vegetation import Vegetation
            >>> veg = Vegetation()
            >>> veg.read_tree_action_fractions()
            >>> veg.tree_action_fractions
        """

        kwargs.update(dict(table_name='TreeActionFractions', update=update, verbose=verbose))
        self._read_data(**kwargs)

    def read_veg_surv_type_class(self, update=False, verbose=False, **kwargs):
        """
        Get data of the table "VegSurvTypeClass".

        :param update: Whether to reprocess the original/raw data and update the preprocessed data;
            defaults to ``False``.
        :type update: bool
        :param verbose: Whether to print relevant information in console; defaults to ``False``.
        :type verbose: bool | int
        :param kwargs: [Optional] parameters of the method
            :meth:`src.preprocessor._base.BaseVegetation._read_data`.
        :return: Data of the table "VegSurvTypeClass".
        :rtype: pandas.DataFrame | None

        **Examples**::

            >>> from src.preprocessor.vegetation import Vegetation
            >>> veg = Vegetation()
            >>> veg.read_veg_surv_type_class()
            >>> veg.veg_surv_type_class
        """

        kwargs.update(dict(table_name='VegSurvTypeClass', update=update, verbose=verbose))
        self._read_data(**kwargs)

    def read_wb_factors(self, update=False, verbose=False, **kwargs):
        """
        Get data of the table "WBFactors".

        :param update: Whether to reprocess the original/raw data and update the preprocessed data;
            defaults to ``False``.
        :type update: bool
        :param verbose: Whether to print relevant information in console; defaults to ``False``.
        :type verbose: bool | int
        :param kwargs: [Optional] parameters of the method
            :meth:`src.preprocessor._base.BaseVegetation._read_data`.
        :return: Data of the table "WBFactors".
        :rtype: pandas.DataFrame | None

        **Examples**::

            >>> from src.preprocessor.vegetation import Vegetation
            >>> veg = Vegetation()
            >>> veg.read_wb_factors()
            >>> veg.wb_factors
        """

        kwargs.update(
            dict(table_name='WBFactors', ivar_name='wb_factors', update=update, verbose=verbose))
        self._read_data(**kwargs)

    def read_weed_spray(self, update=False, verbose=False, **kwargs):
        """
        Get data of the table "Weedspray".

        :param update: Whether to reprocess the original/raw data and update the preprocessed data;
            defaults to ``False``.
        :type update: bool
        :param verbose: Whether to print relevant information in console; defaults to ``False``.
        :type verbose: bool | int
        :param kwargs: [Optional] parameters of the method
            :meth:`src.preprocessor._base.BaseVegetation._read_data`.
        :return: Data of the table "Weedspray".
        :rtype: pandas.DataFrame | None

        **Examples**::

            >>> from src.preprocessor.vegetation import Vegetation
            >>> veg = Vegetation()
            >>> veg.read_weed_spray()
            >>> veg.weed_spray
        """

        kwargs.update(
            dict(table_name='Weedspray', ivar_name='weed_spray', pkey=['RouteAlias'], update=update,
                 verbose=verbose))
        self._read_data(**kwargs)

    def read_work_hours(self, update=False, verbose=False, **kwargs):
        """
        Get data of the table "WorkHours".

        :param update: Whether to reprocess the original/raw data and update the preprocessed data;
            defaults to ``False``.
        :type update: bool
        :param verbose: Whether to print relevant information in console; defaults to ``False``.
        :type verbose: bool | int
        :param kwargs: [Optional] parameters of the method
            :meth:`src.preprocessor._base.BaseVegetation._read_data`.
        :return: Data of the table "WorkHours".
        :rtype: pandas.DataFrame | None

        **Examples**::

            >>> from src.preprocessor.vegetation import Vegetation
            >>> veg = Vegetation()
            >>> veg.read_work_hours()
            >>> veg.work_hours
        """

        kwargs.update(dict(table_name='WorkHours', update=update, verbose=verbose))
        self._read_data(**kwargs)

    def read_furlong_data(self, update=False, verbose=False, **kwargs):
        """
        Get data of the table "FurlongData".

        :param update: Whether to reprocess the original/raw data and update the preprocessed data;
            defaults to ``False``.
        :type update: bool
        :param verbose: Whether to print relevant information in console; defaults to ``False``.
        :type verbose: bool | int
        :param kwargs: [Optional] parameters of the method
            :meth:`src.preprocessor._base.BaseVegetation._read_data`.
        :return: Data of the table "FurlongData".
        :rtype: pandas.DataFrame | None

        **Examples**::

            >>> from src.preprocessor.vegetation import Vegetation
            >>> veg = Vegetation()
            >>> veg.read_furlong_data()
            >>> veg.furlong_data

        .. note::

            - Equipment Class: VL ('VEGETATION - 1/8 MILE SECTION')
            - 1/8 mile = 220 yards = 1 furlong
        """

        kwargs.update(
            dict(table_name='FurlongData', dtype={'StartMileage': str, 'EndMileage': str},
                 update=update, verbose=verbose))
        self._read_data(**kwargs)

    def read_furlong_location(self, update=False, key_columns_only=True, verbose=False, **kwargs):
        """
        Get data of the table "FurlongLocation".

        :param update: Whether to reprocess the original/raw data and update the preprocessed data;
            defaults to ``False``.
        :type update: bool
        :param key_columns_only: Whether to return data with only the key columns, including
            ``['Route', 'RouteAlias', 'DU', 'ELR', 'StartMileage', 'EndMileage', 'Electrified',
            'HazardOnly']``; defaults to ``True``.
        :type key_columns_only: bool
        :param verbose: Whether to print relevant information in console; defaults to ``False``.
        :type verbose: bool | int
        :param kwargs: [Optional] parameters of the method
            :meth:`src.preprocessor._base.BaseVegetation._read_data`.
        :return: Data of the table "FurlongLocation".
        :rtype: pandas.DataFrame | None

        **Examples**::

            >>> from src.preprocessor.vegetation import Vegetation
            >>> veg = Vegetation()
            >>> veg.read_furlong_location()
            >>> veg.furlong_location

        .. note::

            - One set of ELR and mileage may have multiple ``FurlongID``s.
        """

        kwargs.update(
            dict(table_name='FurlongLocation', dtype={'StartMileage': str, 'EndMileage': str},
                 update=update, verbose=verbose))
        self._read_data(**kwargs)

        # Select useful columns only?
        if key_columns_only:
            key_columns = [
                'Route',
                'RouteAlias',
                'DU',
                'ELR',
                'StartMileage',
                'EndMileage',
                'Electrified',
                'HazardOnly',
            ]
            self.furlong_location = self.furlong_location[key_columns]

    def read_hazard_tree(self, update=False, verbose=False, **kwargs):
        """
        Get data of the table "HazardTree".

        :param update: Whether to reprocess the original/raw data and update the preprocessed data;
            defaults to ``False``.
        :type update: bool
        :param verbose: Whether to print relevant information in console; defaults to ``False``.
        :type verbose: bool | int
        :param kwargs: [Optional] parameters of the method
            :meth:`src.preprocessor._base.BaseVegetation._read_data`.
        :return: Data of the table "HazardTree".
        :rtype: pandas.DataFrame | None

        **Examples**::

            >>> from src.preprocessor.vegetation import Vegetation
            >>> veg = Vegetation()
            >>> veg.read_hazard_tree()
            >>> veg.hazard_tree

        .. note::

            - There are errors in:
              - ``FurlongID``. (They could be cancelled out when the ``.hazard_tree`` dataset
                is merged with other data sets on the ``FurlongID``.)
              - ``Easting`` and ``Northing`` columns.
        """

        kwargs.update(
            dict(table_name='HazardTree', dtype={'Mileage': str}, update=update, verbose=verbose))
        self._read_data(**kwargs)

    def _update_tables(self, verbose=True):
        """
        Update the local pickle files for all Tables.

        :param verbose: Whether to print relevant information in console; defaults to ``False``.
        :type verbose: bool | int

        **Examples**::

            >>> from src.preprocessor.vegetation import Vegetation
            >>> veg = Vegetation()
            >>> veg._update_tables(verbose=True)
        """

        if confirmed(f'To update the Tables of "{self.SCHEMA_NAME}"\n?'):
            # self.read_furlong_data(update=True, verbose=verbose)
            # self.read_furlong_location(update=True, verbose=verbose)
            # self.read_hazard_tree(update=True, verbose=verbose)

            for meth in [m for m in dir(self) if m.startswith('read_') and 'mssql' not in m]:
                getattr(self, meth)(update=True, verbose=verbose)

            if verbose:
                print("\nUpdate finished.")

    # == Get views based on the table data =========================================================

    def _vegetation_coverage(self, update=False, verbose=False, **kwargs):
        """
        Get a view of data of vegetation coverage per furlong.

        :param route_name: Name of a Network Rail's Route;
            when ``route_name=None`` (default), it takes all the available Routes.
        :type route_name: str | None
        :param update: Whether to reprocess the original/raw data and update the preprocessed data;
            defaults to ``False``.
        :type update: bool
        :param verbose: Whether to print relevant information; defaults to ``False``.
        :type verbose: bool | int
        :return: Data of vegetation coverage per furlong.
        :rtype: pandas.DataFrame | None
        """

        read_args = dict(update=update, verbose=verbose)
        self.read_furlong_data(**read_args)  # (75247, 40)
        self.read_furlong_location(key_columns_only=False, **read_args)
        self.read_cutting_angle_class(**read_args)  # (5, 1)
        self.read_cutting_depth_class(**read_args)  # (5, 1)
        self.read_routes(**read_args)

        # Merge the data that has been obtained
        vegetation_coverage = self.furlong_data. \
            join(self.furlong_location,
                 on='FurlongID', how='inner', lsuffix='', rsuffix='_FurlongLocation'). \
            join(self.cutting_angle_class,
                 on='CuttingAngle', how='inner'). \
            join(self.cutting_depth_class,
                 on='CuttingDepth', how='inner', lsuffix='_CuttingAngle', rsuffix='_CuttingDepth'). \
            drop(labels=['RouteAlias', 'RouteAlias_FurlongLocation', 'Route_FurlongLocation',
                         'DU', 'ELR', 'StartMileage', 'EndMileage'],
                 axis=1)

        vegetation_coverage.rename(
            columns={'Description_CuttingAngle': 'CuttingAngleDescription',
                     'Description_CuttingDepth': 'CuttingDepthDescription'},
            inplace=True)

        # The total number of trees on both sides
        tree_no_cols = ['TreeNumberUp', 'TreeNumberDown']
        vegetation_coverage['TreeNumber'] = vegetation_coverage[tree_no_cols].sum(1)

        # Rearrange
        vegetation_coverage.sort_values(['StructuredPlantNumber', 'AssetNumber'], inplace=True)

        self._dump_prep_data(
            data=vegetation_coverage, table_name="vegetation_coverage", verbose=verbose,
            pkey=list(vegetation_coverage.index.names), **kwargs)

        return vegetation_coverage

    def view_vegetation_coverage(self, route_name=None, **kwargs):
        """
        Get a view of data of vegetation coverage per furlong.

        :param route_name:
        :param kwargs:
        :return:

        **Examples**::

            >>> from src.preprocessor.vegetation import Vegetation
            >>> veg = Vegetation()
            >>> veg.view_vegetation_coverage()
            >>> veg.vegetation_coverage.shape
            (75247, 57)
            >>> veg.view_vegetation_coverage(route_name='Anglia')
            >>> veg.vegetation_coverage.shape
            (5821, 57)
        """

        kwargs.update(
            dict(table_name='vegetation_coverage', dtype={'StartMileage': str, 'EndMileage': str}))
        self._read_data(**kwargs)

        if route_name is not None:
            if self.routes is None:
                self.read_routes()
            route_name_ = find_similar_str(route_name, list(self.routes['Route'].unique()))
            self.vegetation_coverage = self.vegetation_coverage.query(f'`Route`=="{route_name_}"')

    def _hazardous_trees(self, update=False, verbose=False, **kwargs):
        """
        Get a view of data of hazardous tress.

        :param route_name: name of a Route; if ``None`` (default), all Routes
        :type route_name: str | None
        :param update: whether to check on update and proceed to update the package data,
            defaults to ``True``
        :type update: bool
        :param verbose: whether to print relevant information in console as the function runs,
            defaults to ``False``
        :type verbose: bool | int
        :return: Data of hazardous tress
        :rtype: pandas.DataFrame | None
        """

        read_args = dict(update=update, verbose=verbose)
        self.read_hazard_tree(**read_args)  # (23950, 60) 1770 with FurlongID being -1
        self.read_furlong_location(key_columns_only=False, **read_args)
        self.read_tree_age_class(**read_args)  # (7, 1)
        self.read_tree_size_class(**read_args)  # (5, 1)

        hazardous_trees = self.hazard_tree. \
            join(self.furlong_location,  # (22180, 68)
                 on='FurlongID', how='inner', lsuffix='', rsuffix='_FurlongLocation'). \
            join(self.tree_age_class,  # (22180, 69)
                 on='TreeAgeCatID', how='inner'). \
            join(self.tree_size_class,  # (22180, 70)
                 on='TreeSizeCatID', how='inner',
                 lsuffix='_TreeAgeClass', rsuffix='_TreeSizeClass'). \
            drop(labels=['Route_FurlongLocation', 'RouteAlias_FurlongLocation', 'DU', 'ELR'],
                 axis=1)

        hazardous_trees.rename(
            columns={'Description_TreeAgeClass': 'TreeAgeClassDescription',
                     'Description_TreeSizeClass': 'TreeSizeClassDescription',
                     'StartMileage': 'FurlongStartMileage',
                     'EndMileage': 'FurlongEndMileage',
                     'Electrified': 'FurlongElectrified',
                     'HazardOnly': 'FurlongHazardOnly'},
            inplace=True)

        self._dump_prep_data(
            data=hazardous_trees, table_name="hazardous_trees", verbose=verbose,
            pkey=list(hazardous_trees.index.names), **kwargs)

        return hazardous_trees

    def view_hazardous_trees(self, route_name=None, update=False, verbose=False, **kwargs):
        """
        Get a view of data of hazardous tress.

        :param route_name:
        :param update: whether to check on update and proceed to update the package data,
            defaults to ``True``
        :type update: bool
        :param verbose: whether to print relevant information in console as the function runs,
            defaults to ``False``
        :param kwargs:
        :return:

        **Examples**::

            >>> from src.preprocessor.vegetation import Vegetation
            >>> veg = Vegetation()
            >>> veg.view_hazardous_trees()
            >>> veg.hazardous_trees.shape
            (22180, 79)
        """

        kwargs.update(
            dict(table_name='hazardous_trees',
                 dtype={'Mileage': str, 'FurlongStartMileage': str, 'FurlongEndMileage': str},
                 update=update, verbose=verbose))
        self._read_data(**kwargs)

        if route_name is not None:
            if self.routes is None:
                self.read_routes()
            route_name_ = find_similar_str(route_name, list(self.routes['Route'].unique()))
            self.hazardous_trees = self.hazardous_trees[
                self.hazardous_trees.route_names == route_name_]

    def _vegetation_condition(self, route_name=None, update=False, verbose=False, **kwargs):
        """
        Get a view of vegetation data combined with information of hazardous trees.

        :param route_name: name of a Route; if ``None`` (default), all Routes
        :type route_name: str | None
        :param update: whether to check on update and proceed to update the package data,
            defaults to ``True``
        :type update: bool
        :param verbose: whether to print relevant information in console as the function runs,
            defaults to ``False``
        :type verbose: bool | int
        :return: Vegetation data combined with information of hazardous trees
        :rtype: pandas.DataFrame | None
        """

        if self.hazardous_trees is None:
            self.view_hazardous_trees(update=update, verbose=verbose)

        group_cols = ['ELR', 'DU', 'Route', 'FurlongStartMileage', 'FurlongEndMileage']
        furlong_hazard_tree = self.hazardous_trees.groupby(group_cols).aggregate({
            # 'AssetNumber': np.count_nonzero,
            'Haztreeid': np.count_nonzero,
            'TreeheightM': [tuple, 'min', 'max'],
            'TreediameterM': [tuple, 'min', 'max'],
            'TreeproxrailM': [tuple, 'min', 'max'],
            'Treeprox3py': [tuple, 'min', 'max']})

        furlong_hazard_tree.columns = ['_'.join(x).strip() for x in furlong_hazard_tree.columns]
        furlong_hazard_tree.rename(columns={'Haztreeid_count_nonzero': 'TreeNumber'}, inplace=True)
        furlong_hazard_tree.columns = [
            'Hazard' + x.strip('_<lambda_0>') for x in furlong_hazard_tree.columns]

        #
        if self.vegetation_coverage is None:
            self.view_vegetation_coverage(update=update, verbose=verbose)

        # Processing ...
        vegetation_condition = self.vegetation_coverage.join(
            furlong_hazard_tree, on=['ELR', 'DU', 'Route', 'StartMileage', 'EndMileage'],
            how='left')
        vegetation_condition.sort_values('StructuredPlantNumber', inplace=True)  # (75247, 58)

        self._dump_prep_data(
            data=vegetation_condition, table_name="vegetation_condition", verbose=verbose,
            pkey=list(vegetation_condition.index.names), **kwargs)

        if route_name is not None:
            if self.routes is None:
                self.read_routes(update=update, verbose=verbose)
            route_name_ = find_similar_str(route_name, list(self.routes.Route.unique()))
            vegetation_condition = vegetation_condition[
                vegetation_condition['route_names'] == route_name_]

        return vegetation_condition

    def view_vegetation_condition(self, route_name=None, **kwargs):
        """
        Get a view of vegetation data combined with information of hazardous trees.

        :param route_name:
        :param kwargs:
        :return:

        **Examples**::

            >>> from src.preprocessor.vegetation import Vegetation
            >>> veg = Vegetation()
            >>> veg.view_vegetation_condition()
            >>> veg.vegetation_condition.shape
            (75247, 70)
        """

        kwargs.update(
            dict(table_name='vegetation_condition', dtype={'StartMileage': str, 'EndMileage': str}))
        self._read_data(**kwargs)

        if route_name is not None:
            if self.routes is None:
                self.read_routes()
            route_name_ = find_similar_str(route_name, list(self.routes['Route'].unique()))
            self.vegetation_condition = self.vegetation_condition.query(f'`Route`=="{route_name_}"')

    def _vegetation_condition2(self, route_name=None, update=False, verbose=False, **kwargs):
        """
        Get a view of ELR and mileage data of furlong locations.

        :param route_name: name of a Route; if ``None`` (default), all Routes
        :type route_name: str | None
        :param update: whether to check on update and proceed to update the package data,
            defaults to ``True``
        :type update: bool
        :param verbose: whether to print relevant information in console as the function runs,
            defaults to ``False``
        :type verbose: bool | int
        :return: Vegetation data combined with information of hazardous trees
        :rtype: pandas.DataFrame | None
        """

        if self.vegetation_condition is None:
            self.view_vegetation_condition(update=update, verbose=verbose)

        vegetation_condition2 = self.vegetation_condition.reset_index().set_index('FurlongID')
        vegetation_condition2.sort_index(inplace=True)

        # Column names of mileage data (as string)
        str_mileage_colnames = ['StartMileage', 'EndMileage']
        # Column names of ELR and mileage data (as string)
        elr_mileage_colnames = ['ELR'] + str_mileage_colnames

        vegetation_condition2.drop_duplicates(elr_mileage_colnames, inplace=True)
        empty_start_mileage_idx = vegetation_condition2[
            vegetation_condition2.StartMileage == ''].index
        vegetation_condition2.loc[empty_start_mileage_idx, 'StartMileage'] = [
            vegetation_condition2.StructuredPlantNumber.loc[i][11:17]
            for i in empty_start_mileage_idx]

        # Create two new columns of mileage data (as float)
        num_mileage_colnames = ['StartMileage_num', 'EndMileage_num']
        vegetation_condition2[num_mileage_colnames] = vegetation_condition2[
            str_mileage_colnames].map(mileage_str_to_num)

        # Sort the furlong data by ELR and mileage
        vegetation_condition2.sort_values(['ELR'] + num_mileage_colnames, inplace=True)

        self._dump_prep_data(
            data=vegetation_condition2, table_name="vegetation_condition2", verbose=verbose,
            pkey=list(vegetation_condition2.index.names), **kwargs)

        if route_name is not None:
            if self.routes is None:
                self.read_routes(update=update, verbose=verbose)
            route_name_ = find_similar_str(route_name, list(self.routes['Route'].unique()))
            vegetation_condition2 = vegetation_condition2[
                vegetation_condition2['route_names'] == route_name_]

        return vegetation_condition2

    def view_vegetation_condition2(self, route_name=None, **kwargs):
        """
        Get a view of ELR and mileage data of furlong locations.

        :param route_name:
        :param kwargs:
        :return:

        **Examples**::

            >>> from src.preprocessor.vegetation import Vegetation
            >>> veg = Vegetation()
            >>> veg.view_vegetation_condition2()
            >>> veg.vegetation_condition2.shape
            (75247, 75)
        """

        self._read_data(
            table_name='vegetation_condition2', dtype={'StartMileage': str, 'EndMileage': str},
            **kwargs)

        if route_name is not None:
            if self.routes is None:
                self.read_routes()
            route_name_ = find_similar_str(route_name, list(self.routes['Route'].unique()))
            self.vegetation_condition2 = self.vegetation_condition2.query(
                f'`Route`=="{route_name_}"')

    def _update_views(self, verbose=True):
        """
        Update the Views.

        :param verbose: whether to print relevant information in console as the function runs,
            defaults to ``False``
        :type verbose: bool | int

        **Examples**::

            >>> from src.preprocessor.vegetation import Vegetation
            >>> veg = Vegetation()
            >>> veg._update_views()
            >>> veg.hazardous_trees.shape
            (22180, 79)
            >>> veg.vegetation_condition.shape
            (75247, 70)
            >>> veg.vegetation_condition2.shape
            (75247, 75)
            >>> veg.vegetation_coverage.shape
            (75247, 57)
        """

        if confirmed(f'To update the Views of "{self.SCHEMA_NAME}"\n?'):

            # self.view_hazardous_trees(update=True, verbose=verbose)
            # self.view_vegetation_condition(update=True, verbose=verbose)
            # self.view_vegetation_coverage(update=True, verbose=verbose)
            # self.view_vegetation_condition2(update=True, verbose=verbose)

            for meth in [m for m in dir(self) if m.startswith('view_')]:
                getattr(self, meth)(update=True, verbose=verbose)

            if verbose:
                print("\nUpdate finished.")
