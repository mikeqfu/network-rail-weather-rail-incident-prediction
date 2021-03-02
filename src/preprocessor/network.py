"""
Network nodes and links.
"""

from collections import OrderedDict

from utils import *


class Anglia:
    """
    Anglia Route.

    :ivar str Name: name of the data resource
    :ivar str Filename: name of spreadsheet
    :ivar list SRS_D: list of IDs for the Strategic Route Section 'D'
    :ivar list SRS_E: list of IDs for the Strategic Route Section 'E'
    :ivar list SRS_F: list of IDs for the Strategic Route Section 'F'

    *Test*::

        >>> from preprocessor import Anglia

        >>> anglia = Anglia()

        >>> anglia.Name
        'Anglia'
    """

    def __init__(self):
        self.Name = 'Anglia'

        self.Filename = "Anglia.xlsx"

        self.SRS_D = ['D.01', 'D.02', 'D.03', 'D.04', 'D.05', 'D.06', 'D.07',
                      'D.08', 'D.09', 'D.10', 'D.11', 'D.12', 'D.13', 'D.14',
                      'D.15', 'D.16', 'D.17', 'D.18', 'D.19', 'D.20', 'D.99']
        self.SRS_E = ['E.01', 'E.02', 'E.03', 'E.04', 'E.05', 'E.91', 'E.99']
        self.SRS_F = ['F.01', 'F.02', 'F.99']

    def cdd(self, *sub_dir, mkdir=False):
        """
        Change directory to "data\\network\\routes\\Anglia" and sub-directories / a file.

        :param sub_dir: name of directory or names of directories (and/or a file)
        :type sub_dir: str
        :param mkdir: whether to create a directory, defaults to ``False``
        :type mkdir: bool
        :return: full path to ``"data\\network\\routes\\Anglia"`` and sub-directories / a file
        :rtype: str

        **Test**::

            >>> import os
            >>> from preprocessor.network import Anglia

            >>> anglia = Anglia()

            >>> print(os.path.relpath(anglia.cdd()))
            data\\network\\routes\\Anglia
        """

        path = cdd_network("routes", self.Name, *sub_dir, mkdir=mkdir)

        return path

    def get_anglia_route_srs_id(self, whole=False):
        """
        Get SRS ID for the Anglia Route.

        :param whole: whether to return a list of all SRS ID
        :type whole: bool
        :return: a list of SRS ID for the Anglia Route
        :rtype: list

        **Test**::

            >>> from preprocessor.network import Anglia

            >>> anglia = Anglia()

            >>> anglia_srs_ = anglia.get_anglia_route_srs_id()
            >>> print(anglia_srs_[-1])
            ['F.01', 'F.02', 'F.99']
        """

        if whole:
            anglia_srs = self.SRS_D + self.SRS_E + self.SRS_F
        else:
            anglia_srs = [self.SRS_D, self.SRS_E, self.SRS_F]

        return anglia_srs

    def get_nodes_of_srs(self, srs_id):
        """
        Get a list of nodes for a specific SRS.

        :param srs_id: an ID of Strategic Route Section, e.g. ``'D.01'``
        :type srs_id: str
        :return: a list of nodes for the given ``srs_id``
        :rtype: list

        **Test**::

            >>> from preprocessor.network import Anglia

            >>> anglia = Anglia()

            >>> srs_nodes_ = anglia.get_nodes_of_srs(srs_id='D.01')
            >>> print(srs_nodes_[:5])
            ['Bethnal Green East Junction',
             'Bethnal Green',
             'Bethnal Green North Junction',
             'Cambridge Heath',
             'London Fields']
        """

        # Read excel data into a data frame named srs_df
        srs_df = pd.read_excel(self.cdd(self.Filename), sheet_name=srs_id)
        # Convert a 'Node' Series to a list named srs_nodes
        srs_nodes = [each_node for each_node in srs_df.Node]

        return srs_nodes

    def get_nodes_of_srs_seq(self, srs_id_seq):
        """
        Get a list of nodes for a set of SRS's.

        :param srs_id_seq: one or a sequence of SRS IDs
        :type srs_id_seq: str or iterable
        :return: a list of nodes for the given ``srs_id_seq``
        :rtype: list

        **Test**::

            >>> from preprocessor.network import Anglia

            >>> anglia = Anglia()

            >>> nodes_of_srs = anglia.get_nodes_of_srs_seq(srs_id_seq='D.01')
            >>> print(nodes_of_srs[:5])
            ['Bethnal Green East Junction',
             'Bethnal Green',
             'Bethnal Green North Junction',
             'Cambridge Heath',
             'London Fields']

            >>> nodes_of_srs = anglia.get_nodes_of_srs_seq(srs_id_seq=['D.01', 'D.02'])
            >>> print(nodes_of_srs[-5:])
            ['Southbury',
             'Turkey Street',
             'Theobalds Grove',
             'Bush Hill Park',
             'Enfield Town']
        """

        # Get a list of nodes for all specified SRS's
        if isinstance(srs_id_seq, str):
            nodes_of_srs_seq = self.get_nodes_of_srs(srs_id_seq)

        else:
            srs_seq_nodes = [each_node for srs_id in srs_id_seq
                             for each_node in self.get_nodes_of_srs(srs_id)]
            nodes_of_srs_seq = remove_list_duplicates(srs_seq_nodes)

        return nodes_of_srs_seq

    def get_nodes_of_route_plans(self, rp_id_seq):
        """
        Get a list of all nodes of a Route Plan.

        :param rp_id_seq: (a list of) Route Plan ID(s)
        :type rp_id_seq: str or list
        :return: a list of nodes for the Route Plan
        :rtype: list

        **Test**::

            >>> from preprocessor.network import Anglia

            >>> anglia = Anglia()

            >>> rp_nodes = anglia.get_nodes_of_route_plans(rp_id_seq='D')
            >>> print(rp_nodes[-5:])
            ['Sizewell',
             'Middleton Towers',
             'Griffin Wharf West Bank Terminal',
             'Whitemoor Yard',
             'Whitemoor Local Distribution Centre']

            >>> rp_nodes = anglia.get_nodes_of_route_plans(rp_id_seq=['E', 'F'])
            >>> print(rp_nodes[-5:])
            ['Primrose Hill Junction',
             'Camden Junction',
             'Camden Road Central Junction',
             'Camden Road Incline Junction',
             'Copenhagen Junction']
        """

        def get_nodes(rp_srs_seq):
            # Create an empty list named srs_nodes
            seq_nodes = []
            for srs in rp_srs_seq:
                srs_n = self.get_nodes_of_srs(srs)
                # Add every node to the list srs_nodes
                for each_n in srs_n:
                    seq_nodes.append(each_n)
            return remove_list_duplicates(seq_nodes)

        # route_d_srs = list(i for i in range(1,21))
        # for i in route_d_srs:
        #     if i < 10:
        #         route_d_srs[i-1] = 'D.0' + str(i)
        #     else:
        #         route_d_srs[i-1] = 'D.' + str(i)
        # route_d_srs.append('D.99')
        # route_e_srs = list(i for i in range(1,6))
        # for i in route_e_srs:
        #     route_e_srs[i-1] = 'E.0' + str(i)
        # route_e_srs.append('E.91')
        # route_e_srs.append('E.99')
        # route_f_srs = ['F.01', 'F.02', 'F.99']

        route_d_srs, route_e_srs, route_f_srs = self.get_anglia_route_srs_id()

        rp_srs = []  # Get a list of nodes for all specified SRS's
        if 'D' in rp_id_seq:
            rp_srs = rp_srs + route_d_srs
        elif 'E' in rp_id_seq:
            rp_srs = rp_srs + route_e_srs
        elif 'F' in rp_id_seq:
            rp_srs = rp_srs + route_f_srs

        return get_nodes(rp_srs)

    def get_nodes_on_anglia_route(self):
        """
        Get a data of all nodes on the Anglia Route.

        :return: data of all nodes on the Anglia Rout
        :rtype: pandas.DataFrame

        **Test**::

            >>> from preprocessor.network import Anglia

            >>> anglia = Anglia()

            >>> anglia_nodes_dat = anglia.get_nodes_on_anglia_route()
            >>> print(anglia_nodes_dat)
                                                        Node  ...          Connecting Line
            0                    Bethnal Green East Junction  ...  Great Eastern Main Line
            1                                  Bethnal Green  ...
            2                   Bethnal Green North Junction  ...
            3                                Cambridge Heath  ...
            4                                  London Fields  ...
            ..                                           ...  ...                      ...
            424                                 Thames Haven  ...
            425                         Gas Factory Junction  ...          Essex Thameside
            426                                 Bow Junction  ...  Great Eastern Main Line
            427                        Tilbury West Junction  ...
            428  Tilbury International Rail Freight Terminal  ...
            [429 rows x 6 columns]
        """

        anglia_srs = self.get_anglia_route_srs_id(whole=True)

        df_list = []
        f = pd.ExcelFile(self.cdd(self.Filename))
        for i in range(len(anglia_srs)):
            df_list.append(f.parse(anglia_srs[i]).fillna(''))
        f.close()

        df = pd.concat(df_list, ignore_index=True)

        return df

    def get_list_of_node_dicts(self, srs_id):
        """
        Get a list of dictionaries each referring to a node for a specific SRS.

        :param srs_id: an ID of Strategic Route Section, e.g. 'D.01'
        :type srs_id: str
        :return: a list of ordered dictionaries
        :rtype: list

        **Test**::

            >>> from preprocessor.network import Anglia

            >>> anglia = Anglia()

            >>> srs_nodes_dict_ = anglia.get_list_of_node_dicts(srs_id='D.01')
            >>> list(srs_nodes_dict_[0].keys())
            ['Node', 'Type', 'SRS', 'Connecting SRS', 'Line', 'Connecting Line']
        """

        # Read excel data into a data frame named srs_df
        srs_df = pd.read_excel(self.cdd(self.Filename), sheet_name=srs_id)
        # Get the names of all the columns
        attr_name = srs_df.columns.values.tolist()
        attr_name.insert(3, 'SRS')

        srs_nodes_dict = []
        for i in range(srs_df.index[0], srs_df.shape[0]):
            node_info = list(srs_df.loc[i, :])
            node_info.insert(3, srs_id)
            srs_nodes_dict.append(OrderedDict(zip(attr_name, node_info)))

        return srs_nodes_dict

    @staticmethod
    def construct_nodes_dict(list_of_dicts, key='Node'):
        """
        Construct a dict with each key being a node name and
        the corresponding value being a dict containing the node attr.

        :param list_of_dicts: a list of dictionaries
        :type list_of_dicts: list
        :param key: dict key, defaults to ``'Node'``
        :type key: str
        :return: a dictionary for nodes
        :rtype: dict

        **Test**::

            >>> from preprocessor.network import Anglia

            >>> anglia = Anglia()

            >>> list_of_dicts_ = anglia.get_list_of_node_dicts('D.01')
            >>> new_dict_ = anglia.construct_nodes_dict(list_of_dicts_, key='Node')
            >>> list(new_dict_.keys())[:5]
            ['Bethnal Green East Junction',
             'Bethnal Green',
             'Bethnal Green North Junction',
             'Cambridge Heath',
             'London Fields']
        """

        # enumerate() returns a tuple containing a count (from start which defaults
        # to 0) and the values obtained from iterating over iterable.
        new_dict = dict((d[key], OrderedDict(d)) for (i, d) in enumerate(list_of_dicts))
        # enumerate(list of dictionaries)
        # i: index of a dictionary in a list
        # d: dictionary itself
        # For each pair of (i, d), make a tuple containing a 'key' being specified and a dict

        # Or:
        # new_dict = OrderedDict((d[key], OrderedDict(d)) for (i, d) in enumerate(lst_of_dict))

        # noinspection PyTypeChecker

        for key, val in new_dict.items():
            del val['Node']

        return new_dict

    def get_nodes_dict(self, *srs_id):
        """
        Get a dictionary for nodes for a (sequence of) SRS('s).

        :param srs_id: one or a sequence of SRS IDs, e.g. 'D.01' or 'D.01', 'D.02'
        :type srs_id: str
        :return: a dictionary for nodes of the given (sequence of) SRS('s)
        :rtype: dict

        **Test**::

            >>> from preprocessor.network import Anglia

            >>> anglia = Anglia()

            >>> nodes_dict1 = anglia.get_nodes_dict('D.01')
            >>> list(nodes_dict1.keys())[-5:]
            ['Bishops Stortford',
             'Stansted Mountfitchet',
             'Stansted South Junction',
             'Stansted East Junction',
             'Stansted Airport']

            >>> nodes_dict2 = anglia.get_nodes_dict('D.01', 'D.02')
            >>> list(nodes_dict2.keys())[-5:]
            ['Southbury',
             'Turkey Street',
             'Theobalds Grove',
             'Bush Hill Park',
             'Enfield Town']
        """

        if isinstance(srs_id, str) and len(srs_id) == 4:
            srs_id = srs_id

            nodes_dict = self.construct_nodes_dict(self.get_list_of_node_dicts(srs_id))

            for (key, val) in nodes_dict.items():
                if pd.isnull(nodes_dict[key]['Connecting SRS']):
                    del nodes_dict[key]['Connecting SRS']

                else:
                    nodes_dict[key]['SRS'] = {nodes_dict[key]['SRS'], nodes_dict[key]['Connecting SRS']}
                    del nodes_dict[key]['Connecting SRS']

        else:
            nodes_dict = {}

            for srs_id in srs_id:
                nodes_dict_ = self.construct_nodes_dict(self.get_list_of_node_dicts(srs_id))

                for (key, val) in nodes_dict_.items():
                    if pd.isnull(nodes_dict_[key]['Connecting SRS']):
                        del nodes_dict_[key]['Connecting SRS']
                    else:
                        nodes_dict_[key]['SRS'] = {
                            nodes_dict_[key]['SRS'], nodes_dict_[key]['Connecting SRS']}
                        del nodes_dict_[key]['Connecting SRS']

                nodes_dict = merge_dicts(nodes_dict, nodes_dict_)

        return nodes_dict

    def get_nodes_dict_for_route_plans(self, *rp_id):
        """
        Get a dictionary for nodes for one or more route plans.

        :param rp_id: route plan id, e.g. 'D'
        :type rp_id: str
        :return: a dictionary for nodes for the given route plans
        :rtype: dict

        **Test**::

            >>> from preprocessor.network import Anglia

            >>> anglia = Anglia()

            >>> rp_nodes_dict1 = anglia.get_nodes_dict_for_route_plans('D')
            >>> list(rp_nodes_dict1.keys())[-5:]
            ['Sizewell',
             'Middleton Towers',
             'Griffin Wharf West Bank Terminal',
             'Whitemoor Yard',
             'Whitemoor Local Distribution Centre']

            >>> rp_nodes_dict2 = anglia.get_nodes_dict_for_route_plans('D', 'E')
            >>> list(rp_nodes_dict2.keys())[-5:]
            ['Primrose Hill Junction',
             'Camden Junction',
             'Camden Road Central Junction',
             'Camden Road Incline Junction',
             'Copenhagen Junction']
        """

        assert all(x in 'DEF' for x in rp_id)

        route_d_srs, route_e_srs, route_f_srs = self.get_anglia_route_srs_id()

        rp_srs = []
        # Get a list of nodes for all specified SRS's
        if 'D' in rp_id:
            rp_srs += route_d_srs
        if 'E' in rp_id:
            rp_srs += route_e_srs
        if 'F' in rp_id:
            rp_srs += route_f_srs

        rp_nodes_dict = {}
        for srs_id in rp_srs:
            nodes_dict = self.construct_nodes_dict(self.get_list_of_node_dicts(srs_id))  # A dictionary
            # noinspection PyTypeChecker
            for (key, val) in nodes_dict.items():
                if pd.isnull(nodes_dict[key]['Connecting SRS']):
                    del nodes_dict[key]['Connecting SRS']
                else:
                    nodes_dict[key]['SRS'] = {nodes_dict[key]['SRS'],
                                              nodes_dict[key]['Connecting SRS']}
                    del nodes_dict[key]['Connecting SRS']
            rp_nodes_dict = merge_dicts(rp_nodes_dict, nodes_dict)

        return rp_nodes_dict

    def get_edges_of_anglia_route(self, direct=False):
        """
        Get all edges on the Network of the Anglia Route.

        :param direct: to return all the edges for a directed graph if True,
            otherwise for an undirected graph
        :type direct: bool
        :return: all the edges of the Anglia Route
        :rtype: list

        **Test**::

            >>> from preprocessor.network import Anglia

            >>> anglia = Anglia()

            >>> edges_undirected = anglia.get_edges_of_anglia_route()
            >>> print(edges_undirected[:2])
            [['Bethnal Green East Junction', 2],
             ['Bethnal Green East Junction', 106]]

            >>> edges_direct = anglia.get_edges_of_anglia_route(direct=True)
            >>> print(edges_direct[:2])
            [['Bethnal Green East Junction', 2],
             ['Bethnal Green East Junction', 106]]
        """

        # Adjacency 'matrix' (DataFrame)
        adj_mat = pd.read_excel(self.cdd(self.Filename), sheet_name='AdjacencyMatrix')
        # row names of the adjacency 'matrix'
        # row = adj_mat.index.tolist()
        # column names in a list
        col = adj_mat.columns.values.tolist()

        # Construct pairs of nodes. Each pair refers to an edge
        edges = []
        # edge_temp = []
        # node1 = []
        # node2 = []
        # node2_temp = []
        for node1 in col:
            node2 = adj_mat[adj_mat[node1] == 1].index.tolist()
            for node2_temp in node2:
                edge_temp = [node1, node2_temp]
                edges.append(edge_temp)
        del edge_temp, node1, node2, node2_temp
        if direct is True:
            return edges
        elif direct is False:
            return remove_list_duplicated_lists(edges)
        else:
            print('InputErrors: input of "direct" must be a Boolean variable.')

    def get_edges_of_srs(self, *srs_id, direct=False):
        """
        Get all edges of the given SRS's on the Anglia Route.

        :param srs_id: a sequence of SRS ID's
        :type srs_id: str
        :param direct: to return all the edges for a directed graph if True,
            otherwise for an undirected graph
        :type direct: bool
        :return: all the edges for the given SRS's on an undirected or directed Anglia Network
        :rtype: list

        **Test**::

            >>> from preprocessor.network import Anglia

            >>> anglia = Anglia()

            >>> edges_ = anglia.get_edges_of_srs('D.01')
            >>> print(edges_[-5:])
            [['Stansted South Junction', 33],
             ['Stansted South Junction', 59],
             ['Stansted East Junction', 32],
             ['Stansted East Junction', 34],
             ['Stansted Airport', 33]]

            >>> edges_ = anglia.get_edges_of_srs('D.01', 'D.02', direct=True)
            >>> print(edges_[-5:])
            [['Theobalds Grove', 21],
             ['Theobalds Grove', 46],
             ['Bush Hill Park', 44],
             ['Bush Hill Park', 49],
             ['Enfield Town', 48]]
        """

        route_edges = self.get_edges_of_anglia_route(direct)

        def sub_edges(nodes_set):
            # nodes_set includes all nodes for the specified SRS, e.g. get_nodes_of_srs('D.01')
            edges_subset = []
            for edge in route_edges:
                if edge[0] in nodes_set:
                    edges_subset.append(edge)
            return edges_subset

        if isinstance(srs_id, str) and len(srs_id) == 4:
            nodes_seq = self.get_nodes_of_srs(srs_id)
        else:
            nodes_seq = [n for srs_id in srs_id for n in self.get_nodes_of_srs_seq(srs_id)]

        if direct:
            edges = sub_edges(nodes_seq)
        else:
            edges = remove_list_duplicated_lists(sub_edges(nodes_seq))

        return edges

    def get_edges_of_route_plan(self, *rp_id, direct=False):
        """
        Get all edges of the given route plan of the Anglia Route.

        :param rp_id: route plan id, e.g. 'D'
        :type rp_id: str
        :param direct: to return all the edges for a directed graph if True,
            otherwise for an undirected graph
        :type direct: bool
        :return: all the edges for the given route plan of an undirected or directed Anglia Network
        :rtype: list

        **Test**::

            >>> from preprocessor.network import Anglia

            >>> anglia = Anglia()

            >>> edges_ = anglia.get_edges_of_route_plan('D')
            >>> print(edges_[-5:])
            [['Felixstowe Beach Junction', 234],
             ['Felixstowe Beach Junction', 236],
             ['Felixstowe', 235],
             ['Middleton Towers', 78],
             ['Whitemoor Local Distribution Centre', 240]]

            >>> edges_ = anglia.get_edges_of_route_plan('E', 'F', direct=True)
            >>> print(edges_[-5:])
            [['Ockendon', 351],
             ['Chafford Hundred', 340],
             ['Chafford Hundred', 350],
             ['Thames Haven', 348],
             ['Tilbury International Rail Freight Terminal', 346]]
        """

        assert all(x in 'DEF' for x in rp_id)

        route_d_srs, route_e_srs, route_f_srs = self.get_anglia_route_srs_id()

        rp_srs = []
        # Get a list of nodes for all specified SRS's
        if 'D' in rp_id:
            rp_srs = rp_srs + route_d_srs
        if 'E' in rp_id:
            rp_srs = rp_srs + route_e_srs
        if 'F' in rp_id:
            rp_srs = rp_srs + route_f_srs

        edges = self.get_edges_of_srs(*rp_srs, direct=direct)

        return edges

    # --------------------------------------------------------------------------------------
    # import networkx as nx
    #
    # def create(srs_id_seq, direct=False):
    #     """
    #     :param srs_id_seq: One or a sequence of SRS ID's
    #     :param direct:
    #     :return:
    #     """
    #
    #     # Get all edges for the specified SRS's
    #     global g
    #     edges_set = edges_of_srs_seq(srs_id_seq, direct=direct)
    #
    #     # Convert every element of 'edges_indirect' to a tuple
    #     edges_tuples = []
    #     for edge in edges_set:
    #         edges_tuples.append(tuple(edge))
    #
    #     # Create a graph named G
    #     if direct is True:
    #         g = nx.DiGraph(SRS=srs_id_seq)
    #     elif direct is False:
    #         g = nx.Graph(SRS=srs_id_seq)
    #     else:
    #         pass
    #
    #     for edge in edges_tuples:
    #         g.add_edge(*edge)
    #
    #     print('No. of nodes:', g.number_of_nodes())
    #     print('No. of edges:', g.number_of_edges())
    #     print(g.graph)
    #
    #     # Get a list of nodes for the specified SRS's
    #     # graph.nodes()
    #
    #     # Get a dictionary of the nodes, including the information for each node
    #     # nodes.dic(*rp_id_seq)
    #
    #     # Draw a graph for the Network
    #     return nx.draw(g)
