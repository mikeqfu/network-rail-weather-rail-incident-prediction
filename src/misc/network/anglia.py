"""
Network nodes and links.
"""

from collections import OrderedDict

import pandas as pd

from utils import cdd_network, merge_dicts, remove_list_duplicated_lists, \
    remove_list_duplicates


def get_anglia_route_srs_id(whole=False):
    """
    Get SRS ID for the Anglia Route.

    :param whole: whether to return a list of all SRS ID
    :type whole: bool
    :return: a list of SRS ID for the Anglia Route
    """

    route_d_srs = ['D.01', 'D.02', 'D.03', 'D.04', 'D.05', 'D.06', 'D.07',
                   'D.08', 'D.09', 'D.10', 'D.11', 'D.12', 'D.13', 'D.14',
                   'D.15', 'D.16', 'D.17', 'D.18', 'D.19', 'D.20', 'D.99']
    route_e_srs = ['E.01', 'E.02', 'E.03', 'E.04', 'E.05', 'E.91', 'E.99']
    route_f_srs = ['F.01', 'F.02', 'F.99']

    if whole:
        anglia_srs = route_d_srs + route_e_srs + route_f_srs
    else:
        anglia_srs = [route_d_srs, route_e_srs, route_f_srs]

    return anglia_srs


def get_nodes_of_srs(srs_id):
    """
    Get a list of nodes for a specific SRS.

    :param srs_id: an ID of Strategic Route Section, e.g. 'D.01'
    :type srs_id: str
    :return: a list of nodes for the given ``srs_id``
    :rtype: list

    **Example**::

        srs_id = 'D.01'
        nodes_of_srs(srs_id)
    """

    # Read excel data into a data frame named srs_df
    srs_df = pd.read_excel(cdd_network('routes\\Anglia', 'Anglia.xlsx'),
                           sheet_name=srs_id)
    # Convert a 'Node' Series to a list named srs_nodes
    srs_nodes = [each_node for each_node in srs_df.Node]

    return srs_nodes


def get_nodes_of_srs_seq(srs_id_seq):
    """
    Get a list of nodes for a set of SRS's.

    :param srs_id_seq: one or a sequence of SRS IDs
    :type srs_id_seq: str or iterable
    :return: a list of nodes for the given ``srs_id_seq``
    :rtype: list

    **Examples**::

        srs_id_seq = 'D.01'
        get_nodes_of_srs_seq(srs_id_seq)

        srs_id_seq = ['D.01', 'D.02']
        get_nodes_of_srs_seq(srs_id_seq)
    """

    # Get a list of nodes for all specified SRS's
    if isinstance(srs_id_seq, str):
        return get_nodes_of_srs(srs_id_seq)

    else:
        srs_seq_nodes = [each_node for srs_id in srs_id_seq
                         for each_node in get_nodes_of_srs(srs_id)]
        return remove_list_duplicates(srs_seq_nodes)


def get_nodes_of_route_plans(rp_id_seq):
    """
    Get a list of all nodes for the given Route Plan.

    :param rp_id_seq:
    :return:
    """

    def get_nodes(rp_srs_seq):
        # Create an empty list named srs_nodes
        seq_nodes = []
        for srs in rp_srs_seq:
            srs_n = get_nodes_of_srs(srs)
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

    route_d_srs, route_e_srs, route_f_srs = get_anglia_route_srs_id()

    rp_srs = []  # Get a list of nodes for all specified SRS's
    if 'D' in rp_id_seq:
        rp_srs = rp_srs + route_d_srs
    elif 'E' in rp_id_seq:
        rp_srs = rp_srs + route_e_srs
    elif 'F' in rp_id_seq:
        rp_srs = rp_srs + route_f_srs

    return get_nodes(rp_srs)


def get_nodes_on_anglia_route():
    """
    Get a data of all nodes on the Anglia Route.

    :return: data of all nodes on the Anglia Rout
    :rtype: pandas.DataFrame
    """

    anglia_srs = get_anglia_route_srs_id(whole=True)

    df_list = []
    anglia = pd.ExcelFile(cdd_network('routes\\Anglia', 'Anglia.xlsx'))
    for i in range(len(anglia_srs)):
        df_list.append(anglia.parse(anglia_srs[i]).fillna(''))
    anglia.close()

    df = pd.concat(df_list, ignore_index=True)

    return df


def get_list_of_node_dicts(srs_id):
    """
    Get a list of dictionaries each referring to a node for a specific SRS.

    :param srs_id: an ID of Strategic Route Section, e.g. 'D.01'
    :type srs_id: str
    :return: a list of ordered dictionaries
    :rtype: list

    **Example**::

        srs_id = 'D.01'
        get_list_of_node_dicts(srs_id)
    """

    # Read excel data into a data frame named srs_df
    srs_df = pd.read_excel(cdd_network('routes\\Anglia', 'Anglia.xlsx'),
                           sheet_name=srs_id)
    # Get the names of all the columns
    attr_name = srs_df.columns.values.tolist()
    attr_name.insert(3, 'SRS')

    srs_nodes_dict = []
    for i in range(srs_df.index[0], srs_df.shape[0]):
        node_info = list(srs_df.loc[i, :])
        node_info.insert(3, srs_id)
        srs_nodes_dict.append(OrderedDict(zip(attr_name, node_info)))

    return srs_nodes_dict


def construct_nodes_dict(lst_of_dicts, key='Node'):
    """
    Construct a dict with each key being a node name and
    the corresponding value being a dict containing the node attr.

    :param lst_of_dicts: a list of dictionaries
    :type lst_of_dicts: list
    :param key: dict key, defaults to ``'Node'``
    :type key: str
    :return: a dictionary for nodes
    :rtype: dict
    """

    # enumerate() returns a tuple containing a count (from start which defaults
    # to 0) and the values obtained from iterating over iterable.
    new_dict = dict((d[key], OrderedDict(d)) for (i, d) in enumerate(lst_of_dicts))
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


def get_nodes_dict(*srs_id):
    """
    Get a dictionary for nodes for a (sequence of) SRS('s).

    :param srs_id: one or a sequence of SRS IDs, e.g. 'D.01' or 'D.01', 'D.02'
    :type srs_id: str
    :return: a dictionary for nodes of the given (sequence of) SRS('s)
    :rtype: dict

    **Examples**::

        nodes_dict1 = get_nodes_dict('D.01')

        nodes_dict2 = get_nodes_dict('D.01', 'D.02')
    """

    nodes_dict = {}
    if isinstance(srs_id, str) and len(srs_id) == 4:
        srs_id = srs_id
        nodes_dict_ = construct_nodes_dict(get_list_of_node_dicts(srs_id))  # A dictionary
        # noinspection PyTypeChecker
        for (key, val) in nodes_dict_.items():
            if pd.isnull(nodes_dict_[key]['Connecting SRS']):
                del nodes_dict_[key]['Connecting SRS']
            else:
                nodes_dict_[key]['SRS'] = {nodes_dict_[key]['SRS'],
                                           nodes_dict_[key]['Connecting SRS']}
                del nodes_dict_[key]['Connecting SRS']
        return nodes_dict_
    else:
        for srs_id in srs_id:
            # A dictionary
            nodes_dict_ = construct_nodes_dict(get_list_of_node_dicts(srs_id))
            # noinspection PyTypeChecker
            for (key, val) in nodes_dict_.items():
                if pd.isnull(nodes_dict_[key]['Connecting SRS']):
                    del nodes_dict_[key]['Connecting SRS']
                else:
                    nodes_dict_[key]['SRS'] = {nodes_dict_[key]['SRS'],
                                               nodes_dict_[key]['Connecting SRS']}
                    del nodes_dict_[key]['Connecting SRS']
            nodes_dict = merge_dicts(nodes_dict, nodes_dict_)
        return nodes_dict


def get_nodes_dict_for_route_plans(*rp_id):
    """
    Get a dictionary for nodes for one or more route plans.

    :param rp_id: route plan id, e.g. 'D'
    :type rp_id: str
    :return: a dictionary for nodes for the given route plans
    :rtype: dict

    **Examples**::

        rp_nodes_dict1 = get_nodes_dict_for_route_plans('D')
        
        rp_nodes_dict2 = get_nodes_dict_for_route_plans('D', 'E')
    """

    assert all(x in 'DEF' for x in rp_id)

    route_d_srs, route_e_srs, route_f_srs = get_anglia_route_srs_id()

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
        nodes_dict = construct_nodes_dict(get_list_of_node_dicts(srs_id))  # A dictionary
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


def get_edges_of_anglia_route(direct=False):
    """
    Get all edges on the Network of the Anglia Route.

    :param direct: to return all the edges for a directed graph if True,
        otherwise for an undirected graph
    :type direct: bool
    :return: all the edges of the Anglia Route
    :rtype: list

    **Examples**::

        direct = False
        edges_undirected = get_edges_of_anglia_route(direct)

        direct = True
        edges_direct = get_edges_of_anglia_route(direct)
    """

    # Adjacency 'matrix' (DataFrame)
    adj_mat = pd.read_excel(cdd_network("routes\\Anglia", "Anglia.xlsx"),
                            sheet_name='AdjacencyMatrix')
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


def get_edges_of_srs(*srs_id, direct=False):
    """
    Get all edges of the given SRS's on the Anglia Route.

    :param srs_id: a sequence of SRS ID's
    :type srs_id: str
    :param direct: to return all the edges for a directed graph if True,
        otherwise for an undirected graph
    :type direct: bool
    :return: all the edges for the given SRS's on an undirected or directed Anglia Network
    :rtype: list
    
    **Examples**::

        direct = False
        edges = get_edges_of_srs('D.01', direct=direct)

        direct = True
        edges = get_edges_of_srs('D.01', 'D.02', direct=direct)
    """

    route_edges = get_edges_of_anglia_route(direct)

    def sub_edges(nodes_set):
        # nodes_set includes all nodes for the specified SRS, e.g. get_nodes_of_srs('D.01')
        edges_subset = []
        for edge in route_edges:
            if edge[0] in nodes_set:
                edges_subset.append(edge)
        return edges_subset

    if isinstance(srs_id, str) and len(srs_id) == 4:
        nodes_seq = get_nodes_of_srs(srs_id)
    else:
        nodes_seq = [n for srs_id in srs_id for n in get_nodes_of_srs_seq(srs_id)]

    if direct:
        edges = sub_edges(nodes_seq)
    else:
        edges = remove_list_duplicated_lists(sub_edges(nodes_seq))

    return edges


def get_edges_of_route_plan(*rp_id, direct=False):
    """
    Get all edges of the given route plan of the Anglia Route.

    :param rp_id: route plan id, e.g. 'D'
    :type rp_id: str
    :param direct: to return all the edges for a directed graph if True,
        otherwise for an undirected graph
    :type direct: bool
    :return: all the edges for the given route plan of
        an undirected or directed Anglia Network
    :rtype: list

    **Examples**::

        direct = True
        edges = get_edges_of_route_plan('D', direct=direct)

        direct = False
        edges = get_edges_of_route_plan('E', 'F', direct=direct)
    """

    assert all(x in 'DEF' for x in rp_id)

    route_d_srs, route_e_srs, route_f_srs = get_anglia_route_srs_id()

    rp_srs = []
    # Get a list of nodes for all specified SRS's
    if 'D' in rp_id:
        rp_srs = rp_srs + route_d_srs
    if 'E' in rp_id:
        rp_srs = rp_srs + route_e_srs
    if 'F' in rp_id:
        rp_srs = rp_srs + route_f_srs

    edges = get_edges_of_srs(*rp_srs, direct=direct)

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
