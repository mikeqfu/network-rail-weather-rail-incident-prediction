""" Network nodes and links """

from collections import OrderedDict

import pandas as pd

from utils import cdd


# Remove duplicates in a list
def unique(lst):
    """
    :param lst: list
    :return: list
    """
    output = []
    temp = set()
    for item in lst:
        if item not in temp:
            output.append(item)
            temp.add(item)
    del temp
    return output


# Make each item in a list be unique, where the item is also a list
def unique_lst(lst_lst):
    """
    :param lst_lst: [list] A list of lists
    :return: [list] A list of lists with each item-list being unique
    """
    output = []
    temp = set()
    for lst in lst_lst:
        if any(item not in temp for item in lst):
            # lst[0] not in temp and lst[1] not in temp:
            output.append(lst)
            for item in lst:
                temp.add(item)  # e.g. temp.add(lst[0]); temp.add(lst[1])
    del temp
    return output


# Get a list of nodes for a specific SRS
def nodes_of_srs(srs_id):
    """
    :param srs_id: [string] An ID of Strategic Route Section, e.g. "D.01"
    :return: [list]
    """
    # Read excel data into a data frame named srs_df
    srs_df = pd.read_excel(cdd('Network\\Routes\\Anglia', 'Anglia.xlsx'), sheetname=srs_id)
    # Convert a 'Node' Series to a list named srs_nodes
    srs_nodes = [each_node for each_node in srs_df.Node]
    return srs_nodes


# Get a list of nodes for a set of SRS's
def nodes_of_srs_seq(srs_id_seq):
    """
    :param srs_id_seq: One or a sequence of SRS IDs
    :return:
    """
    # Get a list of nodes for all specified SRS's
    if isinstance(srs_id_seq, str):
        return nodes_of_srs(srs_id_seq)
    else:
        srs_seq_nodes = [each_node for srs_id in srs_id_seq for each_node in nodes_of_srs(srs_id)]
        return unique(srs_seq_nodes)


# Get a list of all nodes for the specified Route Plan
def nodes_of_route_plans(rp_id_seq):
    """
    :param rp_id_seq:
    :return:
    """

    def get_nodes(rp_srs_seq):
        # Create an empty list named srs_nodes
        seq_nodes = []
        for srs in rp_srs_seq:
            srs_n = nodes_of_srs(srs)
            # Add every node to the list srs_nodes
            for each_n in srs_n:
                seq_nodes.append(each_n)
        return unique(seq_nodes)

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

    route_d_srs = ['D.01', 'D.02', 'D.03', 'D.04', 'D.05', 'D.06', 'D.07',
                   'D.08', 'D.09', 'D.10', 'D.11', 'D.12', 'D.13', 'D.14',
                   'D.15', 'D.16', 'D.17', 'D.18', 'D.19', 'D.20', 'D.99']
    route_e_srs = ['E.01', 'E.02', 'E.03', 'E.04', 'E.05', 'E.91', 'E.99']
    route_f_srs = ['F.01', 'F.02', 'F.99']

    rp_srs = []
    # Get a list of nodes for all specified SRS's
    if 'D' in rp_id_seq:
        rp_srs = rp_srs + route_d_srs
    if 'E' in rp_id_seq:
        rp_srs = rp_srs + route_e_srs
    if 'F' in rp_id_seq:
        rp_srs = rp_srs + route_f_srs
    return get_nodes(rp_srs)


def anglia_nodes():
    route_d_srs = ['D.01', 'D.02', 'D.03', 'D.04', 'D.05', 'D.06', 'D.07',
                   'D.08', 'D.09', 'D.10', 'D.11', 'D.12', 'D.13', 'D.14',
                   'D.15', 'D.16', 'D.17', 'D.18', 'D.19', 'D.20', 'D.99']
    route_e_srs = ['E.01', 'E.02', 'E.03', 'E.04', 'E.05', 'E.91', 'E.99']
    route_f_srs = ['F.01', 'F.02', 'F.99']
    anglia_srs = route_d_srs + route_e_srs + route_f_srs

    network_file = cdd('Network\\Routes\\Anglia', 'Anglia.xlsx')
    df_list = []
    anglia = pd.ExcelFile(network_file)
    for i in range(len(anglia_srs)):
        df_list.append(anglia.parse(anglia_srs[i]).fillna(''))
    anglia.close()
    df = pd.concat(df_list, ignore_index=True)
    return df


# Get a list of dictionaries each referring to a node for a specific SRS
def list_of_dicts(srs_id):
    """

    :param srs_id: An ID of Strategic Route Section, e.g. "D.01"
    :return: A list of dictionaries
    """
    # Read excel data into a data frame named srs_df
    srs_df = pd.read_excel(cdd('Network\\Routes\\Anglia', 'Anglia.xlsx'), sheetname=srs_id)
    # Get the names of all the columns
    attr_name = srs_df.columns.values.tolist()
    attr_name.insert(3, 'SRS')

    srs_nodes_dict = []
    for i in range(srs_df.index[0], srs_df.shape[0]):
        node_info = list(srs_df.ix[i, :])
        node_info.insert(3, srs_id)
        srs_nodes_dict.append(OrderedDict(zip(attr_name, node_info)))
    return srs_nodes_dict


# Get the index of a dictionary in a list by a given (key, value)
def get_index(lst_of_dict, key, value):
    """
    :param lst_of_dict:
    :param key:
    :param value:
    :return:
    """
    next(index for (index, d) in enumerate(lst_of_dict) if d[key] == value)


# Construct a dict with each key referring to a node name and the corresponding value is a dict containing the node attr
def build_dict(lst_of_dicts, key='Node'):
    """
    :param lst_of_dicts:
    :param key: String (Default: 'Node')
    :return:
    """
    # enumerate() returns a tuple containing a count (from start which defaults
    # to 0) and the values obtained from iterating over iterable.
    new_dict = dict((d[key], OrderedDict(d)) for (i, d) in enumerate(lst_of_dicts))
    # enumerate(list of dictionaries)
    # i: index of a dictionary in a list
    # d: dictionary itself
    # For each pair of (i, d), make a tuple containing a 'key' being specified
    # and a dictionary

    # Or:
    # new_dict = OrderedDict((d[key], OrderedDict(d))
    #                        for (i, d) in enumerate(lst_of_dict))

    for (key, val) in new_dict.items():
        del val['Node']
    return new_dict


# Given two dicts, merge them into a new dict as a shallow copy.
def merge_two_dicts(dict1, dict2):
    """
    :param dict1:
    :param dict2:
    :return:
    """
    new_dict = dict1.copy()
    new_dict.update(dict2)
    return new_dict


# Given any number of dicts, shallow copy and merge into a new dict, precedence goes to key value pairs in latter dicts.
def merge_dicts(*dicts):
    """
    :param dicts:
    :return:
    """
    new_dict = {}
    for d in dicts:
        new_dict.update(d)
    return new_dict


# Get a dictionary for nodes for a set of SRS's
def dic(srs_id_seq):
    """
    :param srs_id_seq: One or a sequence of SRS IDs, e.g. 'D.01' or 'D.01, D.02'
    :return:
    """
    new_dict = {}
    if isinstance(srs_id_seq, str) and len(srs_id_seq) == 4:
        srs_id = srs_id_seq
        nodes_dict = build_dict(list_of_dicts(srs_id))  # A dictionary
        for (key, val) in nodes_dict.items():
            if pd.isnull(nodes_dict[key]['Connecting SRS']):
                del nodes_dict[key]['Connecting SRS']
            else:
                nodes_dict[key]['SRS'] = {nodes_dict[key]['SRS'], nodes_dict[key]['Connecting SRS']}
                del nodes_dict[key]['Connecting SRS']
        return nodes_dict
    else:
        for srs_id in srs_id_seq:
            nodes_dict = build_dict(list_of_dicts(srs_id))  # A dictionary
            for (key, val) in nodes_dict.items():
                if pd.isnull(nodes_dict[key]['Connecting SRS']):
                    del nodes_dict[key]['Connecting SRS']
                else:
                    nodes_dict[key]['SRS'] = {nodes_dict[key]['SRS'], nodes_dict[key]['Connecting SRS']}
                    del nodes_dict[key]['Connecting SRS']
            new_dict = merge_dicts(new_dict, nodes_dict)
        return new_dict


# Get a dictionary for nodes for one or more specific route plans
def dic_r(rp_id_seq):
    route_d_srs = ['D.01', 'D.02', 'D.03', 'D.04', 'D.05', 'D.06', 'D.07',
                   'D.08', 'D.09', 'D.10', 'D.11', 'D.12', 'D.13', 'D.14',
                   'D.15', 'D.16', 'D.17', 'D.18', 'D.19', 'D.20', 'D.99']
    route_e_srs = ['E.01', 'E.02', 'E.03', 'E.04', 'E.05', 'E.91', 'E.99']
    route_f_srs = ['F.01', 'F.02', 'F.99']

    rp_srs = []
    # Get a list of nodes for all specified SRS's
    if 'D' in rp_id_seq:
        rp_srs = rp_srs + route_d_srs
    if 'E' in rp_id_seq:
        rp_srs = rp_srs + route_e_srs
    if 'F' in rp_id_seq:
        rp_srs = rp_srs + route_f_srs

    new_dict = {}
    for srs_id in rp_srs:
        nodes_dict = build_dict(list_of_dicts(srs_id))  # A dictionary
        for (key, val) in nodes_dict.items():
            if pd.isnull(nodes_dict[key]['Connecting SRS']):
                del nodes_dict[key]['Connecting SRS']
            else:
                nodes_dict[key]['SRS'] = {nodes_dict[key]['SRS'], nodes_dict[key]['Connecting SRS']}
                del nodes_dict[key]['Connecting SRS']
        new_dict = merge_dicts(new_dict, nodes_dict)
    return new_dict


# Get all edges on the network of the Anglia
def edges_of_route(direct=False):
    """
    :param direct: If True, return all the edges for a Directed graph
                   If False, return all the edges for an Undirected graph
    :return: All the edges for an Undirected/Directed Anglia network
    """
    # Adjacency 'matrix' (DataFrame)
    adj_mat = pd.read_excel(cdd('Network\\Routes\\Anglia', 'Anglia.xlsx'), sheetname='AdjacencyMatrix')
    # row names of the adjacency 'matrix'
    # row = adj_mat.index.tolist()
    # column names in a list
    col = adj_mat.columns.values.tolist()

    # Construct pairs of nodes. Each pair refers to an edge
    edges = []
    edge_temp = []
    node1 = []
    node2 = []
    node2_temp = []
    for node1 in col:
        node2 = adj_mat[adj_mat[node1] == 1].index.tolist()
        for node2_temp in node2:
            edge_temp = [node1, node2_temp]
            edges.append(edge_temp)
    del edge_temp, node1, node2, node2_temp
    if direct is True:
        return edges
    elif direct is False:
        return unique_lst(edges)
    else:
        print('InputErrors: input of "direct" must be a Boolean variable.')


# Get all edges of specified SRS's on the network of the Anglia
def edges_of_srs_seq(srs_id_seq, direct=False):
    """
    :param srs_id_seq: A sequence of SRS ID's
    :param direct: If True, return all the edges for a Directed graph
                   If False, return all the edges for an Undirected graph
    :return: All the edges for specified SRS's of an Undirected/Directed
             Anglia network
    """
    route_edges = edges_of_route(direct)

    def sub_edges(nodes_set):
        # nodes_set includes all nodes for the specified SRS
        # e.g. nodes("D.01")
        edges_subset = []
        for edge in route_edges:
            if all(node in nodes_set for node in edge):
                edges_subset.append(edge)
        return edges_subset

    if isinstance(srs_id_seq, str) and len(srs_id_seq) == 4:
        nodes_seq = nodes_of_srs(srs_id_seq)
    else:
        nodes_seq = [n for srs_id in srs_id_seq for n in nodes_of_srs_seq(srs_id)]

    if direct:
        return sub_edges(nodes_seq)
    elif not direct:
        return unique_lst(sub_edges(nodes_seq))
    else:
        print('InputErrors: input of "direct" must be a Boolean variable.')


def edges_of_route_plan(rp_id_seq, direct=False):
    route_d_srs = ['D.01', 'D.02', 'D.03', 'D.04', 'D.05', 'D.06', 'D.07',
                   'D.08', 'D.09', 'D.10', 'D.11', 'D.12', 'D.13', 'D.14',
                   'D.15', 'D.16', 'D.17', 'D.18', 'D.19', 'D.20', 'D.99']
    route_e_srs = ['E.01', 'E.02', 'E.03', 'E.04', 'E.05', 'E.91', 'E.99']
    route_f_srs = ['F.01', 'F.02', 'F.99']

    rp_srs = []
    # Get a list of nodes for all specified SRS's
    if 'D' in rp_id_seq:
        rp_srs = rp_srs + route_d_srs
    if 'E' in rp_id_seq:
        rp_srs = rp_srs + route_e_srs
    if 'F' in rp_id_seq:
        rp_srs = rp_srs + route_f_srs

    return edges_of_srs_seq(rp_srs, direct=direct)


# ------------------------------------------------------------------------------
# import networkx as nx
#
# def create(srs_id_seq, direct=False):
#     """
#     :param srs_id_seq: One or a sequence of SRS ID's
#     :param direct:
#     :return:
#     """
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
#     # Draw a graph for the network
#     return nx.draw(g)
