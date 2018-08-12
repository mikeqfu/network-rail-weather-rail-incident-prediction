"""
Data source: http://www.railwaycodes.org.uk

CRS, NLC, TIPLOC and STANOX Codes (Reference: http://www.railwaycodes.org.uk/crs/CRS0.shtm)

This links to a four-way listing of railway codes:

    - Computer reservation system (CRS) codes
        * [replaced by national reservation system (NRS) codes from late 2004, but the codes are the same]
    - National location codes (NLC)
    - Timing point locations (TIPLOC)
    - Station number names (STANME)
    - Station numbers (STANOX)
"""

import collections
import itertools
import os
import re
import string
import urllib.parse

import bs4
import dateutil.parser
import more_itertools
import pandas as pd
import pandas.errors
import requests

from converters import miles_chains_to_mileage
from utils import is_float, load_pickle, save_pickle, load_json, save_json

# ====================================================================================================================
""" Change directories """


# Change directory to "2017-04 RailwayCodes-Python-utils" and sub-directories
def cdd_rc_base(*directories):
    path = os.path.join(os.path.dirname(os.getcwd()), "2017-04 RailwayCodes-Python-utils")
    for directory in directories:
        path = os.path.join(path, directory)
    return path


# Change directory to "2017-04 RailwayCodes-Python-utils\\dat" and sub-directories
def cdd_rc_dat(*directories):
    path = cdd_rc_base('dat')
    for directory in directories:
        path = os.path.join(path, directory)
    return path


# ====================================================================================================================
""" General parser functions """


# Parse date string
def parse_date(str_date, as_date_type=False):
    """
    :param str_date: [str]
    :param as_date_type: [bool]
    :return: the date formatted as requested
    """
    parsed_date = dateutil.parser.parse(str_date, fuzzy=True)
    # Or, parsed_date = datetime.strptime(last_update_date[12:], '%d %B %Y')
    parsed_date = parsed_date.date() if as_date_type else str(parsed_date.date())
    return parsed_date


# Show last update date
def get_last_updated_date(url, parsed=True, date_type=False):
    """
    :param url: [str] URL link of a requested web page
    :param parsed: [bool] indicator of whether to reformat the date
    :param date_type: [bool] 
    :return:[str] date of when the specified web page was last updated
    """
    # Request to get connected to the given url
    source = requests.get(url)
    web_page_text = source.text
    # Parse the text scraped from the requested web page
    # (Optional parsers: 'lxml', 'html5lib' and 'html.parser')
    parsed_text = bs4.BeautifulSoup(web_page_text, 'lxml')
    # Find 'Last update date'
    update_tag = parsed_text.find('p', {'class': 'update'})
    if update_tag is not None:
        last_update_date = update_tag.text
        # Decide whether to convert the date's format
        if parsed:
            # Convert the date to "yyyy-mm-dd" format
            last_update_date = parse_date(last_update_date, date_type)
    else:
        last_update_date = None  # print('Information not available.')
    return last_update_date


# Get a list of parsed HTML tr's
def parse_tr(header, trs):
    """
    :param header: [list] list of column names of a requested table
    :param trs: [list] contents under tr tags of the web page
    :return: [list] list of lists each comprising a row of the requested table

    Get a list of parsed contents of tr-tag's, each of which corresponds to a piece of record
    *This is a key function to drive its following functions
    Reference: stackoverflow.com/questions/28763891/what-should-i-do-when-tr-has-rowspan

    """
    tbl_lst = []
    for row in trs:
        data = []
        for dat in row.find_all('td'):
            txt = dat.get_text()
            if '\n' in txt:
                t = txt.split('\n')
                txt = '%s (%s)' % (t[0], t[1:]) if '(' not in txt and ')' not in txt else '%s %s' % (t[0], t[1:])
                data.append(txt)
            else:
                data.append(txt)
        tbl_lst.append(data)

    row_spanned = []
    for no, tr in enumerate(trs):
        for td_no, rho in enumerate(tr.find_all('td')):
            # print(data.has_attr("rowspan"))
            if rho.has_attr('rowspan'):
                row_spanned.append((no, int(rho['rowspan']), td_no, rho.text))

    if row_spanned:
        d = collections.defaultdict(list)
        for k, *v in row_spanned:
            d[k].append(v)
        row_spanned = list(d.items())

        for x in row_spanned:
            i = x[0]
            to_repeat = x[1]
            for y in to_repeat:
                for j in range(1, y[0]):
                    tbl_lst[i + j].insert(y[1], y[2])

    # if row_spanned:
    #     for x in row_spanned:
    #         for j in range(1, x[2]):
    #             # Add value in next tr
    #             idx = x[0] + j
    #             # assert isinstance(idx, int)
    #             if x[1] >= len(tbl_lst[idx]):
    #                 tbl_lst[idx].insert(x[1], x[3])
    #             elif x[3] in tbl_lst[x[0]]:
    #                 tbl_lst[idx].insert(tbl_lst[x[0]].index(x[3]), x[3])
    #             else:
    #                 tbl_lst[idx].insert(x[1] + 1, x[3])

    for k in range(len(tbl_lst)):
        n = len(header) - len(tbl_lst[k])
        if n > 0:
            tbl_lst[k].extend(['\xa0'] * n)
        elif n < 0 and tbl_lst[k][2] == '\xa0':
            del tbl_lst[k][2]

    return tbl_lst


# Parse the acquired list to make it be ready for creating the DataFrame
def parse_table(source, parser='lxml'):
    """
    :param source: response object to connecting a URL to request a table
    :param parser: [str] Optional parsers: 'lxml', 'html5lib', 'html.parser'
    :return [tuple] ([list] of lists each comprising a row of the requested table - (see also parse_trs())
                     [list] of column names of the requested table)
    """
    # (If source.status_code == 200, the requested URL is available.)
    # Get plain text from the source URL
    web_page_text = source.text
    # Parse the text
    parsed_text = bs4.BeautifulSoup(web_page_text, parser)
    # Get all data under the HTML label 'tr'
    table_temp = parsed_text.find_all('tr')
    # Get a list of column names for output DataFrame
    headers = table_temp[0]
    header = [header.text for header in headers.find_all('th')]
    # Get a list of lists, each of which corresponds to a piece of record
    trs = table_temp[1:]
    # Return a list of parsed tr's, each of which corresponds to one df row
    return parse_tr(header, trs), header


# Parse location note
def parse_loc_note(x):
    # Data
    # d = re.search('[\w ,]+(?=[ \n]\[)', x)
    # if d is not None:
    #     dat = d.group()
    # else:
    d = re.search('.*(?= \[[\"\']\()', x)
    if d is not None:
        dat = d.group()
    elif ' [unknown feature, labelled "do not use"]' in x:
        dat = re.search('\w+(?= \[unknown feature, )', x).group()
    else:
        m_pattern = re.compile(
            '[Oo]riginally |[Ff]ormerly |[Ll]ater |[Pp]resumed | \(was | \(in | \(at | \(also known as |'
            ' \(second code set |\?|\"|\n')
        # dat = re.search('["\w ,]+(?= [[(?\'])|["\w ,]+', x).group(0) if re.search(m_pattern, x) else x
        dat = ' '.join(x.replace(x[x.find('('):x.find(')') + 1], '').split()) if re.search(m_pattern, x) else x
    # Note
    n = re.search('(?<=\[[\'\"]\().*(?=\)[\'\"]\])', x)
    if n is None:
        n = re.search('(?<=[\n ][\[\'(])[\w ,\'\"/?]+', x)
    elif n.group() == "'" or n.group() == '"':
        n = re.search(r'(?<=[\[(])[\w ,?]+(?=[])])', x)
    note = n.group() if n is not None else ''
    if 'STANOX ' in dat and 'STANOX ' in x and note == '':
        dat = x[0:x.find('STANOX')].strip()
        note = x[x.find('STANOX'):]
    return dat, note


# ====================================================================================================================
""" Engineer's Line References and mileages"""


# Change directory to "Line data\\ELRs and mileages"
def cdd_elr_mileage(*directories):
    path = cdd_rc_dat("Line data", "ELRs and mileages")
    for directory in directories:
        path = os.path.join(path, directory)
    return path


# Scrape Engineer's Line References (ELRs)
def scrape_elrs(keyword, update=False):
    """
    :param keyword: [str] usually an initial letter of ELR, e.g. 'a', 'b'
    :param update: [bool] indicate whether to re-scrape the data from online
    :return: [dict] {'ELRs_mileages_keyword': [DataFrame] data of ELRs whose names start with the given 'keyword',
                                                including ELR names, line name, mileages, datum and some notes,
                     'Last_updated_date_keyword': [str] date of when the data was last updated}
    """
    path_to_file = cdd_elr_mileage("A-Z", keyword.title() + ".pickle")
    if os.path.isfile(path_to_file) and not update:
        elrs = load_pickle(path_to_file)
    else:
        # Specify the requested URL
        url = 'http://www.railwaycodes.org.uk/elrs/ELR{}.shtm'.format(keyword.lower())
        last_updated_date = get_last_updated_date(url)
        try:
            source = requests.get(url)  # Request to get connected to the url
            records, header = parse_table(source, parser='lxml')
            # Create a DataFrame of the requested table
            data = pd.DataFrame([[x.replace('=', 'See').strip('\xa0') for x in i] for i in records], columns=header)
        except IndexError:  # If the requested URL is not available:
            data = None

        # Return a dictionary containing both the DataFrame and its last updated date
        elr_keys = [s + keyword.title() for s in ('ELRs_mileages_', 'Last_updated_date_')]
        elrs = dict(zip(elr_keys, [data, last_updated_date]))
        save_pickle(elrs, path_to_file)

    return elrs


# Get all ELRs and mileages
def get_elrs(update=False):
    """
    :param update: [bool]
    :return [dict] {'ELRs_mileages': [DataFrame] data of (almost all) ELRs whose names start with the given 'keyword',
                                        including ELR names, line name, mileages, datum and some notes,
                    'Last_updated_date': [str] date of when the data was last updated}
    """
    path_to_file = cdd_elr_mileage("ELRs.pickle")
    if os.path.isfile(path_to_file) and not update:
        elrs = load_pickle(path_to_file)
    else:
        data = [scrape_elrs(i, update) for i in string.ascii_lowercase]
        # Select DataFrames only
        elrs_data = (item['ELRs_mileages_{}'.format(x)] for item, x in zip(data, string.ascii_uppercase))
        elrs_data_table = pd.concat(elrs_data, axis=0, ignore_index=True)

        # Get the latest updated date
        last_updated_dates = (item['Last_updated_date_{}'.format(x)] for item, x in zip(data, string.ascii_uppercase))
        last_updated_date = max(d for d in last_updated_dates if d is not None)

        elrs = {'ELRs_mileages': elrs_data_table, 'Last_updated_date': last_updated_date}

        save_pickle(elrs, path_to_file)

    return elrs


# Parse mileage column
def parse_mileage(mileage):
    """
    :param mileage:
    :return:
    """
    if mileage.dtype == pd.np.float64:
        temp_mileage = mileage
        mileage_note = [''] * len(temp_mileage)
    else:
        temp_mileage, mileage_note = [], []
        for m in mileage:
            if pd.isnull(m):
                mileage_note.append('Unknown')
                temp_mileage.append(m)
            elif m.startswith('(') and m.endswith(')'):
                temp_mileage.append(m[m.find('(') + 1:m.find(')')])
                mileage_note.append('Reference')
            elif m.startswith('~'):
                temp_mileage.append(m[1:])
                mileage_note.append('Approximate')
            else:
                if isinstance(m, str):
                    temp_mileage.append(m.strip(' '))
                else:
                    temp_mileage.append(m)
                mileage_note.append('')

    temp_mileage = [miles_chains_to_mileage(m) for m in temp_mileage]

    return pd.DataFrame({'Mileage': temp_mileage, 'Mileage_Note': mileage_note})


# Separate node and connection
# noinspection PyTypeChecker
def parse_node_and_connection(node):
    """
    :param node:
    :return:
    """

    def preprocess_node(node_x):
        if re.match('\w+.*( with)?( \(\d+\.\d+\))?(/| and )\w+.*( with)?[ A-Z0-9]?( \(\d+\.\d+\))?', node_x):
            init_conn_info = [match.group(0) for match in re.finditer(' with \w+( \(\d+\.\d+\))?', node_x)]
            if '/' in node_x:
                node_info = [y.replace(conn_inf, '') for y, conn_inf in zip(node_x.split('/'), init_conn_info)]
            else:
                node_info = [y.replace(conn_inf, '') for y, conn_inf in zip(node_x.split(' and '), init_conn_info)]
            conn_info = [conn_inf.replace(' with ', '') for conn_inf in init_conn_info]
            return '/'.join(node_info) + ' with ' + ' and '.join(conn_info)
        else:
            return node_x

    parsed_node_info = [preprocess_node(n) for n in node]

    temp_node = pd.DataFrame([n.replace(' with Freightliner terminal', ' & Freightliner Terminal').
                             replace(' with curve to', ' with').
                             replace(' (later divergence with MIN)', '').
                             replace(' (0.37 long)', '').split(' with ')
                              for n in parsed_node_info], columns=['Node', 'Connection'])
    conn_node_list = []
    x = 2  # x-th occurrence
    for c in temp_node.Connection:
        if c is not None:
            cnode = c.split(' and ')
            if len(cnode) > 2:
                cnode = [' and '.join(cnode[:x]), ' and '.join(cnode[x:])]
        else:
            cnode = [c]
        conn_node_list.append(cnode)

    if all(len(c) == 1 for c in conn_node_list):
        # noinspection PyTypeChecker
        conn_node = pd.DataFrame([c + [None] for c in conn_node_list], columns=['Connection1', 'Connection2'])
    else:
        assert isinstance(conn_node_list, list)
        for i in [conn_node_list.index(c) for c in conn_node_list if len(c) > 1]:
            conn_node_list[i] = [v for lst in [x.rstrip(',').lstrip('later ').split(' and ')
                                               for x in conn_node_list[i]] for v in lst]
            conn_node_list[i] = [v for lst in [x.split(', ') for x in conn_node_list[i]] for v in lst]

        no_conn = max(len(c) for c in conn_node_list)
        conn_node_list = [c + [None] * (no_conn - len(c)) for c in conn_node_list]
        conn_node = pd.DataFrame(conn_node_list, columns=['Connection' + str(n + 1) for n in range(no_conn)])

    return temp_node.loc[:, ['Node']].join(conn_node)


# Parse data for both mileage node and connection
def parse_mileage_node_and_connection(dat):
    """
    :param dat:
    :return:
    """
    mileage, node = dat.iloc[:, 0], dat.iloc[:, 1]
    parsed_mileage = parse_mileage(mileage)
    parsed_node_and_connection = parse_node_and_connection(node)
    parsed_dat = parsed_mileage.join(parsed_node_and_connection)
    return parsed_dat


# Parse mileage file for the given ELR
def parse_mileage_file(mileage_file, elr):
    """
    :param mileage_file:
    :param elr:
    :return:
    """
    dat = mileage_file[elr]
    if isinstance(dat, dict) and len(dat) > 1:
        dat = {h: parse_mileage_node_and_connection(d) for h, d in dat.items()}
    else:  # isinstance(dat, pd.DataFrame)
        dat = parse_mileage_node_and_connection(dat)
    mileage_file[elr] = dat
    return mileage_file


# Read (from online) the mileage file for the given ELR
def scrape_mileage_file(elr):
    """
    :param elr:
    :return:

    Note:
        - In some cases, mileages are unknown hence left blank, e.g. ANI2, Orton Junction with ROB (~3.05)
        - Mileages in parentheses are not on that ELR, but are included for reference, e.g. ANL, (8.67) NORTHOLT [
        London Underground]
        - As with the main ELR list, mileages preceded by a tilde (~) are approximate.

    """
    try:
        url = 'http://www.railwaycodes.org.uk/elrs'
        # The URL of the mileage file for the ELR
        mileage_file_url = '/'.join([url, '_mileages', elr[0], elr + '.txt'])

        # Request to get connected to the given url
        try:
            mileages = pd.read_table(mileage_file_url)
        except pandas.errors.ParserError:
            temp = pd.read_csv(mileage_file_url)
            header = temp.columns[0].split('\t')
            data = [v.split('\t', 1) for val in temp.values for v in val]
            data = [[x.replace('\t', '') for x in dat] for dat in data]
            mileages = pd.DataFrame(data, columns=header)

        line = {'Line': mileages.columns[1]}

        check_idx = mileages[elr].map(is_float)
        to_check = mileages[~check_idx]
        if to_check.empty:
            dat = {elr: mileages[check_idx]}
            note = {'Note': None}
        else:
            if len(to_check) == 1:
                note = {'Note': to_check[elr].iloc[0]}
                dat = {elr: mileages[check_idx]}
                dat[elr].index = range(len(dat[elr]))
            else:
                idx_vals = to_check.index.get_values()
                diff = list(pd.np.diff(idx_vals)) + [len(mileages) - pd.np.diff(idx_vals)[-1]]
                sliced_dat = {mileages[elr][i]: mileages[i + 1:i + d] for i, d in zip(idx_vals, diff)}
                if len(idx_vals) == 2:
                    note = {'Note': None}
                else:
                    note = {'Note': k for k, v in sliced_dat.items() if v.empty}
                    del sliced_dat[note['Note']]
                for _, dat in sliced_dat.items():
                    dat.index = range(len(dat))
                dat = {elr: sliced_dat}

        mileage_file = dict(pair for d in [dat, line, note] for pair in d.items())
        mileage_file = parse_mileage_file(mileage_file, elr)

        path_to_file = cdd_elr_mileage("mileage_files", elr[0].title(), elr + ".pickle")
        save_pickle(mileage_file, path_to_file)

    except Exception as e:
        print("Scraping the mileage file for '{}' ... failed due to '{}'.".format(elr, e))
        mileage_file = None

    return mileage_file


# Get the mileage file for the given ELR (firstly try to load the local data file if available)
def get_mileage_file(elr, update=False):
    """
    :param elr: [str]
    :param update: [bool] indicate whether to re-scrape the data from online
    :return: [dict] {elr: [DataFrame] mileage file data,
                    'Line': [str] line name,
                    'Note': [str] additional information/notes, or None}
    """
    path_to_file = cdd_elr_mileage("mileage_files", elr[0].title(), elr + ".pickle")

    file_exists = os.path.isfile(path_to_file)
    mileage_file = load_pickle(path_to_file) if file_exists and not update else scrape_mileage_file(elr)

    return mileage_file


# Get to end and start mileages for StartELR and EndELR, respectively, for the connection point
def get_conn_end_start_mileages(start_elr, end_elr, update=False):
    """
    :param start_elr:
    :param end_elr:
    :param update:
    :return:
    """
    start_elr_mileage_file = get_mileage_file(start_elr, update)[start_elr]
    if isinstance(start_elr_mileage_file, dict):
        for k in start_elr_mileage_file.keys():
            if re.match('^Usual|^New', k):
                start_elr_mileage_file = start_elr_mileage_file[k]

    start_conn_cols = [c for c in start_elr_mileage_file.columns if re.match('^Connection', c)]

    start_conn_mileage, end_conn_mileage = None, None

    for start_conn_col in start_conn_cols:
        start_conn = start_elr_mileage_file[start_conn_col].dropna()
        for i in start_conn.index:
            if end_elr in start_conn[i]:
                start_conn_mileage = start_elr_mileage_file.Mileage.loc[i]
                if re.match('\w+(?= \(\d+\.\d+\))', start_conn[i]):
                    end_conn_mileage = miles_chains_to_mileage(re.search('(?<=\w \()\d+\.\d+', start_conn[i]).group(0))
                    break
                elif end_elr == start_conn[i]:

                    end_elr_mileage_file = get_mileage_file(end_elr, update)[end_elr]
                    if isinstance(end_elr_mileage_file, dict):
                        for k in end_elr_mileage_file.keys():
                            if re.match('^Usual|^New', k):
                                end_elr_mileage_file = end_elr_mileage_file[k]

                    end_conn_cols = [c for c in end_elr_mileage_file.columns if re.match('^Connection', c)]
                    for end_conn_col in end_conn_cols:
                        end_conn = end_elr_mileage_file[end_conn_col].dropna()
                        for j in end_conn.index:
                            if start_elr in end_conn[j]:
                                end_conn_mileage = end_elr_mileage_file.Mileage.loc[j]
                                break
                        if start_conn_mileage is not None and end_conn_mileage is not None:
                            break
                    if start_conn_mileage is not None and end_conn_mileage is not None:
                        break

            else:
                try:
                    link_elr = re.search('\w+(?= \(\d+\.\d+\))', start_conn[i]).group(0)
                except AttributeError:
                    link_elr = start_conn[i]

                if re.match('[A-Z]{3}(0-9)?$', link_elr):
                    try:
                        link_elr_mileage_file = get_mileage_file(link_elr, update)[link_elr]

                        if isinstance(link_elr_mileage_file, dict):
                            for k in link_elr_mileage_file.keys():
                                if re.match('^Usual|^New', k):
                                    link_elr_mileage_file = link_elr_mileage_file[k]

                        link_conn_cols = [c for c in link_elr_mileage_file.columns if re.match('^Connection', c)]
                        for link_conn_col in link_conn_cols:
                            link_conn = link_elr_mileage_file[link_conn_col].dropna()
                            for l in link_conn.index:
                                if start_elr in link_conn[l]:
                                    start_conn_mileage = link_elr_mileage_file.Mileage.loc[l]
                                    break
                            for l in link_conn.index:
                                if end_elr in link_conn[l]:
                                    if re.match('\w+(?= \(\d+\.\d+\))', link_conn[l]):
                                        end_conn_mileage = miles_chains_to_mileage(
                                            re.search('(?<=\w \()\d+\.\d+', link_conn[l]).group(0))
                                    elif end_elr == link_conn[l]:
                                        end_conn_mileage = link_elr_mileage_file.Mileage.loc[l]
                                    break
                            if start_conn_mileage is not None and end_conn_mileage is not None:
                                break
                    except (TypeError, AttributeError):
                        pass
                else:
                    pass

            if start_conn_mileage is not None and end_conn_mileage is not None:
                break
        if start_conn_mileage is not None and end_conn_mileage is not None:
            break

    if start_conn_mileage is None or end_conn_mileage is None:
        start_conn_mileage, end_conn_mileage = None, None

    return start_conn_mileage, end_conn_mileage


# ====================================================================================================================
""" Locations and CRS, NLC, TIPLOC, STANME and STANOX codes """


# Change directory to "Line data\\CRS, NLC, TIPLOC and STANOX codes"
def cdd_loc_codes(*directories):
    path = cdd_rc_dat("Line data", "CRS, NLC, TIPLOC and STANOX codes")
    for directory in directories:
        path = os.path.join(path, directory)
    return path


# Location name modifications
def create_location_name_mod_dict():
    location_name_mod_dict = {
        'Location': {re.compile(' And | \+ '): ' & ',
                     re.compile('-By-'): '-by-',
                     re.compile('-In-'): '-in-',
                     re.compile('-En-Le-'): '-en-le-',
                     re.compile('-La-'): '-la-',
                     re.compile('-Le-'): '-le-',
                     re.compile('-On-'): '-on-',
                     re.compile('-The-'): '-the-',
                     re.compile(' Of '): ' of ',
                     re.compile('-Super-'): '-super-',
                     re.compile('-Upon-'): '-upon-',
                     re.compile('-Under-'): '-under-',
                     re.compile('-Y-'): '-y-'}}
    return location_name_mod_dict


# Addition note page
def parse_additional_note_page(url, parser='lxml'):
    source = requests.get(url)
    web_page_text = bs4.BeautifulSoup(source.text, parser).find_all(['p', 'pre'])
    parsed_text = [x.text for x in web_page_text if isinstance(x.next_element, str)]
    parsed_texts = []
    for x in parsed_text:
        if '\n' in x:
            text = re.sub('\t+', ',', x).replace('\t', ' ').replace('\xa0', '').split('\n')
        else:
            text = x.replace('\t', ' ').replace('\xa0', '')
        if isinstance(text, list):
            text = [t.split(',') for t in text if t != '']
            parsed_texts.append(pd.DataFrame(text, columns=['Location', 'CRS', 'CRS_alt1', 'CRS_alt2']).fillna(''))
        else:
            to_remove = ['click the link', 'click your browser', 'Thank you', 'shown below']
            if text != '' and not any(t in text for t in to_remove):
                parsed_texts.append(text)
    return parsed_texts


# Locations and CRS, NLC, TIPLOC, STANME and STANOX codes
def scrape_location_codes(keyword, update=False):
    """
    :param keyword: [str] initial letter of station/junction name or certain word for specifying URL
    :param update: [bool]
    :return [tuple] ([DataFrame] CRS, NLC, TIPLOC and STANOX data of (almost) all stations/junctions,
                     [str]} date of when the data was last updated)
    """
    path_to_file = cdd_loc_codes("A-Z", keyword.title() + ".pickle")
    if os.path.isfile(path_to_file) and not update:
        location_codes = load_pickle(path_to_file)
    else:
        # Specify the requested URL
        url = 'http://www.railwaycodes.org.uk/CRS/CRS{}.shtm'.format(keyword)
        last_updated_date = get_last_updated_date(url)
        # Request to get connected to the URL
        try:
            source = requests.get(url)
            tbl_lst, header = parse_table(source, parser='lxml')

            # Get a raw DataFrame
            reps = {'\b-\b': '', '\xa0\xa0': ' ', '&half;': ' and 1/2'}
            pattern = re.compile("|".join(reps.keys()))
            tbl_lst = [[pattern.sub(lambda x: reps[x.group(0)], item) for item in record] for record in tbl_lst]
            data = pd.DataFrame(tbl_lst, columns=header)
            data = data.replace({'\xa0': ''})

            """ Extract additional information as note """

            data[['Location', 'Location_Note']] = data.Location.map(parse_loc_note).apply(pd.Series)

            # CRS, NLC, TIPLOC, STANME
            drop_pattern = re.compile('[Ff]ormerly|[Ss]ee[ also]|Also .[\w ,]+')
            idx = [data[data.CRS == x].index[0] for x in data.CRS if re.match(drop_pattern, x)]
            data.drop(labels=idx, axis=0, inplace=True)

            def extract_others_note(x):
                n = re.search('(?<=[\[(\'])[\w,? ]+(?=[)\]\'])', x)
                note = n.group() if n is not None else ''
                return note

            def strip_others_note(x):
                d = re.search('[\w ,]+(?= [\[(\'])', x)
                dat = d.group() if d is not None else x
                return dat

            other_codes_col = ['CRS', 'NLC', 'TIPLOC', 'STANME']
            other_notes_col = [x + '_Note' for x in other_codes_col]

            data[other_notes_col] = data[other_codes_col].applymap(extract_others_note)
            data[other_codes_col] = data[other_codes_col].applymap(strip_others_note)

            # STANOX
            def parse_stanox_note(x):
                if x == '-':
                    dat, note = '', ''
                else:
                    d = re.search('[\w *,]+(?= [\[(\'])', x)
                    dat = d.group() if d is not None else x
                    note = 'Pseudo STANOX' if '*' in dat else ''
                    n = re.search('(?<=[\[(\'])[\w, ]+.(?=[)\]\'])', x)
                    if n is not None:
                        note = '; '.join(x for x in [note, n.group()] if x != '')
                    if '(' not in note and note.endswith(')'):
                        note = note.rstrip(')')
                    dat = dat.rstrip('*') if '*' in dat else dat
                return dat, note

            if not data.empty:
                data[['STANOX', 'STANOX_Note']] = data.STANOX.map(parse_stanox_note).apply(pd.Series)
            else:  # It is likely that no data is available on the web page for the given 'key_word'
                data['STANOX_Note'] = data.STANOX

            if any('see note' in crs_note for crs_note in data.CRS_Note):
                loc_idx = [i for i, crs_note in enumerate(data.CRS_Note) if 'see note' in crs_note]
                web_page_text = bs4.BeautifulSoup(source.text, 'lxml')
                note_urls = [urllib.parse.urljoin(url, l['href'])
                             for l in web_page_text.find_all('a', href=True, text='note')]
                additional_notes = [parse_additional_note_page(note_url) for note_url in note_urls]
                additional_note = dict(zip(data.CRS.iloc[loc_idx], additional_notes))
            else:
                additional_note = None

            location_name_mod_dict = create_location_name_mod_dict()
            data = data.replace(location_name_mod_dict, regex=True)

            data.STANOX = data.STANOX.replace({'-': ''})

            data.index = range(len(data))  # Rearrange index

        except Exception as e:
            print("Scraping location data ... failed due to {}.".format(e))
            data = None
            additional_note = None

        location_codes_keys = [s + keyword.title() for s in ('Locations_', 'Last_updated_date_', 'Additional_note_')]
        location_codes = dict(zip(location_codes_keys, [data, last_updated_date, additional_note]))
        save_pickle(location_codes, path_to_file)

    return location_codes


# Scrape data for other systems
def scrape_other_systems(update=False):
    path_to_file = cdd_loc_codes("Other-systems-location-codes.pickle")
    if os.path.isfile(path_to_file) and not update:
        other_systems_codes = load_pickle(path_to_file)
    else:
        try:
            url = 'http://www.railwaycodes.org.uk/crs/CRS1.shtm'
            source = requests.get(url)
            web_page_text = bs4.BeautifulSoup(source.text, 'lxml')
            # Get system name
            system_names = [k.text for k in web_page_text.find_all('h3')]
            system_names = [n.replace('Tramlnk', 'Tramlink') if 'Tramlnk' in n else n for n in system_names]
            # Get column names for the other systems table
            headers = list(more_itertools.unique_everseen([h.text for h in web_page_text.find_all('th')]))
            # Parse table data for each system
            table_data = web_page_text.find_all('table', {'width': '1100px'})
            tables = [pd.DataFrame(parse_tr(headers, table.find_all('tr')), columns=headers) for table in table_data]
            codes = [tables[i] for i in range(len(tables)) if i % 2 != 0]
            # Create a dict
            other_systems_codes = dict(zip(system_names, codes))
        except Exception as e:
            print("Scraping location data for other systems ... failed due to '{}'.".format(e))
            other_systems_codes = None

    return other_systems_codes


# Get note pertaining to CRS
def get_additional_crs_note(update=False):
    path_to_file = cdd_loc_codes("additional-CRS-note.pickle")
    if os.path.isfile(path_to_file) and not update:
        additional_note = load_pickle(path_to_file)
    else:
        try:
            note_url = 'http://www.railwaycodes.org.uk/crs/CRS2.shtm'
            additional_note = parse_additional_note_page(note_url)
            save_pickle(additional_note, path_to_file)
        except Exception as e:
            print("Getting additional note for CRS ... failed due to '{}'.".format(e))
            additional_note = None
    return additional_note


# All Location, with CRS, NLC, TIPLOC, STANME and STANOX codes
def get_location_codes(update=False):
    path_to_file = cdd_loc_codes("CRS-NLC-TIPLOC-STANOX-codes.pickle")

    if os.path.isfile(path_to_file) and not update:
        location_codes = load_pickle(path_to_file)
    else:
        # Get every data table
        data = [scrape_location_codes(i, update) for i in string.ascii_lowercase]

        # Select DataFrames only
        location_codes_data = (item['Locations_{}'.format(x)] for item, x in zip(data, string.ascii_uppercase))
        location_codes_data_table = pd.concat(location_codes_data, axis=0, ignore_index=True)

        # Get the latest updated date
        last_updated_dates = (item['Last_updated_date_{}'.format(x)] for item, x in zip(data, string.ascii_uppercase))
        last_updated_date = max(d for d in last_updated_dates if d is not None)

        # Get additional note
        additional_note = get_additional_crs_note(update)

        # Get other systems codes
        other_systems_codes = scrape_other_systems(update)

        # Create a dict to include all information
        location_codes = {'Locations': location_codes_data_table,
                          'Latest_updated_date': last_updated_date,
                          'Additional_note': additional_note,
                          'Other_systems': other_systems_codes}

        save_pickle(location_codes, path_to_file)

    return location_codes


# Get a dict for location code data for the given keyword
def get_location_codes_dictionary(keyword, initial=None, drop_duplicates=True, main_key=None, update=False):
    """
    :param keyword: [str] 'CRS', 'NLC', 'TIPLOC', 'STANOX'
    :param initial: [str] or None: one of string.ascii_letters, or (default) None
    :param drop_duplicates: [bool] If drop_duplicates is False, loc_dict will take the last item to be the value
    :param main_key: [str] or None
    :param update: [bool]
    :return:
    """
    assert keyword in ['CRS', 'NLC', 'TIPLOC', 'STANOX', 'STANME']

    json_filename = keyword + ".json"
    path_to_json = cdd_loc_codes(json_filename)

    if os.path.isfile(path_to_json) and not update:
        location_codes_dictionary = load_json(path_to_json)
    else:
        if initial is not None and initial in string.ascii_letters:
            location_code = scrape_location_codes(initial)['Locations_' + initial.capitalize()]
        else:
            location_code = get_location_codes()['Locations']

        assert isinstance(location_code, pd.DataFrame)

        try:
            loc_code_original = location_code[['Location', keyword]]
            loc_code_original = loc_code_original[loc_code_original[keyword] != '']
            if drop_duplicates:
                if drop_duplicates is True:
                    loc_code = loc_code_original.drop_duplicates(subset=keyword, keep='first')
                else:
                    loc_code = loc_code_original
                loc_dict = loc_code.set_index(keyword).to_dict()
            else:  # drop_duplicates is False
                loc_code = loc_code_original.groupby(keyword).aggregate(list)
                loc_code.Location = loc_code.Location.map(lambda x: x[0] if len(x) == 1 else x)
                loc_dict = loc_code.to_dict()

            if main_key is not None:
                loc_dict[main_key] = loc_dict.pop('Location')
                location_codes_dictionary = loc_dict
            else:
                location_codes_dictionary = loc_dict['Location']

            save_json(location_codes_dictionary, path_to_json)

        except Exception as e:
            print("Failed to get location code reference dictionary. This is due to {}.".format(e))
            location_codes_dictionary = None

    return location_codes_dictionary


# Get a dict (as a dataframe) for location code data for the grouped keywords
def get_location_codes_dictionary_v2(keywords, initial=None, as_dict=False, main_key=None, update=False):
    """
    :param keywords: [list] e.g. ['CRS', 'NLC', 'TIPLOC', 'STANOX', 'STANME']
    :param initial: [str] one of string.ascii_letters
    :param as_dict:
    :param main_key: [str] or None
    :param update: [bool]
    :return:
    """
    assert isinstance(keywords, list) and all(x in ['CRS', 'NLC', 'TIPLOC', 'STANOX', 'STANME'] for x in keywords)

    filename = "-".join(keywords) + "-v2" + (".json" if as_dict else ".pickle")
    path_to_file = cdd_loc_codes(filename)

    if os.path.isfile(path_to_file) and not update:
        location_code_ref_dict = load_json(path_to_file) if as_dict else load_pickle(path_to_file)
    else:

        if initial is not None and initial in string.ascii_letters:
            location_code = scrape_location_codes(initial)['Locations_' + initial.capitalize()]
        else:
            location_code = get_location_codes()['Locations']

        # Deep cleansing location_code
        try:
            loc_code = location_code[['Location'] + keywords]
            loc_code = loc_code.query(' | '.join(["{} != ''".format(k) for k in keywords]))

            loc_code_unique = loc_code.drop_duplicates(subset=keywords, keep=False)
            loc_code_unique.set_index(keywords, inplace=True)

            duplicated_temp_1 = loc_code[loc_code.duplicated(subset=['Location'] + keywords, keep=False)]
            duplicated_temp_2 = loc_code[loc_code.duplicated(subset=keywords, keep=False)]
            duplicated_1 = duplicated_temp_2[duplicated_temp_1.eq(duplicated_temp_2)].dropna().drop_duplicates()
            duplicated_2 = duplicated_temp_2[~duplicated_temp_1.eq(duplicated_temp_2)].dropna()
            duplicated = pd.concat([duplicated_1, duplicated_2], axis=0)
            loc_code_duplicated = duplicated.groupby(keywords).agg(list)
            loc_code_duplicated.Location = loc_code_duplicated.Location.map(lambda x: x[0] if len(set(x)) == 1 else x)

            loc_code_ref = pd.concat([loc_code_unique, loc_code_duplicated], axis=0)

            if as_dict:
                loc_code_ref_dict = loc_code_ref.to_dict()
                if main_key is not None:
                    loc_code_ref_dict[main_key] = loc_code_ref_dict.pop('Location')
                    location_code_ref_dict = loc_code_ref_dict
                else:
                    location_code_ref_dict = loc_code_ref_dict['Location']
                save_json(location_code_ref_dict, path_to_file)
            else:
                location_code_ref_dict = loc_code_ref
                save_pickle(location_code_ref_dict, path_to_file)

        except Exception as e:
            print("Failed to get multiple location code indexed reference. This is due to {}.".format(e))
            location_code_ref_dict = None

    return location_code_ref_dict


# ====================================================================================================================
""" ELRs, mileages, operators, grid references """


def cdd_stn_loc(*directories):
    path = cdd_rc_dat("Other assets", "Stations", "ELRs, mileages, operators, grid references")
    for directory in directories:
        path = os.path.join(path, directory)
    return path


def parse_current_operator(x):
    contents = re.split('\\r| \[\'|\\\\r| {2}\'\]|\', \'|\\n', x.lstrip(' [\'').rstrip('  \']').lstrip('\n').strip())
    contents = [x for x in contents if x != '']
    operators = []
    for y in contents:
        # Operators names
        operator_name = re.search('.*(?= \(from \d+ \w+ \d+(.*)?\))', y)
        operator_name = operator_name.group() if operator_name is not None else ''
        # Start dates
        start_date = re.search('(?<= \(from )\d+ \w+ \d+( to \d+ \w+ \d+(.*))?(?=\))', y)
        start_date = start_date.group() if start_date is not None else ''
        # Form a tuple
        operators.append((operator_name, start_date))
    return operators


# Railway station data by keywords
def scrape_station_locations(keyword, update=False):
    """
    :param keyword: [str] station data (including the station name, ELR, mileage, status, owner, operator, degrees of
                        longitude and latitude, and grid reference) for stations whose name start with
    :param update: [bool]
    :return [dict] {keyword: [DataFrame] railway station data,
                    'Last_updated_date': [str] date of when the data was last updated}
    """
    path_to_file = cdd_stn_loc("A-Z", keyword.title() + ".pickle")
    if os.path.isfile(path_to_file) and not update:
        station_locations = load_pickle(path_to_file)
    else:
        url = 'http://www.railwaycodes.org.uk/stations/station{}.shtm'.format(keyword.lower())
        last_updated_date = get_last_updated_date(url)
        try:
            source = requests.get(url)  # Request to get connected to the url
            records, header = parse_table(source, parser='lxml')
            # Create a DataFrame of the requested table
            dat = [[x.replace('=', 'See').strip('\xa0') for x in i] for i in records]
            col = [h.replace('\r\n', ' ').replace('\r', ' ') for h in header]
            station_locations_data = pd.DataFrame(dat, columns=col)

            station_locations_data[['Degrees Longitude', 'Degrees Latitude']] = \
                station_locations_data[['Degrees Longitude', 'Degrees Latitude']].applymap(
                    lambda x: pd.np.nan if x == '' else float(x))

            station_locations_data[['Station', 'Station_Note']] = \
                station_locations_data.Station.map(parse_loc_note).apply(pd.Series)

            # Operator
            temp = list(station_locations_data.Operator.map(parse_current_operator))
            length = len(max(temp, key=len))
            col_names_current = ['Operator', 'Date']
            prev_no = list(itertools.chain.from_iterable(itertools.repeat(x, 2) for x in list(range(1, length))))
            col_names = zip(col_names_current * (length - 1), prev_no)
            col_names = col_names_current + ['_'.join(['Prev', x, str(d)]) for x, d in col_names]

            for i in range(len(temp)):
                if len(temp[i]) < length:
                    temp[i] += [(None, None)] * (length - len(temp[i]))

            temp = pd.DataFrame(temp)
            operators = [pd.DataFrame(temp)[col].apply(pd.Series) for col in temp.columns]
            operators = pd.concat(operators, axis=1)
            operators.columns = col_names

            station_locations_data.drop('Operator', axis=1, inplace=True)
            station_locations_data = station_locations_data.join(operators)

            station_locations = {'Station_{}'.format(keyword.upper()): station_locations_data,
                                 'Last_updated_date': last_updated_date}

            save_pickle(station_locations, path_to_file)

        except Exception as e:
            if keyword not in ['x', 'X', 'z', 'Z']:
                print("Failed to scrape station locations for the keyword \"{}\" due to '{}'".format(keyword, e))
            station_locations = {'Station_{}'.format(keyword.upper()): pd.DataFrame(),
                                 'Last_updated_date': last_updated_date}

    return station_locations


# Get all railway station data
def get_station_locations(update=False):
    """
    :param update: [bool]
    :return [dict] {keyword: [DataFrame] station data, including the station name, ELR, mileage, status, owner,
                                operator, degrees of longitude and latitude, and grid reference,
                    'Last_updated_date': [str] date of when the data was last updated}
    """
    path_to_file = cdd_stn_loc("ELRs-mileages-operators-grid.pickle")
    if os.path.isfile(path_to_file) and not update:
        station_locations = load_pickle(path_to_file)
    else:
        try:
            stn_locs, last_updated_dates = [], []
            for k in string.ascii_lowercase:
                stn_loc, updated_date = scrape_station_locations(k, update).values()
                if stn_loc is not None:
                    stn_locs.append(stn_loc)
                if updated_date is not None:
                    last_updated_dates.append(updated_date)

            stn_dat = pd.concat(stn_locs, ignore_index=True, sort=False)
            station_locations = {'Station': stn_dat, 'Last_updated_date': max(last_updated_dates)}

            save_pickle(station_locations, path_to_file)

        except Exception as e:
            print("Failed to get \"station data\" due to \"{}\".".format(e))
            station_locations = {'Station': pd.DataFrame(), 'Last_updated_date': None}

    return station_locations
