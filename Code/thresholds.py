"""
Weather-Thresholds_9306121.html

Description:

"The following table defines weather thresholds used to determine the classification of weather as Normal, Alert,
Adverse or Extreme. Note that the 'Alert' interval is inside the 'Normal' range."

"These are national thresholds. Route-specific thresholds may also be defined at some point."

"""
import os
import pandas as pd

from utils import cdd, cdd_schedule8, save_pickle, load_pickle


#
def read_thresholds_from_html():
    # thr: thresholds
    thr = pd.read_html(cdd("METEX\\Weather thresholds", "Weather-Thresholds_9306121.html"))
    thr = thr[0]
    # Specify column names
    hdr = thr.ix[0].tolist()
    thr.columns = hdr
    # Drop the first row, which has been used as the column names
    thr.drop(0, inplace=True)
    # clas: classification
    assert isinstance(thr, pd.DataFrame)
    clas = thr[pd.isnull(thr).any(axis=1)]['Classification']
    clas_list = []
    for i in range(len(clas)):
        # rpt: repeat
        to_rpt = (clas.index[i + 1] - clas.index[i] - 1) if i + 1 < len(clas) else (thr.index[-1] - clas.index[i])
        clas_list += [clas.iloc[i]] * to_rpt
    thr.drop(clas.index, inplace=True)
    thr.index = clas_list
    thr.index.names = ['Classification']
    thr.rename(columns={'Classification': 'Description'}, inplace=True)

    # Add 'VariableName' and 'Unit'
    variables = ['T', 'x', 'r', 'w']
    units = ['celsius degree', 'cm', 'mm', 'mph']
    variables_list, units_list = [], []
    for i in range(len(clas)):
        variables_temp = [variables[i]] * thr.index.tolist().count(clas.iloc[i])
        units_temp = [units[i]] * thr.index.tolist().count(clas.iloc[i])
        variables_list += variables_temp
        units_list += units_temp
    thr.insert(1, 'VariableName', variables_list)
    thr.insert(2, 'Unit', units_list)

    # Retain main description
    desc_temp = thr['Description'].tolist()
    for i in range(len(desc_temp)):
        desc_temp[i] = desc_temp[i].replace('\xa0', ' ')
        desc_temp[i] = desc_temp[i].replace(' ( oC )', '')
        desc_temp[i] = desc_temp[i].replace(', x (cm)', '')
        desc_temp[i] = desc_temp[i].replace(', r (mm)', '')
        desc_temp[i] = desc_temp[i].replace(', w (mph)', '')
        desc_temp[i] = desc_temp[i].replace(' (mph)', ' ')
    thr['Description'] = desc_temp

    # Upper and lower boundaries
    def boundary(df, col, sep1=None, sep2=None):
        if sep1:
            lst_lb = [thr[col].iloc[0].split(sep1)[0]]
            lst_lb += [v.split(sep2)[0] for v in thr[col].iloc[1:]]
            df.insert(df.columns.get_loc(col) + 1, col + 'LowerBound', lst_lb)
        if sep2:
            lst_ub = [thr[col].iloc[0].split(sep2)[1]]
            lst_ub += [v.split(sep1)[-1] for v in thr[col].iloc[1:]]
            if sep1:
                df.insert(df.columns.get_loc(col) + 2, col + 'UpperBound', lst_ub)
            else:
                df.insert(df.columns.get_loc(col) + 1, col + 'Threshold', lst_ub)

    boundary(thr, 'Normal', sep1=None, sep2='up to ')  # Normal
    boundary(thr, 'Alert', sep1=' \u003C ', sep2=' \u2264 ')  # Alert
    boundary(thr, 'Adverse', sep1=' \u003C ', sep2=' \u2264 ')  # Adverse
    extreme = [thr['Extreme'].iloc[0].split(' \u2264 ')[1]]  # Extreme
    extreme += [v.split(' \u2265 ')[1] for v in thr['Extreme'].iloc[1:]]
    thr['ExtremeThreshold'] = extreme

    return thr


# The threshold data is also available in the following file: "S8_Weather 02_06_2006 - 31-03-2014.xlsm"
def read_thresholds_from_workbook(update=False):
    path_to_file = cdd_schedule8("Reports", "Worksheet_Thresholds.pickle")
    if os.path.isfile(path_to_file) and not update:
        thresholds = load_pickle(path_to_file)
    else:
        try:
            thresholds = pd.read_excel(cdd_schedule8("Reports", "S8_Weather 02_06_2006 - 31-03-2014.xlsm"),
                                       sheetname="Thresholds", parse_cols="A:F")
            save_pickle(thresholds, path_to_file)
        except Exception as e:
            print("Reading weather thresholds from the workbook ... failed due to '{}'.".format(e))
            thresholds = None
    return thresholds


def get_weather_thresholds(update=False):
    path_to_file = cdd("METEX\\Weather thresholds", "Thresholds.pickle")
    if os.path.isfile(path_to_file):
        thresholds = load_pickle(path_to_file)
    else:
        try:
            thr0 = read_thresholds_from_html()
            thr1 = read_thresholds_from_workbook(update)
            thresholds = [thr0, thr1]
            save_pickle(thresholds, path_to_file)
        except Exception as e:
            print("Getting weather thresholds ... failed due to '{}'.".format(e))
            thresholds = None
    return thresholds
