""" Dictionary for errata """

import re

import pandas as pd
from pyhelpers.store import load_json

from utils import cdd_rc


# Create a dict for replace location names
def location_names_replacement_dict(k=None, as_dataframe=False):
    """
    :param k:
    :param as_dataframe:
    :return:
    """
    location_name_replacement_dict = load_json(cdd_rc("location-names-replacement.json"))
    if k:
        replacement_dict = {k: location_name_replacement_dict}
    else:
        replacement_dict = location_name_replacement_dict

    if as_dataframe:
        replacement_dict = pd.DataFrame.from_dict(replacement_dict)

    return replacement_dict


# Create a regex dict for replace location names
def location_names_regexp_replacement_dict(k=None, as_dataframe=False):
    """
    :param k:
    :param as_dataframe:
    :return:
    """
    location_regexp_replacement_dict = {
        re.compile(r' \(DC lines\)'): ' [DC lines]',
        re.compile(r' And | \+ '): ' & ',
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
        re.compile('-Y-'): '-y-',
        re.compile(r' A\.C\.C\.'): ' Avon County Council',
        re.compile(r' A\.B\.P\.'): ' Associated British Ports',
        re.compile(' A C E '): " Area Civil Engineer's ",
        re.compile(' Bardon Aggs'): ' Bardon Aggregates',
        re.compile('Bordsley'): 'Bordesley',
        re.compile(r' B\.T\.P\.'): ' British Tar Products',
        re.compile(r' C\. Sidings| C\.S\.$| Cs$'): ' Carriage Sidings',
        re.compile(r' C\.C\.E\.'): " Chief Civil Engineer's Sidings",
        re.compile(r' C\.E\.Sdgs'): " Civil Engineer's Sidings",
        re.compile(r' Car\. ?M\.D\.?| Cmd'): ' Carriage Maintenance Depot',
        re.compile(r' C\.H\.S\.?| C H S'): ' Carriage Holding Sidings',
        re.compile(r' Charrgtn'): ' Charrington Coal Concentration Depot',
        re.compile(r'CLAPS47|CLAPJ 147'): 'Clapham Junction Signal T147',
        re.compile(' Cm$'): ' CM',
        re.compile(r' C\.O\.D\.'): ' Central Ordnance Depot',
        re.compile(r' C\.P\.A\.'): ' Clyde Ports Authority',
        re.compile(r' C\.Pt\. '): ' Crown Point ',
        re.compile(r' C\.S\.M\.D\.'): ' Carriage Servicing & Maintenance Depot',
        re.compile(r' C\.S\.D\.'): ' Carriage Servicing Depot',
        re.compile(r' \(DBS\)'): ' DB Schenker',
        re.compile(r' Depot\.'): ' Depot',
        re.compile(r' D&U\.G\.L\.'): ' Down & Up Goods Loop',
        re.compile(r' D\.D\. H\.S\.'): ' Diesel Depot Holding Sidings',
        re.compile(r' D\.G\.L\.?| D G L'): ' Down Goods Loop',
        re.compile(r' D\.H\.S\.'): ' Down Holding Sidings',
        re.compile(r' D\.?P\.?L\.?'): ' Down Passenger Loop',
        re.compile(r' D\.M\.U\.D\.?| DMU Depot'): ' Diesel Multiple Unit Depot',
        re.compile(r' D\.R\.S\.'): ' Down Refuge Siding',
        re.compile(' Dn '): ' Down ',
        re.compile(' Dsn'): ' Down Sidings North',
        re.compile(' Dss'): ' Down Sidings South',
        re.compile(r'\. E\. '): ' East ',
        re.compile('Earls '): "Earl's ",
        re.compile(' Eccq '): ' ECC Quarries ',
        re.compile(' Emd$'): ' Electric Maintenance Depot',
        re.compile(r' E\.M\.U\.D\.?'): ' Electric Multiple Unit Depot',
        re.compile(r' E\.M\.U\. '): ' Electric Multiple Unit ',
        re.compile(r' E\.P\.S\.'): ' European Passenger Services',
        re.compile(r' \(EWS\)'): ' English Welsh & Scottish Railway',
        re.compile(r' Eur Frt Ops Cntre'): ' European Freight Operations Centre',
        re.compile(r' F\.C\.'): ' Flat Crossing',
        re.compile(r' F\.D\.'): ' Freight Depot',
        re.compile(r' F\.O\.B\.'): ' Foreign Ore Branch',
        re.compile(r' Ept'): ' Europort',
        re.compile(r"\(F'Liners\)|F/L"): 'Freightliner',
        re.compile(r'\(Ff\)'): ' Fastline Freight',
        re.compile(r' F[Ll]?[Hh][Hh]| \(F[Ll]?[Hh][Hh]\)| Fliner HH'): ' Freightliner Heavy Haul',
        re.compile(r' F\.L\.T\.| FLT| \(FLT\)|\.\.F\.L\.T\.'): ' Freightliner Terminal',
        re.compile(r' \(F[Ll][Tt]\)'): ' Freightliner',
        re.compile(r' Ryans F\.W\.'): ' Ryans Fletchers Wharf',
        re.compile(r' GBR[Ff]| \(GBR[Ff]\)| Gbf'): ' GB Railfreight',
        re.compile(r' G\.C\.'): ' Garden City',
        re.compile(r' G\.F\.'): ' Ground Frame',
        re.compile(' Gp '): ' Group ',
        re.compile(r' G\.S\.P\.'): ' Ground Shunt Panel',
        re.compile(' Gds Lp| Gds Loop'): ' Goods Loop',
        re.compile(r' H\.L\.'): ' High Level',
        re.compile(r' H\.S\.'): ' Holding Sidings',
        re.compile(' Ntl Pwr'): ' National Power',
        re.compile(r' Nth\.? '): ' North ',
        re.compile(r' I\.B\.'): ' Intermediate Block',
        re.compile(r' I\.C\.I\.'): ' ICI',
        re.compile(r' I\.?R\.?F\.?T\.?'): ' International Rail Freight Terminal',
        re.compile(' I[Ss][Uu]'): ' Infrastructure Servicing Unit',
        re.compile(r' Isu \(CE\)'): " Civil Engineer's Sidings",
        re.compile(' Int Rft Recep '): ' International Rail Freight Reception ',
        re.compile(' Intl E'): ' International East',
        re.compile(' Intl W'): ' International West',
        re.compile(r' Jn\.?| Jcn'): ' Junction',
        re.compile(' JN HL '): ' Junction High Level ',
        re.compile(r' J\.Yd '): ' Junction Yard ',
        re.compile(r' L\.C\.| L Xing'): ' Level Crossing',
        re.compile(r' L\.D\.C\.? '): ' Local Distribution Centre ',
        re.compile(r' L\.H\.S\.'): ' Loco Holding Siding',
        re.compile(r' L[. ]I[. ]P\.? ?'): ' Loco Inspection Point',
        re.compile(r' L\.L\.| Ll'): ' Low Level',
        re.compile(r' Lmd| L\.M\.D\.'): ' Light Maintenance Depot',
        re.compile(' Ln'): ' Lane',
        re.compile(r' L\.N\.W\. Junction Derby'): ' Derby LNW Junction',
        re.compile(' Loco Hs'): ' Loco Holding Sidings',
        re.compile(r' M\.C\.T\.'): ' Marine Container Terminal',
        re.compile(r' M\.& E\.E\\.'): ' Mechanical & Electrical Engineer',
        re.compile(r' M\.R\.C\.'): ' Midland Railway Centre',
        re.compile(r' N\.L\.F\.C\.'): ' North London FC',
        re.compile(' Ntl '): ' National ',
        re.compile(r' N\.R\.M\.'): ' National Railway Museum',
        re.compile(r' N\.Y\.| NY| N\. Y\.'): ' New Yard',
        re.compile(r' P\.A\.D\.'): ' Pre-Assembly Depot',
        re.compile(r' P\.S\.| P Stn| Power Stn'): ' Power Station',
        re.compile(" P'Way"): ' Permanent Way',
        re.compile(' Pwr '): ' Power ',
        re.compile(' Prdc'): ' Princess Royal Distribution Centre',
        re.compile(r' R\.C\.T\.'): ' Riverside Container Terminal',
        re.compile(' Rd.?'): ' Road',
        re.compile(r' Recp\.'): ' Reception',
        re.compile(r' \(?RFD\)?'): ' Railfreight Distribution',
        re.compile(r' R\.T\.S\.?'): ' Refuse Transfer Station',
        re.compile(r' R\.S GB '): ' Refuge Siding GB ',
        re.compile(r' S\.B\.?| Sb| S B| Ss'): ' Signal Box',
        re.compile(r' S\.S\.'): ' Sorting Sidings',
        re.compile(' S C C E'): " Sandiacre Chief Civil Engineer's",
        re.compile(r' Sdg\.?| Siding\.'): ' Siding',
        re.compile(' Sdgs'): ' Sidings', re.compile(' Sdgs '): ' Sidings ',
        re.compile(r' S[Ii][Gg]\.? '): ' Signal ',
        re.compile(r'Sig\.Ty357'): 'Signal TY357',
        re.compile(r' Sth\.? '): ' South ',
        re.compile(r' South C\.E\.'): " South Civil Engineer's Sidings",
        re.compile(r' \(S\.Yorks\)'): 'Swinton (South Yorks)',
        re.compile(' Steetley Coy'): ' Steetley Company',
        re.compile(' Terml| Terminal\\.'): ' Terminal',
        re.compile(r' T\.C\.'): ' Terminal Complex',
        re.compile(r' T\.?M\.?D\.?'): ' Traction Maintenance Depot',
        re.compile(r' T\.?& ?R\.S\.M\.D\.?'): ' Traction & Rolling Stock Maintenance Depot',
        re.compile(r' T\.C\.'): ' Terminal Complex',
        re.compile(' U&Dgl'): ' Up & Down Goods Loop',
        re.compile(r' U\.C\.H\.S\.'): ' Up Carriage Holding Sidings',
        re.compile(r' U\.G\.L\.'): ' Up Goods Loop',
        re.compile(r' U\.P\.L\.'): ' Up Passenger Loop',
        re.compile(r' U\.R\.S\.'): ' Up Relief Siding',
        re.compile(' Usn'): ' Up Sidings North',
        re.compile(' Uss'): ' Up Sidings South',
        re.compile(r' \(VWL\)'): ' Victa Westlink Rail',
        re.compile(r' \(Vq\)'): ' Virtual Quarry',
        re.compile(r' \(West Mids\)'): ' (West Midlands)',
        re.compile(r' W\.R\.D\.'): ' Wagon Repair Depot',
        re.compile(' W Yard'): ' West Yard',
        re.compile('west533'): 'Westerton Signal YH533',
        re.compile(' Wks Lafarg'): ' Works Lafarg',
        re.compile(' TURNBACK'): ' Turnback Siding',
        re.compile(' Wtr Wh?f '): ' Water Wharf ',
        re.compile(r'Warrington C\.E\. Sidings'): "Warrington NCL/Civil Engineer's Sidings",
        re.compile(r' Wm Csd\.'): ' West Marina Carriage Servicing Depot',
        re.compile(' Yd '): ' Yard '}

    if k is not None:
        replacement_dict = {k: location_regexp_replacement_dict}
    else:
        replacement_dict = location_regexp_replacement_dict

    if as_dataframe:
        replacement_dict = pd.DataFrame.from_dict(replacement_dict)

    return replacement_dict
