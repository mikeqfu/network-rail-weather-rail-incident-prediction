""" Dictionary for errata """

import re
import pandas as pd
from utils import cdd_rc, load_json


# Create a dict for replace location names
def create_location_names_replacement_dict(k=None, as_dataframe=False):
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
def create_location_names_regexp_replacement_dict(k=None, as_dataframe=False):
    """
    :param k:
    :param as_dataframe:
    :return:
    """
    location_regexp_replacement_dict = {
        re.compile(' \(DC lines\)'): ' [DC lines]',
        re.compile(' And | \+ '): ' & ',
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
        re.compile(' A\.C\.C\.'): ' Avon County Council',
        re.compile(' A\.B\.P\.'): ' Associated British Ports',
        re.compile(' A C E '): " Area Civil Engineer's ",
        re.compile(' Bardon Aggs'): ' Bardon Aggregates',
        re.compile('Bordsley'): 'Bordesley',
        re.compile(' B\.T\.P\.'): ' British Tar Products',
        re.compile(' C\. Sidings| C\.S\.$| Cs$'): ' Carriage Sidings',
        re.compile(' C\.C\.E\.'): " Chief Civil Engineer's Sidings",
        re.compile(' C\.E\.Sdgs'): " Civil Engineer's Sidings",
        re.compile(' Car\. ?M\.D\.?| Cmd'): ' Carriage Maintenance Depot',
        re.compile(' C\.H\.S\.?| C H S'): ' Carriage Holding Sidings',
        re.compile(' Charrgtn'): ' Charrington Coal Concentration Depot',
        re.compile('CLAPS47|CLAPJ 147'): 'Clapham Junction Signal T147',
        re.compile(' Cm$'): ' CM',
        re.compile(' C\.O\.D\.'): ' Central Ordnance Depot',
        re.compile(' C\.P\.A\.'): ' Clyde Ports Authority',
        re.compile(' C\.Pt\. '): ' Crown Point ',
        re.compile(' C\.S\.M\.D\.'): ' Carriage Servicing & Maintenance Depot',
        re.compile(' C\.S\.D\.'): ' Carriage Servicing Depot',
        re.compile(' \(DBS\)'): ' DB Schenker',
        re.compile(' Depot\.'): ' Depot',
        re.compile(' D&U\.G\.L\.'): ' Down & Up Goods Loop',
        re.compile(' D\.D\. H\.S\.'): ' Diesel Depot Holding Sidings',
        re.compile(' D\.G\.L\.?| D G L'): ' Down Goods Loop',
        re.compile(' D\.H\.S\.'): ' Down Holding Sidings',
        re.compile(' D\.?P\.?L\.?'): ' Down Passenger Loop',
        re.compile(' D\.M\.U\.D\.?| DMU Depot'): ' Diesel Multiple Unit Depot',
        re.compile(' D\.R\.S\.'): ' Down Refuge Siding',
        re.compile(' Dn '): ' Down ',
        re.compile(' Dsn'): ' Down Sidings North',
        re.compile(' Dss'): ' Down Sidings South',
        re.compile('\. E\. '): ' East ',
        re.compile('Earls '): "Earl's ",
        re.compile(' Eccq '): ' ECC Quarries ',
        re.compile(' Emd$'): ' Electric Maintenance Depot',
        re.compile(' E\.M\.U\.D\.?'): ' Electric Multiple Unit Depot',
        re.compile(' E\.M\.U\. '): ' Electric Multiple Unit ',
        re.compile(' E\.P\.S\.'): ' European Passenger Services',
        re.compile(' \(EWS\)'): ' English Welsh & Scottish Railway',
        re.compile(' Eur Frt Ops Cntre'): ' European Freight Operations Centre',
        re.compile(' F\.C\.'): ' Flat Crossing',
        re.compile(' F\.D\.'): ' Freight Depot',
        re.compile(' F\.O\.B\.'): ' Foreign Ore Branch',
        re.compile(' Ept'): ' Europort',
        re.compile("\(F'Liners\)|F/L"): 'Freightliner',
        re.compile('\(Ff\)'): ' Fastline Freight',
        re.compile(' F[Ll]?[Hh][Hh]| \(F[Ll]?[Hh][Hh]\)| Fliner HH'): ' Freightliner Heavy Haul',
        re.compile(' F\.L\.T\.| FLT| \(FLT\\)|\.\.F\.L\.T\.'): ' Freightliner Terminal',
        re.compile(' \(F[Ll][Tt]\)'): ' Freightliner',
        re.compile(' Ryans F\.W\.'): ' Ryans Fletchers Wharf',
        re.compile(' GBR[Ff]| \(GBR[Ff]\)| Gbf'): ' GB Railfreight',
        re.compile(' G\.C\.'): ' Garden City',
        re.compile(' G\.F\.'): ' Ground Frame',
        re.compile(' Gp '): ' Group ',
        re.compile(' G\.S\.P\.'): ' Ground Shunt Panel',
        re.compile(' Gds Lp| Gds Loop'): ' Goods Loop',
        re.compile(' H\.L\.'): ' High Level',
        re.compile(' H\.S\.'): ' Holding Sidings',
        re.compile(' Ntl Pwr'): ' National Power',
        re.compile(' Nth\.? '): ' North ',
        re.compile(' I\.B\.'): ' Intermediate Block',
        re.compile(' I\.C\.I\.'): ' ICI',
        re.compile(' I\.?R\.?F\.?T\.?'): ' International Rail Freight Terminal',
        re.compile(' I[Ss][Uu]'): ' Infrastructure Servicing Unit',
        re.compile(' Isu \(CE\)'): " Civil Engineer's Sidings",
        re.compile(' Int Rft Recep '): ' International Rail Freight Reception ',
        re.compile(' Intl E'): ' International East',
        re.compile(' Intl W'): ' International West',
        re.compile(' Jn\.?| Jcn'): ' Junction',
        re.compile(' JN HL '): ' Junction High Level ',
        re.compile(' J\.Yd '): ' Junction Yard ',
        re.compile(' L\.C\.| L Xing'): ' Level Crossing',
        re.compile(' L\.D\.C\.? '): ' Local Distribution Centre ',
        re.compile(' L\.H\.S\.'): ' Loco Holding Siding',
        re.compile(' L[. ]I[. ]P\.? ?'): ' Loco Inspection Point',
        re.compile(' L\.L\.| Ll'): ' Low Level',
        re.compile(' Lmd| L\.M\.D\.'): ' Light Maintenance Depot',
        re.compile(' Ln'): ' Lane',
        re.compile(' L\.N\.W\. Junction Derby'): ' Derby LNW Junction',
        re.compile(' Loco Hs'): ' Loco Holding Sidings',
        re.compile(' M\.C\.T\.'): ' Marine Container Terminal',
        re.compile(' M\.& E\.E\\.'): ' Mechanical & Electrical Engineer',
        re.compile(' M\.R\.C\.'): ' Midland Railway Centre',
        re.compile(' N\.L\.F\.C\.'): ' North London FC',
        re.compile(' Ntl '): ' National ',
        re.compile(' N\.R\.M\.'): ' National Railway Museum',
        re.compile(' N\.Y\.| NY| N\. Y\.'): ' New Yard',
        re.compile(' P\.A\.D\.'): ' Pre-Assembly Depot',
        re.compile(' P\.S\.| P Stn| Power Stn'): ' Power Station',
        re.compile(" P'Way"): ' Permanent Way',
        re.compile(' Pwr '): ' Power ',
        re.compile(' Prdc'): ' Princess Royal Distribution Centre',
        re.compile(' R\.C\.T\.'): ' Riverside Container Terminal',
        re.compile(' Rd.?'): ' Road',
        re.compile(' Recp\.'): ' Reception',
        re.compile(' \(?RFD\)?'): ' Railfreight Distribution',
        re.compile(' R\.T\.S\.?'): ' Refuse Transfer Station',
        re.compile(' R\.S GB '): ' Refuge Siding GB ',
        re.compile(' S\.B\.?| Sb| S B| Ss'): ' Signal Box',
        re.compile(' S\.S\.'): ' Sorting Sidings',
        re.compile(' S C C E'): " Sandiacre Chief Civil Engineer's",
        re.compile(' Sdg\.?| Siding\.'): ' Siding',
        re.compile(' Sdgs'): ' Sidings', re.compile(' Sdgs '): ' Sidings ',
        re.compile(' S[Ii][Gg]\.? '): ' Signal ',
        re.compile('Sig\.Ty357'): 'Signal TY357',
        re.compile(' Sth\.? '): ' South ',
        re.compile(' South C\.E\.'): " South Civil Engineer's Sidings",
        re.compile(' \(S\.Yorks\)'): 'Swinton (South Yorks)',
        re.compile(' Steetley Coy'): ' Steetley Company',
        re.compile(' Terml| Terminal\\.'): ' Terminal',
        re.compile(' T\.C\.'): ' Terminal Complex',
        re.compile(' T\.?M\.?D\.?'): ' Traction Maintenance Depot',
        re.compile(' T\.?& ?R\.S\.M\.D\.?'): ' Traction & Rolling Stock Maintenance Depot',
        re.compile(' T\.C\.'): ' Terminal Complex',
        re.compile(' U&Dgl'): ' Up & Down Goods Loop',
        re.compile(' U\.C\.H\.S\.'): ' Up Carriage Holding Sidings',
        re.compile(' U\.G\.L\.'): ' Up Goods Loop',
        re.compile(' U\.P\.L\.'): ' Up Passenger Loop',
        re.compile(' U\.R\.S\.'): ' Up Relief Siding',
        re.compile(' Usn'): ' Up Sidings North',
        re.compile(' Uss'): ' Up Sidings South',
        re.compile(' \(VWL\)'): ' Victa Westlink Rail',
        re.compile(' \(Vq\)'): ' Virtual Quarry',
        re.compile(' \(West Mids\)'): ' (West Midlands)',
        re.compile(' W\.R\.D\.'): ' Wagon Repair Depot',
        re.compile(' W Yard'): ' West Yard',
        re.compile('west533'): 'Westerton Signal YH533',
        re.compile(' Wks Lafarg'): ' Works Lafarg',
        re.compile(' TURNBACK'): ' Turnback Siding',
        re.compile(' Wtr Wh?f '): ' Water Wharf ',
        re.compile('Warrington C\.E\. Sidings'): "Warrington NCL/Civil Engineer's Sidings",
        re.compile(' Wm Csd\.'): ' West Marina Carriage Servicing Depot',
        re.compile(' Yd '): ' Yard '}

    if k is not None:
        replacement_dict = {k: location_regexp_replacement_dict}
    else:
        replacement_dict = location_regexp_replacement_dict

    if as_dataframe:
        replacement_dict = pd.DataFrame.from_dict(replacement_dict)

    return replacement_dict
