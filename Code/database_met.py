""" Read and clean data of NR_METEX database """

import os
import re

import datetime_truncate
import matplotlib.patches
import matplotlib.pyplot as plt
import mpl_toolkits.basemap
import pandas as pd
import shapely.geometry

import database_utils as db
import database_veg as dbv
import railwaycodes_utils as rc
from converters import yards_to_mileage
from utils import cd, cdd_rc, cdd_delay_attr, save, save_pickle, load_pickle, save_json, load_json, find_match

# ====================================================================================================================
""" Change directories """


# Change directory to "Data\\METEX\\Database\\Tables" and sub-directorie
def cdd_metex_db_tables(*directories):
    path = db.cdd_metex_db("Tables")
    os.makedirs(path, exist_ok=True)
    for directory in directories:
        path = os.path.join(path, directory)
    return path


# Change directory to "Data\\METEX\\Database\\Views" and sub-directorie
def cdd_metex_db_views(*directories):
    path = db.cdd_metex_db("Views")
    os.makedirs(path, exist_ok=True)
    for directory in directories:
        path = os.path.join(path, directory)
    return path


# Change directory to "METEX\\Database\\Figures" and sub-directories
def cdd_metex_db_fig(*directories):
    path = db.cdd_metex_db("Figures")
    os.makedirs(path, exist_ok=True)
    for directory in directories:
        path = os.path.join(path, directory)
    return path


# Change directory to "Publications\\Journal\\Figures" and sub-directories
def cdd_metex_db_fig_pub(pid, *directories):
    path = cd("Publications", "Journals", pid, "Figures")
    for directory in directories:
        path = os.path.join(path, directory)
    return path


# ====================================================================================================================
""" Special utils """


# Create a dict for replace location names
def create_loc_name_replacement_dict(k=None):
    loc_repl_dict = {
        '"Tyndrum Upper" (Upper Tyndrum)': 'Upper Tyndrum',
        '"Wigston South" (South Wigston)': 'South Wigston',
        '03405': 'Stirling Sidings', 'AND008': 'Andover Signal 8',
        'ATLBRJN': 'Attleborough South Junction',
        "Allington West Junction ['(formerly Allington Junction)']": 'Allington West Junction',
        'Angerstein Wharf (Bardon)': 'Angerstein Wharf Bardon',
        'Angerstein Wharf (Tarmac)': 'Angerstein Wharf Tarmac',
        'Appleby LC': 'Scunthorpe Appleby Level Crossing',
        'Avonmouth (Fastline)': 'Avonmouth Bennets Fastline',
        'BARONS COURT LT': 'Barons Court LT',
        'BCST196': 'Bicester Signal ME196',
        'BOW OLYMPICS (GBRf)': 'Bow Depot Olympics GB Railfreight',
        'BSNGW34': 'Basingstoke Signal YW34',
        'Barnetby Down Sidings': 'Barnetby Down/On Track Machine Sidings',
        'Barry Docks Abp Shipment': 'Barry Docks ABP Shipment',
        'Barry Docks BP Chemicals [later Zeon]': 'Barry Docks Zeon',
        'Barry Docks [station]': 'Barry Docks',
        'Bentley (S.Yorks)': 'Bentley (Yorks)',
        'Bury St. Edmunds Cmd': 'Bury St Edmunds Carriage Maintenance Depot',
        'CADOEWS': 'Cadoxton English Welsh & Scottish Railway',
        'CARDCSS': 'Cardiff Docks Coastal Shipping',
        'CARLISLE NY DBS LOCO MAIN PT': 'Carlisle New Yard DB Schenker Locomotive Maintenance Point',
        'CLAPS47': 'Clapham Junction Signal W1047',
        'CLHMABL': 'Chelmsford Arbour Lane',
        'CREWE SSN SIG. NH19': 'Crewe Signal NH19',
        'CTNM': 'Castleton Moor',
        'Cadder Loops/Cadder Yard DRS': 'Cadder Yard DRS',
        'Cambridge T.&R.S.M.D.': 'Cambridge Traction & Rolling Stock Maintenance Depot',
        'Carlisle Kingmoor Jcn': 'Kingmoor Junction',
        'Carlisle Kingmoor V.Q.(RT)': 'Carlisle Kingmoor Yard English Welsh & Scottish Railway/Marcroft/Wabtec',
        'Castleton R.W.D.': 'Castleton Wagon Repair Depot',
        'Chapel-En-Le-Frith': 'Chapel-en-le-Frith',
        'Coleham Isu (CE)': "Coleham Civil Engineer's Sidings",
        'Crewe Wagon Shop Centrac (later Crewe Gresty Bridge DRS)': 'Crewe Gresty Bridge DRS',
        'Dalston Junction (ELL)': 'Dalston Junction (East London Line)',
        'Dean (Wilts)': 'Dean',
        'Derby North Doc Sdgs': 'Derby North Dock Sidings',
        'Ditton (later Ditton East Junction)': 'Ditton East Junction',
        "Ditton Foundry Lane Victa Westlink Rail (['codes no longer used'])": 'Ditton Foundry Lane Victa Westlink Rail',
        'Ditton Reception (FLT)': 'Ditton Foundry Lane AHC Freightliner',
        'Doncaster Rfs Engineering': 'Doncaster RFS Engineering/Wabtec',
        'Doncaster Up Decoy (Fastline)': 'Doncaster Up Decoy GB Railfreight',
        'East Cowton Xovers': 'Darlington East Cowton Crossovers',
        'East Somerset Junction (Witham)': 'East Somerset Junction',
        'Eastfields (Mitcham)': 'Mitcham Eastfields',
        'Eastleigh Works-Alstom-Fl': 'Eastleigh Works Alstom Freightliner',
        'Ely (Cambs)': 'Ely', 'Epsom - Up Sidings': 'Epsom Up Sidings',
        'Exeter St Davids Prem Tran': 'Exeter St Davids Premier Transport',
        'Exetr Sd Carr Sdgs Wessex': 'Exeter St Davids Carriage Sidings Wessex',
        'Farnborough (Main)': 'Farnborough Main',
        'Faversham Down Sdg No1': 'Faversham Down Siding No 1',
        'Fenny Compton M.O.D.': 'Fenny Compton MOD',
        'Garston(Speke) T.C.': 'Garston Speke Terminal Complex',
        'Gascoigne Wood  Up Sidings': 'Gascoigne Wood Up Sidings',
        'Gascoigne Wood Down Bunker Sidings (?)': 'Gascoigne Wood Down (DBS)',
        'Gillingham E.M.U.D.': 'Gillingham (Kent) Electric Multiple Unit Depot',
        'Gladstone Dock BT (Fastline)': 'Liverpool Bulk Terminal Fastline Freight',
        'Grangemouth (VWL)': 'Grangemouth TDG Victa Westlink Rail',
        'Grangemouth BP (later Innovene) Chemicals': 'Grangemouth Innovene (Chemicals)',
        'Grangemouth BP (later Innovene) Holding Sidings': 'Grangemouth Innovene Holding Sidings',
        'Grangemouth Tdg (EWS)': 'Grangemouth TDG English Welsh & Scottish Railway',
        'Grangemouth Tdg Main Term': 'Grangemouth TDG Main Terminal',
        'Grangemouth Wh Malcolm': 'Grangemouth WH Malcolm',
        'Grosvr Cs': 'Victoria Grosvenor Carriage Shed',
        'Guide Bridge Advenza Freight (?)': 'Guide Bridge Advenza Sidings FLHH',
        'Guildford Up Recp': 'Guildford Up Reception',
        'HIGH MEADS JN': 'High Meads Junction',
        'Heysham Harbour [station]': 'Heysham Harbour',
        'High Marnham (Freightliner Heavy Haul)': 'High Marnham Freightliner Heavy Haul',
        'High Wycombe (West Wycombe)': 'West Wycombe',
        'Hitchin Up Scrap Siding (Advenza': 'Hitchin Advenza Freight',
        'Hoo Junction Signal Nk1611': 'Hoo Junction Signal NK1611',
        'Hope (Earles Sidings) Fhh': 'Earles Sidings (Hope) Freightliner Heavy Haul',
        'IMMINGHAM B4 SIDINGS (DBS)': 'Immingham B4 Sidings',
        'Ilford Lon End Junction': 'Ilford London End Junction',
        'Immingham Dock Cotswold Rail (?)': 'Immingham Dock Coals Rail',
        'Immingham Dock Ct (Fastline)': 'Immingham Hargreaves Container Terminal Fastline Freight',
        'Immingham Dock [unknown feature]': 'Immingham Reception (FLHH)',
        'Immingham HIT (Fastline)': 'Immingham Humber International Terminal Freightliner Heavy Haul',
        'Immingham NCB PAD1 (Fastline)': 'Immingham NCB Pad 1 Fastline Freight',
        'Imperial Wharf (originally planned to be Chelsea Harbour)': 'Imperial Wharf',
        'Invergordon Distillery (No 2 Up Siding)': 'Invergordon Distillery',
        'Ironbridge Power Station (also known as Buildwas CEGB)': 'Ironbridge Power Station',
        'KNOTTINGLEY SIG FE6418': 'Knottingley Signal FE6418',
        'Kennington Junction (also listed as Kennington Junction Points Heaters)': 'Kennington Junction',
        'Kings Cross Copenhagen (formerly Freight Terminal) Junction': "King's Cross Copenhagen Junction",
        'Kings Norton Ot Plant Dept': 'Kings Norton On Track Plant Depot',
        'Kingsbury G Cohen (later Kingsbury Coal Sidings?)': 'Kingsbury Coal Sidings',
        'Kirkhmtip': 'Kirkham & Wesham Tip',
        "L'Pool Sth Pw Hl (Allertn)": 'Liverpool South Parkway High Level',
        'Leicester Car Sidings': 'Leicester Carriage Sidings',
        'Lindleys Lane (Kirkby South Junction)': 'Kirkby South Junction',
        'Liverpool Street Sig L572': 'Billericay Signal L572',
        'London Bridge (Central) (Platforms 14 - 16)': 'London Bridge',
        'MLRHM21': 'Millerhill Signal M21',
        'MOORPK': 'Moor Park',
        'MORRISC': 'Morris Cowley',
        'Maltby Colliery National Coal Board/RJB Mining': 'Maltby Colliery',
        'Northampton Down Goods Loop (formerly Northampton No 4)': 'Northampton Down Goods Loop',
        'Old Oak Common C.S. (EWS)': 'Old Oak Common Carriage Sidings English Welsh & Scottish Railway',
        'PEEL GROUP SIDING ELLESMERE PORT': 'Ellesmere Port Freightliner Heavy Haul',
        'PLAISTOW L.T.': 'Plaistow LT',
        'Paddn Yd Marcon Topmix': 'Paddington Yard Marcon Topmix',
        'Parc Slip - Celtic Energy': 'Margam Parc Slip Celtic Energy',
        'Peckham Rye (Catford Loop Lines)': 'Peckham Rye',
        'Portbury Coal Terml (Fastline)': 'Portbury Coal Terminal Fastline Freight',
        'Purfleet Deep Wharf (VWL)': 'Purfleet Deep Water Wharf Victa Westlink Rail',
        'RETFORD Signal D1348': 'Retford Signal D1348',
        "RIDHAM DOCK B'WTRS SDG": 'Ridham Sidings',
        'Railtrack Head Quarters Training Location Delta': 'Dollands Moor CTRL',
        'Ripple Lane Advenza Frt': 'Ripple Lane Advenza Railfreight',
        'Rugeley B PS (FGD) (GBRF)': 'Rugeley Power Station GB Railfreight',
        'STRNHDS': 'Stourbridge Junction Light Maintenance Depot Headshunt',
        'Scunthorpe B.S.C.(Ent.C.)': 'Scunthorpe BSC Entrance C',
        'Seaforth C.T. Mdhc (EWS)': 'Seaforth Container Terminal Mersey Docks & Harbour Commission',
        'Shadwell (ELL)': 'Shadwell',
        'Shrewsbury Abbey Frgte Cs': 'Shrewsbury Abbey Foregate Sidings',
        'Small Heath Lafarge Aggr': 'Small Heath Lafarge Aggregates',
        'Smitham (renamed Coulsdon Town)': 'Smitham',
        'Southend Airport Stn': 'Southend Airport [station]',
        'St Neots Fd (E W & S)': 'St Neots Freight Depot English Welsh & Scottish Railway',
        "St Nicholas (Carlisle) Civil Engineer's Sidings": "Carlisle St Nicholas Civil Engineer's Sidings",
        'St Pancras International (MML)': 'St Pancras [domestic station]',
        'St Peters (formerly Sunderland Monkseaton)': 'St Peters',
        'Stoke [-on-Trent] Junction': 'Stoke Junction',
        'Sudforth Lane Up Rs': 'Sudforth Lane Up Reception Sidings',
        'Swinderby Thorpe-On-The-Hill': 'Swinderby Thorpe-on-the-Hill',
        'TONBRIDGE ENGINEERS SIDING': 'Tonbridge Engineers Siding',
        'TOTONWRD': 'Toton Wagon Repair Depot',
        'Temple Mills Orient CS': 'Temple Mills Orient Way Carriage Sidings',
        'Temple Mills S.S.': 'Temple Mills Sorting Sidings',
        'Terrace Carriage Holding Sidings [Lincoln]': 'Lincoln Terrace Carriage Holding Sidings',
        'Thornhill [Scotland]': 'Thornhill',
        'Three Bridges P & Md': 'Three Bridges P & MD',
        'Tilbury I.R.F.T. Colas': 'Tilbury International Rail Freight Terminal Victa Advenza/Colas',
        'Tilbury IRFT (DRS)': 'Tilbury International Rail Freight Terminal DRS',
        'Upton Lovell (by 2010 STANOX listed as "now deleted")': 'Upton Lovell',
        'WLNDGL': 'Wellingborough Down Goods Loop',
        'WLSD2DT': 'Willesden Brent 2 Down Through Siding/Up & Down Goods',
        'Warrdalln': 'Warrington Dallam J G Russell',
        'Warrington C.E. Sdgs': "Warrington NCL/Civil Engineer's Sidings",
        'Warrington Latchford Sdgs (DBS)': 'Latchford Sidings',
        'Warrington Latchford Sdgs (FLHH)': 'Latchford Sidings Freightliner Heavy Haul',
        'Waterloo (Ballast)': 'Waterloo Ballast',
        'Wavertree P.C.D. (Oou)': 'Wavertree PCD',
        'Wellingborough Up TC GBRF': 'Wellingborough Up Sidings GB Railfreight',
        'Whitechapel (ELL)': 'Whitechapel',
        'Willesden Brent (FLT)': 'Willesden Brent Sidings Freightliner',
        'Willesden Brent Freightliner': 'Willesden Brent Sidings Freightliner',
        'Wilsdn Yd': 'Willesden Yard CTT Forwardings',
        'Woodhouse Junc Sdgs (Fhh)': 'Woodhouse Junction Sidings Freightliner Heavy Haul',
        'Yarmouth [Great Yarmouth]': 'Yarmouth'}
    if k:
        return {k: loc_repl_dict}
    else:
        return loc_repl_dict


# Create a regex dict for replace location names
def create_loc_name_regexp_replacement_dict(k=None):
    loc_regexp_repl_dict = {
        re.compile('\\['): '(',
        re.compile(']'): ')',
        re.compile(' \\[station]'): '',
        re.compile(' And '): ' & ',
        re.compile(' \\(was Eurostar Depot\\)'): '',
        re.compile('-In-| In '): '-in-',
        re.compile('-La-'): '-la-',
        re.compile('-Le-'): '-le-',
        re.compile('-On-| On '): '-on-',
        re.compile(' Of '): ' of ',
        re.compile('-Super-'): '-super-',
        re.compile('-Upon-| Upon '): '-upon-',
        re.compile('-Under-'): '-under-',
        re.compile('-Y-'): '-y-',
        re.compile('Depot \\(E\\)'): '',
        re.compile(' A\\.C\\.C\\.'): ' Avon County Council',
        re.compile(' A\\.?B\\.?P\\.?'): ' Associated British Ports',
        re.compile(' A C E '): " Area Civil Engineer's ",
        re.compile(' Bardon Aggs'): ' Bardon Aggregates',
        re.compile(' B\\.T\\.P\\.'): ' British Tar Products',
        re.compile(' C\\. Sidings'): ' Carriage Sidings',
        re.compile(' C\\.C\\.E\\.'): " Chief Civil Engineer's Sidings",
        re.compile('Canada water'): 'Canada Water',
        re.compile(' C\\.E\\.Sdgs'): " Civil Engineer's Sidings",
        re.compile(' Car\\. ?M\\.D\\.?| Cmd'): ' Carriage Maintenance Depot',
        re.compile(' C\\.S\\.M\\.D\\.'): ' Carriage Servicing & Maintenance Depot',
        re.compile(' C\\.Pt\\. '): ' Crown Point ',
        re.compile(' C\\.H\\.S\\.'): ' Carriage Holding Sidings',
        re.compile(' Charrgtn'): ' Charrington Coal Concentration Depot',
        re.compile(' C\\.P\\.A\\.'): ' Clyde Ports Authority',
        re.compile(' C\\.S\\.D\\.'): ' Carriage Servicing Depot',
        re.compile(' \\(DBS\\)'): ' DB Schenker',
        re.compile(' Depot\\.'): ' Depot',
        re.compile(' D&U\\.G\\.L\\.'): ' Down & Up Goods Loop',
        re.compile(' D\\.D\\. H\\.S\\.'): ' Diesel Depot Holding Sidings',
        re.compile(' D\\.G\\.L\\.?| D G L'): ' Down Goods Loop',
        re.compile(' D\\.H\\.S\\.'): ' Down Holding Sidings',
        re.compile(' D\\.?P\\.?L\\.?'): ' Down Passenger Loop',
        re.compile(' D\\.M\\.U\\.D| DMU Depot'): ' Diesel Multiple Unit Depot',
        re.compile(' D\\.R\\.S\\.| DRS'): ' Down Refuge Siding',
        re.compile(' Dn '): ' Down ',
        re.compile(' Dsn'): ' Down Sidings North',
        re.compile(' Dss'): ' Down Sidings South',
        re.compile('\\. E\\. '): ' East ',
        re.compile('Earls '): "Earl's ",
        re.compile(' Eccq '): ' ECC Quarries ',
        re.compile(' E\\.M\\.U\\.D\\.?'): ' Electric Multiple Unit Depot',
        re.compile(' E\\.M\\.U\\. '): ' Electric Multiple Unit ',
        re.compile(' E\\.P\\.S\\.'): ' European Passenger Services',
        re.compile(' EWS| \\(EWS\\)'): ' English Welsh & Scottish Railway',
        re.compile(' Eur Frt Ops Cntre'): ' European Freight Operations Centre',
        re.compile(' F\\.C\\.'): ' Flat Crossing',
        re.compile(' F\\.D\\.'): ' Freight Depot',
        re.compile(' Ept'): ' Europort',
        re.compile("\\(F'Liners\\)|F/L"): 'Freightliner',
        re.compile('\\(Ff\\)'): ' Fastline Freight',
        re.compile(' F[Ll]?[Hh][Hh]| \\(F[Ll]?[Hh][Hh]\\)| Fliner HH'): ' Freightliner Heavy Haul',
        re.compile(' F\\.L\\.T\\.| FLT| \\(FLT\\)'): ' Freightliner Terminal',
        re.compile(' \\(F[Ll][Tt]\\)'): ' Freightliner',
        re.compile(' Ryans F\\.W\\.'): ' Ryans Fletchers Wharf',
        re.compile(' GBR[Ff]| \\(GBR[Ff]\\)| Gbf'): ' GB Railfreight',
        re.compile(' G\\.C\\.'): ' Garden City',
        re.compile(' G\\.F\\.'): ' Ground Frame',
        re.compile(' Gp '): ' Group ',
        re.compile(' G\\.S\\.P\\.'): ' Ground Shunt Panel',
        re.compile(' Gds Lp| Gds Loop'): ' Goods Loop',
        re.compile(' H\\.L\\.'): ' High Level',
        re.compile(' H\\.S\\.'): ' Holding Sidings',
        re.compile(' Ntl Pwr'): ' National Power',
        re.compile(' Nth\\.? '): ' North ',
        re.compile(' I\\.B\\.'): ' Intermediate Block',
        re.compile(' I\\.?R\\.?F\\.?T\\.?'): ' International Rail Freight Terminal',
        re.compile(' I[Ss][Uu]'): ' Infrastructure Servicing Unit',
        re.compile(' Isu \\(CE\\)'): " Civil Engineer's Sidings",
        re.compile(' Int Rft Recep '): 'International Rail Freight Reception ',
        re.compile(' Intl E'): ' International East',
        re.compile(' Intl W'): ' International West',
        re.compile(' Jn\\.?| Jcn'): ' Junction',
        re.compile(' JN HL '): ' Junction High Level ',
        re.compile(' J\\.Yd '): ' Junction Yard ',
        re.compile(' L\\.C\\.| L Xing'): ' Level Crossing',
        re.compile(' L\\.D\\.C\\.? '): ' Local Distribution Centre ',
        re.compile(' L\\.H\\.S\\.'): ' Loco Holding Siding',
        re.compile(' L\\.I\\.P.'): ' Loco Inspection Point',
        re.compile(' L\\.L\\.| Ll'): ' Low Level',
        re.compile(' Lmd'): ' Light Maintenance Depot',
        re.compile(' Ln'): ' Lane',
        re.compile(' L\\.N\\.W\\. Junction Derby'): ' Derby LNW Junction',
        re.compile(' Loco Hs'): ' Loco Holding Sidings',
        re.compile(' M\\.& E\\.E\\.'): ' Mechanical & Electrical Engineer',
        re.compile(' M\\.R\\.C\\.'): ' Midland Railway Centre',
        re.compile(' N\\.L\\.F\\.C\\.'): ' North London FC',
        re.compile(' Ntl '): ' National ',
        re.compile(' N\\.Y\\.| NY'): ' New Yard',
        re.compile(' P\\.A\\.D\\.'): ' Pre-Assembly Depot',
        re.compile(' P\\.S\\.'): ' Power Station',
        re.compile(" P'Way"): ' Permanent Way',
        re.compile(' Pwr '): ' Power ',
        re.compile(' Prdc'): ' Princess Royal Distribution Centre',
        re.compile(' R\\.C\\.T\\.'): ' Riverside Container Terminal',
        re.compile(' Rd'): ' Road',
        re.compile(' Recp\\.'): ' Reception',
        re.compile(' \\(RFD\\)'): ' Railfreight Distribution',
        re.compile(' R\\.T\\.S\\.'): ' Refuse Transfer Station',
        re.compile(' R\\.S GB '): ' Refuge Siding GB ',
        re.compile(' S\\.B\\.| Sb'): ' Signal Box',
        re.compile(' S C C E'): " Sandiacre Chief Civil Engineer's",
        re.compile(' Sdg| Siding\\.'): ' Siding',
        re.compile(' Sdgs'): ' Sidings', re.compile(' Sdgs '): ' Sidings ',
        re.compile(' S[Ii][Gg]\\.? '): ' Signal ',
        re.compile('Sig\\.Ty357'): 'Signal TY357',
        re.compile(' Sth\\.? '): ' South ',
        re.compile(' South C\\.E\\.'): " South Civil Engineer's Sidings",
        re.compile(' S\\.S\\.'): ' Signal Box',
        re.compile(' Steetley Coy'): ' Steetley Company',
        re.compile(' Terml'): 'Terminal',
        re.compile(' T\\.C\\.'): ' Terminal Complex',
        re.compile(' Terminal\\.'): ' Terminal',
        re.compile(' T\\.?M\\.?D\\.?'): ' Traction Maintenance Depot',
        re.compile(' T\\.?&R\\.S\\.M\\.D\\.?'): ' Traction & Rolling Stock Maintenance Depot',
        re.compile(' T\\.C\\.'): ' Terminal Complex',
        re.compile(' U&Dgl'): ' Up & Down Goods Loop',
        re.compile(' U\\.G\\.L\\.'): ' Up Goods Loop',
        re.compile(' U\\.P\\.L\\.'): ' Up Passenger Loop',
        re.compile(' U\\.R\\.S\\.'): ' Up Relief Siding',
        re.compile(' Usn'): ' Up Sidings North',
        re.compile(' Uss'): ' Up Sidings South',
        re.compile(' \\(VWL\\)'): ' Victa Westlink Rail',
        re.compile(' W\\.R\\.D\\.'): ' Wagon Repair Depot',
        re.compile(' W Yard'): ' West Yard',
        re.compile('west533'): 'Westerton Signal YH533',
        re.compile(' Wks Lafarg'): ' Works Lafarg',
        re.compile(' TURNBACK'): ' Turnback Siding',
        re.compile(' Wtr Wh?f '): ' Water Wharf ',
        re.compile('Warrington C\\.E\\. Sidings'): "Warrington NCL/Civil Engineer's Sidings",
        re.compile(' Wm Csd\\.'): ' West Marina Carriage Servicing Depot',
        re.compile(' Yd '): ' Yard ',
        re.compile(' N\\.R\\.M\\.'): ' National Railway Museum'}
    if k is not None:
        return {k: loc_regexp_repl_dict}
    else:
        return loc_regexp_repl_dict


# Compare the difference between two columns and replace items if appropriate
def compare_and_replace(loc, to_replace, with_col):
    # Given length
    temp = loc[[to_replace, with_col]].applymap(len)
    replace_list = temp[temp[to_replace] <= temp[with_col]].index.tolist()
    loc[to_replace][replace_list] = loc[with_col][replace_list]


# ====================================================================================================================
""" Get table data from the NR_METEX database """


# Get primary keys of a table in database NR_METEX
def metex_pk(table_name):
    pri_key = db.get_pri_keys(db_name="NR_METEX", table_name=table_name)
    return pri_key


# Get Performance Event Code
def get_performance_event_code(update=False):
    filename = "performance_event_code"
    path_to_file = cdd_delay_attr(filename + ".pickle")
    if os.path.isfile(path_to_file) and not update:
        performance_event_code = load_pickle(path_to_file)
    else:
        try:
            performance_event_code = pd.read_excel(cdd_delay_attr("Historic delay attribution glossary.xlsx"),
                                                   sheetname="Performance Event Code")
            # Rename columns
            performance_event_code.columns = [x.replace(' ', '') for x in performance_event_code.columns]
            # Set an index
            performance_event_code.set_index('PerformanceEventCode', inplace=True)
            # Save the data as .pickle
            save_pickle(performance_event_code, path_to_file)
        except Exception as e:
            print("Getting '{}' ... failed due to '{}'.".format(filename, e))
            performance_event_code = None

    return performance_event_code


# Get Performance Event Code
def get_incident_reason_info_ref(update=False):
    path_to_file = cdd_delay_attr("incident_reason_info.pickle")
    if os.path.isfile(path_to_file) and not update:
        incident_reason_info = load_pickle(path_to_file)
    else:
        try:
            # Get data from the original glossary file
            path_to_file0 = cdd_delay_attr("Historic delay attribution glossary.xlsx")
            incident_reason_info = pd.read_excel(path_to_file0, sheetname="Incident Reason")
            incident_reason_info.columns = [x.replace(' ', '') for x in incident_reason_info.columns]
            incident_reason_info.set_index('IncidentReason', inplace=True)
            # Save the data
            save_pickle(incident_reason_info, path_to_file)
        except Exception as e:
            print("Getting '{}' ... failed due to '{}'.".format("incident_reason_info_ref", e))
            incident_reason_info = None
    return incident_reason_info


# Transform a DataFrame to dictionary
def group_items(data_frame, by, to_group, group_name, level=None, as_dict=False):
    # Create a dictionary
    temp_obj = data_frame.groupby(by, level=level)[to_group]
    d = {group_name: {k: list(v) for k, v in temp_obj}}
    if as_dict:
        return d
    else:
        d_df = pd.DataFrame(d)
        d_df.index.name = by
        return d_df


# ====================================================================================================================
""" Get table data from the NR_METEX database """


# Get IMDM
def get_imdm(as_dict=False, update=False):
    table_name = 'IMDM'
    path_to_file = cdd_metex_db_tables(table_name + ".pickle")

    if as_dict:
        path_to_file = path_to_file.replace(table_name, table_name + "_dict")

    if os.path.isfile(path_to_file) and not update:
        imdm = load_pickle(path_to_file)
    else:
        try:
            imdm = db.read_metex_table(table_name, index_col=metex_pk(table_name), save_as=".csv")
            imdm.index.rename(name='IMDM', inplace=True)  # Rename a column and index
            imdm.rename(columns={'Name': 'IMDM'}, inplace=True)
            if as_dict:
                imdm_dict = imdm.to_dict()
                imdm = imdm_dict['Route']
                imdm.pop('None', None)
            save_pickle(imdm, path_to_file)
        except Exception as e:
            print("Getting '{}' ... failed due to '{}'.".format(table_name, e))
            imdm = None

    return imdm


# Get ImdmAlias
def get_imdm_alias(as_dict=False, update=False):
    table_name = 'ImdmAlias'
    path_to_file = cdd_metex_db_tables(table_name + ".pickle")

    if as_dict:
        path_to_file = path_to_file.replace(table_name, table_name + "_dict")

    if os.path.isfile(path_to_file) and not update:
        imdm_alias = load_pickle(path_to_file)
    else:
        try:
            imdm_alias = db.read_metex_table(table_name, index_col=metex_pk(table_name), save_as=".csv")
            imdm_alias.rename(columns={'Imdm': 'IMDM'}, inplace=True)  # Rename a column
            imdm_alias.index.rename(name='ImdmAlias', inplace=True)  # Rename index
            if as_dict:
                imdm_alias_dict = imdm_alias.to_dict()
                imdm_alias = imdm_alias_dict['IMDM']
            save_pickle(imdm_alias, path_to_file)
        except Exception as e:
            print("Getting '{}' ... failed due to '{}'.".format(table_name, e))
            imdm_alias = None

    return imdm_alias


# Get IMDMWeatherCellMap
def get_imdm_weather_cell_map(grouped=False, update=False):
    table_name = 'IMDMWeatherCellMap'
    path_to_file = cdd_metex_db_tables(table_name + ".pickle")

    if grouped:
        path_to_file = path_to_file.replace(table_name, table_name + "_grouped")

    if os.path.isfile(path_to_file) and not update:
        weather_cell_map = load_pickle(path_to_file)
    else:
        try:
            # Read IMDMWeatherCellMap table
            weather_cell_map = db.read_metex_table(table_name, index_col=metex_pk(table_name), save_as=".csv")
            weather_cell_map.rename(columns={'WeatherCell': 'WeatherCellId'}, inplace=True)  # Rename a column
            weather_cell_map.index.rename('IMDMWeatherCellMapId', inplace=True)  # Rename index
            if grouped:  # Transform the dataframe into a dictionary-like form
                weather_cell_map = group_items(weather_cell_map, by='WeatherCellId', to_group='IMDM', group_name='IMDM')
            save_pickle(weather_cell_map, path_to_file)
        except Exception as e:
            print("Getting '{}' ... failed due to '{}'.".format(table_name, e))
            weather_cell_map = None

    return weather_cell_map


# Get IncidentReasonInfo
def get_incident_reason_info(database_plus=True, update=False):
    table_name = 'IncidentReasonInfo'
    path_to_file = cdd_metex_db_tables(table_name + ".pickle")
    if database_plus:
        path_to_file = path_to_file.replace(table_name, table_name + "_plus")

    if os.path.isfile(path_to_file) and not update:
        incident_reason_info = load_pickle(path_to_file)
    else:
        try:
            # Get data from the database
            incident_reason_info = db.read_metex_table(table_name, index_col=metex_pk(table_name), save_as=".csv")
            # Rename columns
            incident_reason_info.rename(columns={'Description': 'IncidentReasonDescription',
                                                 'Category': 'IncidentCategory',
                                                 'CategoryDescription': 'IncidentCategoryDescription'}, inplace=True)
            # Rename index label
            incident_reason_info.index.rename('IncidentReason', inplace=True)

            if database_plus:
                reason_info_plus = get_incident_reason_info_ref()
                incident_reason_info = reason_info_plus.join(incident_reason_info, how='outer', rsuffix='_orig')
                incident_reason_info.dropna(axis=1, inplace=True)

            save_pickle(incident_reason_info, path_to_file)

        except Exception as e:
            print("Getting '{}' ... failed due to '{}'.".format(table_name, e))
            incident_reason_info = None

    return incident_reason_info


# Get WeatherCategoryLookup
def get_weather_category_lookup(as_dict=False, update=False):
    table_name = 'WeatherCategoryLookup'
    path_to_file = cdd_metex_db_tables(table_name + ".pickle")

    if as_dict:
        path_to_file = path_to_file.replace(table_name, table_name + "_dict")

    if os.path.isfile(path_to_file) and not update:
        weather_category_lookup = load_pickle(path_to_file)
    else:
        try:
            weather_category_lookup = db.read_metex_table(table_name, index_col=metex_pk(table_name), save_as=".csv")
            # Rename a column and index label
            weather_category_lookup.rename(columns={'Name': 'WeatherCategory'}, inplace=True)
            weather_category_lookup.index.rename(name='WeatherCategoryCode', inplace=True)
            # Transform the DataFrame to a dictionary?
            if as_dict:
                weather_category_lookup = weather_category_lookup.to_dict()
            save_pickle(weather_category_lookup, path_to_file)
        except Exception as e:
            print("Getting '{}' ... failed due to '{}'.".format(table_name, e))
            weather_category_lookup = None

    return weather_category_lookup


# Get IncidentRecord and fill 'None' value with NaN
def get_incident_record(update=False):
    table_name = 'IncidentRecord'
    path_to_file = cdd_metex_db_tables(table_name + ".pickle")

    if os.path.isfile(path_to_file) and not update:
        incident_record = load_pickle(path_to_file)
    else:
        try:
            # Read the 'IncidentRecord' table
            incident_record = db.read_metex_table(table_name, index_col=metex_pk(table_name), save_as=".csv")
            # Rename column names
            incident_record.rename(columns={'CreateDate': 'IncidentRecordCreateDate',
                                            'Reason': 'IncidentReason'}, inplace=True)
            # Rename index name
            incident_record.index.rename('IncidentRecordId', inplace=True)
            # Get a weather category lookup dictionary
            weather_category_lookup = get_weather_category_lookup(as_dict=True)
            # Replace the weather category code with the corresponding full name
            incident_record.replace(weather_category_lookup, inplace=True)
            incident_record.fillna(value='', inplace=True)
            # Save the data
            save_pickle(incident_record, path_to_file)
        except Exception as e:
            print("Getting '{}' ... failed due to '{}'.".format(table_name, e))
            incident_record = None

    return incident_record


# Get Location
def get_location(update=False):
    table_name = 'Location'
    path_to_file = cdd_metex_db_tables(table_name + ".pickle")

    if os.path.isfile(path_to_file) and not update:
        location = load_pickle(path_to_file)
    else:
        try:
            # Read 'Location' table
            location = db.read_metex_table(table_name, index_col=metex_pk(table_name), save_as=".csv")
            # Rename a column and index label
            location.rename(columns={'Imdm': 'IMDM'}, inplace=True)
            location.index.rename('LocationId', inplace=True)
            # location['WeatherCell'].fillna(value='', inplace=True)
            location.WeatherCell = location.WeatherCell.apply(lambda x: '' if pd.np.isnan(x) else int(x))
            location.loc[610096, 0:4] = [-0.0751, 51.5461, -0.0751, 51.5461]
            # Save the data
            save_pickle(location, path_to_file)
        except Exception as e:
            print("Getting '{}' ... failed due to '{}'.".format(table_name, e))
            location = None

    return location


# Get PfPI (Process for Performance Improvement)
def get_pfpi(update=False):
    table_name = 'PfPI'
    path_to_file = cdd_metex_db_tables(table_name + ".pickle")

    if os.path.isfile(path_to_file) and not update:
        pfpi = load_pickle(path_to_file)
    else:
        try:
            # Read the 'PfPI' table
            pfpi = db.read_metex_table(table_name, index_col=metex_pk(table_name), save_as=".csv")
            # Rename a column name
            pfpi.index.rename('PfPIId', inplace=True)
            # To replace Performance Event Code
            performance_event_code = get_performance_event_code()
            # Merge pfpi and pe_code
            pfpi = pfpi.join(performance_event_code, on='PerformanceEventCode')
            # Change columns' order
            cols = pfpi.columns.tolist()
            pfpi = pfpi[cols[0:2] + cols[-2:] + cols[2:4]]
            # Save the data
            save_pickle(pfpi, path_to_file)
        except Exception as e:
            print("Getting '{}' ... failed due to '{}'.".format(table_name, e))
            pfpi = None

    return pfpi


# Get Route (Note that there is only one column in the original table)
def get_route(update=False):
    table_name = "Route"
    path_to_file = cdd_metex_db_tables(table_name + ".pickle")

    if os.path.isfile(path_to_file) and not update:
        route = load_pickle(path_to_file)
    else:
        try:
            route = db.read_metex_table(table_name, save_as=".csv")
            # Rename a column
            route.rename(columns={'Name': 'Route'}, inplace=True)
            # Save the processed data
            save_pickle(route, path_to_file)
        except Exception as e:
            print("Getting '{}' ... failed due to '{}'.".format(table_name, e))
            route = None

    return route


# Get StanoxLocation
def get_stanox_location(nr_mileage_format=True, update=False):
    table_name = 'StanoxLocation'
    path_to_file = cdd_metex_db_tables(table_name + ".pickle")

    if not nr_mileage_format:
        path_to_file = path_to_file.replace(table_name, table_name + "_miles")

    if os.path.isfile(path_to_file) and not update:
        stanox_location = load_pickle(path_to_file)
    else:
        try:
            # Read StanoxLocation table from the database
            stanox_location = db.read_metex_table(table_name, index_col=metex_pk(table_name), save_as=".csv")
            # Pre-cleaning the original data - replacing location names
            stanox_location.rename(columns={'Description': 'Location', 'Name': 'LocationAlias'}, inplace=True)

            # Create dictionaries for location names: {STANME: Location name}, {TIPLOC: Location name}
            key = 'Location'
            try:
                stanmes = load_json(cdd_rc("STANME.json"))
                stanme_dict = {key: stanmes}
            except FileNotFoundError:
                stanme_dict = rc.get_location_dictionary('STANME', drop_duplicates=True, main_key=key)
                save_json(stanme_dict[key], cdd_rc("STANME.json"))
            try:
                tiplocs = load_json(cdd_rc("TIPLOC.json"))
                tiploc_dict = {key: tiplocs}
            except FileNotFoundError:
                tiploc_dict = rc.get_location_dictionary('TIPLOC', drop_duplicates=True, main_key=key)
                save_json(tiploc_dict[key], cdd_rc("TIPLOC.json"))

            # Replace existing location names with likely full name
            loc = stanox_location[['Location', 'LocationAlias']].fillna('').replace(stanme_dict).replace(tiploc_dict)
            # Firstly, match 'STANME' and/or 'TIPLOC' -------------------------------
            loc['LocationAlias_upper'] = loc.LocationAlias.apply(lambda x: x.upper())
            loc['Location_full_1'] = loc.LocationAlias_upper.replace(stanme_dict[key]).replace(tiploc_dict[key])
            # Remove items of which all letters are upper-case from 'Location_full_2'
            loc['Location_full_1'] = loc.Location_full_1.apply(lambda x: '' if x.isupper() else x)
            compare_and_replace(loc, 'Location', 'Location_full_1')

            # Secondly, match 'STANOX' ----------------------------------------------
            try:
                stanox_dict = load_json(cdd_rc("STANOX.json"))
            except FileNotFoundError:
                stanox_dict = rc.get_location_dictionary('STANOX')
                save_json(stanox_dict, cdd_rc("STANOX.json"))
            loc['STANOX'] = loc.index
            loc['Location_full_2'] = loc.STANOX.replace(stanox_dict).apply(lambda x: '' if x.isdigit() else x)
            valid_data_idx = loc[loc.Location_full_2 != ''].index.tolist()
            loc['Location'][valid_data_idx] = loc.Location_full_2[valid_data_idx]
            loc.drop('STANOX', axis=1, inplace=True)

            # Thirdly, use loc_name_replacement_dict -------------------------------
            loc_name_replacement_dict = create_loc_name_replacement_dict(k='Location')
            loc.replace(loc_name_replacement_dict, inplace=True)

            # Fourthly, use an extra dictionary ------------------------------------------------------
            loc_name_regexp_replacement_dict = create_loc_name_regexp_replacement_dict(k='Location')
            loc.replace(loc_name_regexp_replacement_dict, regex=True, inplace=True)
            loc.drop(['Location_full_1', 'LocationAlias_upper', 'Location_full_2'], axis=1, inplace=True)

            # Finally, -----------------------------
            loc_cols = ['Location', 'LocationAlias']
            stanox_location = loc.join(stanox_location.drop(loc_cols, axis=1))
            stanox_location[loc_cols] = stanox_location[loc_cols].applymap(lambda x: x.strip(' '))

            # For 'ELR', replace NaN with ''
            stanox_location.ELR = stanox_location.ELR.fillna(value='')
            # For 'LocationId'
            stanox_location.LocationId = stanox_location.LocationId.map(lambda x: int(x) if not pd.np.isnan(x) else '')
            # For 'Mileages'
            if nr_mileage_format:
                yards = stanox_location.Yards.map(lambda x: yards_to_mileage(x) if not pd.isnull(x) else '')
            else:  # to convert yards to miles (Note: Not the 'mileage' used by Network Rail)
                yards = stanox_location.Yards.map(lambda x: x / 1760 if not pd.isnull(x) else '')
            stanox_location.Yards = yards
            stanox_location.rename(columns={'Yards': 'Mileage'}, inplace=True)

            # Revise '52053'
            stanox_location.loc['52053', 2:] = ['BOK1', '3.0792', 534877]
            # Revise '52074'
            stanox_location.loc['52074', 2:] = ['ELL1', '0.0440', 610096]

            save_pickle(stanox_location, path_to_file)
        except Exception as e:
            print("Getting '{}' ... failed due to '{}'.".format(table_name, e))
            stanox_location = None

    return stanox_location


# Get StanoxSection
def get_stanox_section(update=False):
    table_name = 'StanoxSection'
    path_to_file = cdd_metex_db_tables(table_name + ".pickle")

    if os.path.isfile(path_to_file) and not update:
        stanox_section = load_pickle(path_to_file)
    else:
        try:
            # Read StanoxSection table from the database
            stanox_section = db.read_metex_table(table_name, index_col=metex_pk(table_name), save_as=".csv")
            # Pre-cleaning the original data
            stanox_section.LocationId = stanox_section.LocationId.apply(lambda x: int(x) if not pd.np.isnan(x) else '')
            stanox_section.index.name = 'StanoxSectionId'
            # Firstly, create a stanox-to-location dictionary, and replace STANOX with location names
            stanox_loc = get_stanox_location(nr_mileage_format=True)
            stanox_dict_1 = stanox_loc['Location'].to_dict()
            stanox_dict_2 = rc.get_location_dictionary('STANOX', drop_duplicates=False)
            # Processing 'StartStanox'
            stanox_section['StartStanox_loc'] = stanox_section.StartStanox.replace(stanox_dict_1).replace(stanox_dict_2)
            # Processing 'EndStanox'
            stanox_section['EndStanox_loc'] = stanox_section.EndStanox.replace(stanox_dict_1).replace(stanox_dict_2)
            # Secondly, process 'STANME' and 'TIPLOC'
            stanme_dict = rc.get_location_dictionary('STANME')
            tiploc_dict = rc.get_location_dictionary('TIPLOC')
            loc_name_replacement_dict = create_loc_name_replacement_dict()
            loc_name_regexp_replacement_dict = create_loc_name_regexp_replacement_dict()
            # Processing 'StartStanox_loc'
            stanox_section.StartStanox_loc = stanox_section.StartStanox_loc. \
                replace(stanme_dict).replace(tiploc_dict). \
                replace(loc_name_replacement_dict).replace(loc_name_regexp_replacement_dict)
            # Processing 'EndStanox_loc'
            stanox_section.EndStanox_loc = stanox_section.EndStanox_loc. \
                replace(stanme_dict).replace(tiploc_dict). \
                replace(loc_name_replacement_dict).replace(loc_name_regexp_replacement_dict)
            # Create 'STANOX' sections
            start_end = stanox_section.StartStanox_loc + ' - ' + stanox_section.EndStanox_loc
            ploc_idx = stanox_section.StartStanox_loc == stanox_section.EndStanox_loc
            start_end[ploc_idx] = stanox_section.StartStanox_loc[ploc_idx]
            stanox_section['StanoxSection'] = start_end
            # Finalising the cleaning process
            stanox_section.drop('Description', axis=1, inplace=True)  # Drop original
            # Rename the columns of the start and end locations
            stanox_section.rename(columns={'StartStanox_loc': 'StanoxSection_Start',
                                           'EndStanox_loc': 'StanoxSection_End'}, inplace=True)
            # Reorder columns
            stanox_section = stanox_section[[
                'StanoxSection', 'StanoxSection_Start', 'StartStanox', 'StanoxSection_End', 'EndStanox',
                'LocationId', 'ApproximateLocation']]
            # Save the data
            save_pickle(stanox_section, path_to_file)
        except Exception as e:
            print("Getting '{}' ... failed due to '{}'.".format(table_name, e))
            stanox_section = None

    return stanox_section


# Get TrustIncident
def get_trust_incident(financial_years_06_14=True, update=False):
    table_name = 'TrustIncident'
    path_to_file = cdd_metex_db_tables(table_name + ".pickle")

    if financial_years_06_14:  # StartDate is between 01/04/2006 and 31/03/2014
        path_to_file = path_to_file.replace(table_name, table_name + "_06_14")

    if os.path.isfile(path_to_file) and not update:
        trust_incident = load_pickle(path_to_file)
    else:
        try:
            # Read 'TrustIncident' table
            trust_incident = db.read_metex_table(table_name, index_col=metex_pk(table_name), save_as=".csv")
            # Rename column names
            trust_incident.rename(columns={'Imdm': 'IMDM', 'Year': 'FinancialYear'}, inplace=True)
            # Rename index label
            trust_incident.index.name = 'TrustIncidentId'
            # Convert float to int values for 'SourceLocationId'
            trust_incident.SourceLocationId = \
                trust_incident.SourceLocationId.apply(lambda x: '' if pd.isnull(x) else int(x))
            # Retain data of which the StartDate is between 01/04/2006 and 31/03/2014?
            if financial_years_06_14:
                trust_incident = trust_incident[(trust_incident.FinancialYear >= 2006) &
                                                (trust_incident.FinancialYear <= 2014)]
            # Save the processed data
            save_pickle(trust_incident, path_to_file)
        except Exception as e:
            print("Getting '{}' ... failed due to '{}'.".format(table_name, e))
            trust_incident = None

    return trust_incident


# Get Weather
def get_weather(update=False):
    table_name = 'Weather'
    path_to_file = cdd_metex_db_tables(table_name + ".pickle")

    if os.path.isfile(path_to_file) and not update:
        weather_data = load_pickle(path_to_file)
    else:
        try:
            # Read 'Weather' table
            weather_data = db.read_metex_table(table_name, index_col=metex_pk(table_name))
            # Save original data read from the database (the file is too big)
            if not os.path.isfile(db.cdd_metex_db("Tables_original", table_name + ".csv")):
                save(weather_data, db.cdd_metex_db("Tables_original", table_name + ".csv"))
            # Firstly,
            i = 0
            snowfall, precipitation = weather_data.Snowfall.tolist(), weather_data.TotalPrecipitation.tolist()
            while i + 3 < len(weather_data):
                snowfall[i + 1: i + 3] = pd.np.linspace(snowfall[i], snowfall[i + 3], 4)[1:3]
                precipitation[i + 1: i + 3] = pd.np.linspace(precipitation[i], precipitation[i + 3], 4)[1:3]
                i += 3
            # Secondly,
            if i + 2 == len(weather_data):
                snowfall[-1:], precipitation[-1:] = snowfall[-2], precipitation[-2]
            elif i + 3 == len(weather_data):
                snowfall[-2:], precipitation[-2:] = [snowfall[-3]] * 2, [precipitation[-3]] * 2
            else:
                pass
            # Finally,
            weather_data.Snowfall = snowfall
            weather_data.TotalPrecipitation = precipitation
            # Save the processed data
            save_pickle(weather_data, path_to_file)
        except Exception as e:
            print("Getting '{}' ... failed due to '{}'.".format(table_name, e))
            weather_data = None

    return weather_data


# Get Weather data in a chunk-wise way
def get_weather_by_part(chunk_size=100000, index=True, save_as=None, save_by_chunk=False, save_by_value=False):
    """
    Note that it might be too large for pd.read_sql to read with low memory. Instead, we may read the 'Weather' table
    chunk-wise and assemble the full data set from individual pieces afterwards, especially when we'd like to save
    the data locally.
    """

    weather = db.read_table_by_part(db_name="NR_METEX",
                                    table_name="Weather",
                                    index_col=metex_pk("Weather") if index is True else None,
                                    parse_dates=None,
                                    chunk_size=chunk_size,
                                    save_as=save_as,
                                    save_by_chunk=save_by_chunk,
                                    save_by_value=save_by_value)

    return weather


# Get WeatherCell
def get_weather_cell(update=False, show_map=False, projection='tmerc', save_map_as=".png", dpi=600):
    table_name = 'WeatherCell'
    path_to_file = cdd_metex_db_tables(table_name + ".pickle")

    if os.path.isfile(path_to_file) and not update:
        weather_cell_map = load_pickle(path_to_file)
    else:
        try:
            # Read 'WeatherCell' table
            weather_cell_map = db.read_metex_table(table_name, index_col=metex_pk(table_name), save_as=".csv")
            weather_cell_map.index.rename('WeatherCellId', inplace=True)  # Rename index
            # Lower left corner:
            ll_longitude = weather_cell_map.Longitude  # - weather_cell_map.width / 2
            weather_cell_map['ll_Longitude'] = ll_longitude
            ll_latitude = weather_cell_map.Latitude  # - weather_cell_map.height / 2
            weather_cell_map['ll_Latitude'] = ll_latitude
            # Upper left corner:
            ul_lon = weather_cell_map.Longitude  # - cell['width'] / 2
            weather_cell_map['ul_lon'] = ul_lon
            ul_lat = weather_cell_map.Latitude + weather_cell_map.height  # / 2
            weather_cell_map['ul_lat'] = ul_lat
            # Upper right corner:
            ur_longitude = weather_cell_map.Longitude + weather_cell_map.width  # / 2
            weather_cell_map['ur_Longitude'] = ur_longitude
            ur_latitude = weather_cell_map.Latitude + weather_cell_map.height  # / 2
            weather_cell_map['ur_Latitude'] = ur_latitude
            # Lower right corner:
            lr_lon = weather_cell_map.Longitude + weather_cell_map.width  # / 2
            weather_cell_map['lr_lon'] = lr_lon
            lr_lat = weather_cell_map.Latitude  # - weather_cell_map.height / 2
            weather_cell_map['lr_lat'] = lr_lat
            # Get weather cell map
            imdm_weather_cell_map = get_imdm_weather_cell_map()
            # Get IMDM info
            imdm = get_imdm(as_dict=False)
            # Merge the acquired data set
            weather_cell_map = imdm_weather_cell_map.join(weather_cell_map, on='WeatherCellId').join(imdm, on='IMDM')
            # Save the processed data
            save_pickle(weather_cell_map, path_to_file)
        except Exception as e:
            print("Getting '{}' ... failed due to '{}'.".format(table_name, e))
            weather_cell_map = None

    # Plot the weather cells on the map?
    if show_map:
        print("Plotting the map ...", end="")

        plt.figure(figsize=(7, 8))

        ll = weather_cell_map[['ll_Longitude', 'll_Latitude']].apply(min).values
        ur = weather_cell_map[['ur_Longitude', 'ur_Latitude']].apply(max).values

        base_map = mpl_toolkits.basemap.Basemap(projection=projection,  # Transverse Mercator Projection
                                                lon_0=-2.,
                                                lat_0=49.,
                                                ellps='WGS84',
                                                epsg=27700,
                                                llcrnrlon=ll[0] - 0.285,  # -0.570409,  #
                                                llcrnrlat=ll[1] - 0.255,  # 51.23622,  #
                                                urcrnrlon=ur[0] + 0.285,  # 1.915975,  #
                                                urcrnrlat=ur[1] + 0.255,  # 53.062591,  #
                                                lat_ts=0,
                                                resolution='h',
                                                suppress_ticks=True)
        # base_map.drawmapboundary(color='none', fill_color='white')
        # base_map.drawcoastlines()
        # base_map.fillcontinents()
        base_map.arcgisimage(service='World_Shaded_Relief', xpixels=1500, dpi=300, verbose=False)

        weather_cell_map_plot = weather_cell_map.drop_duplicates([s for s in weather_cell_map.columns if '_' in s])

        for i in weather_cell_map_plot.index:
            ll_x, ll_y = base_map(weather_cell_map_plot['ll_Longitude'][i], weather_cell_map_plot['ll_Latitude'][i])
            ul_x, ul_y = base_map(weather_cell_map_plot['ul_lon'][i], weather_cell_map_plot['ul_lat'][i])
            ur_x, ur_y = base_map(weather_cell_map_plot['ur_Longitude'][i], weather_cell_map_plot['ur_Latitude'][i])
            lr_x, lr_y = base_map(weather_cell_map_plot['lr_lon'][i], weather_cell_map_plot['lr_lat'][i])
            xy = zip([ll_x, ul_x, ur_x, lr_x], [ll_y, ul_y, ur_y, lr_y])
            poly = matplotlib.patches.Polygon(list(xy), fc='#fff68f', ec='b', alpha=0.5)
            plt.gca().add_patch(poly)
        plt.tight_layout()

        # # Plot points
        # from shapely.geometry import Point, MultiPoint, Polygon, MultiPolygon
        # ll_lon_lat = [Point(m(lon, lat)) for lon, lat in zip(cell['ll_Longitude'], cell['ll_Latitude'])]
        # ur_lon_lat = [Point(m(lon, lat)) for lon, lat in zip(cell['ur_Longitude'], cell['ur_Latitude'])]
        # map_points = MultiPoint(ll_lon_lat + ur_lon_lat)
        # base_map.scatter([geom.x for geom in map_points], [geom.y for geom in map_points],
        #                  marker='x', s=16, lw=1, facecolor='#5a7b6c', edgecolor='w', label='Hazardous tress',
        #                  alpha=0.6, antialiased=True, zorder=3)
        #
        # # Plot squares
        # for i in range(len(cell)):
        #     ll_x, ll_y = base_map(cell['ll_Longitude'].iloc[i], cell['ll_Latitude'].iloc[i])
        #     ur_x, ur_y = base_map(cell['ur_Longitude'].iloc[i], cell['ur_Latitude'].iloc[i])
        #     ul_x, ul_y = base_map(cell['ul_lon'].iloc[i], cell['ul_lat'].iloc[i])
        #     lr_x, lr_y = base_map(cell['lr_lon'].iloc[i], cell['lr_lat'].iloc[i])
        #     base_map.plot([ll_x, ul_x], [ll_y, ul_y], color='#5a7b6c')
        #     base_map.plot([ul_x, ur_x], [ul_y, ur_y], color='#5a7b6c')
        #     base_map.plot([ur_x, lr_x], [ur_y, lr_y], color='#5a7b6c')
        #     base_map.plot([lr_x, ll_x], [lr_y, ll_y], color='#5a7b6c')

        print("Done.")
        # Save the map
        if save_map_as:
            plt.savefig(cdd_metex_db_fig(table_name + save_map_as), dpi=dpi)

    return weather_cell_map


# Get the lower-left and upper-right boundaries of weather cells
def get_weather_cell_map_boundary(route=None, adjusted=(0.285, 0.255)):
    # Get weather cell
    weather_cell = get_weather_cell()
    # For a certain Route?
    if route is not None:
        rte = find_match(route, get_route().Route.tolist())
        weather_cell = weather_cell[weather_cell.Route == rte]  # Select data for the specified route only
    ll = tuple(weather_cell[['ll_Longitude', 'll_Latitude']].apply(pd.np.min))
    lr = weather_cell.lr_lon.max(), weather_cell.lr_lat.min()
    ur = tuple(weather_cell[['ur_Longitude', 'ur_Latitude']].apply(pd.np.max))
    ul = weather_cell.ul_lon.min(), weather_cell.ul_lat.max()
    # Adjust (broaden) the boundaries?
    if adjusted:
        adj_values = pd.np.array(adjusted)
        ll = ll - adj_values
        lr = lr + (adj_values, -adj_values)
        ur = ur + adj_values
        ul = ul + (-adj_values, adj_values)
    return shapely.geometry.Polygon((ll, lr, ur, ul))


"""
update = True

get_performance_event_code(update=update)
get_incident_reason_info_ref(update=update)
get_imdm(as_dict=False, update=update)
get_imdm_alias(as_dict=False, update=update)
get_imdm_weather_cell_map(grouped=False, update=update)
get_incident_reason_info(database_plus=True, update=update)
get_incident_record(update=update)
get_location(update=update)
get_pfpi(update=update)
get_route(update=update)
get_stanox_location(nr_mileage_format=True, update=update)
get_stanox_section(update=update)
get_trust_incident(financial_years_06_14=True, update=update)
get_weather(update=update)
get_weather_category_lookup(as_dict=False, update=update)
get_weather_cell(update=update, show_map=True, projection='tmerc', save_map_as=".png", dpi=600)
"""


# ====================================================================================================================
""" Utils for creating views """


# Finalise the required data given 'route' and 'weather'
def subset(data, route=None, weather=None, reset_index=False):
    route_lookup = get_route()
    weather_category_lookup = get_weather_category_lookup()
    # Select data for a specific route and weather category
    if not route and not weather:
        dat = data.copy(deep=True)
    elif route and not weather:
        dat = data[data.Route == find_match(route, route_lookup.Route)]
    elif not route and weather:
        dat = data[data.WeatherCategory == find_match(weather, weather_category_lookup.WeatherCategory)]
    else:
        dat = data[(data.Route == find_match(route, route_lookup.Route)) &
                   (data.WeatherCategory == find_match(weather, weather_category_lookup.WeatherCategory))]
    # Reset index
    if reset_index:
        dat.reset_index(inplace=True)  # dat.index = range(len(dat))
    return dat


# Calculate the DelayMinutes and DelayCosts for grouped data
def agg_pfpi_stats(dat, selected_features, sort_by=None):
    data = dat.groupby(selected_features[1:-2]).aggregate({
        # 'IncidentId_and_CreateDate': {'IncidentCount': np.count_nonzero},
        'PfPIId': pd.np.count_nonzero,
        'PfPIMinutes': pd.np.sum,
        'PfPICosts': pd.np.sum})
    data.columns = ['IncidentCount', 'DelayMinutes', 'DelayCost']

    # data = dat.groupby(selected_features[1:-2]).aggregate({
    #     # 'IncidentId_and_CreateDate': {'IncidentCount': np.count_nonzero},
    #     'PfPIId': {'IncidentCount': np.count_nonzero},
    #     'PfPIMinutes': {'DelayMinutes': np.sum},
    #     'PfPICosts': {'DelayCost': np.sum}})
    # data.columns = data.columns.droplevel(0)

    data.reset_index(inplace=True)  # Reset the grouped indexes to columns
    if sort_by is not None:
        data.sort_values(sort_by, inplace=True)

    return data


# Form a file name in terms of specific 'Route' and 'weather' category
def make_filename(base_name, route, weather, *extra_suffixes, save_as=".pickle"):
    if route is not None:
        route_lookup = get_route()
        route = find_match(route, route_lookup.Route)
    if weather is not None:
        weather_category_lookup = get_weather_category_lookup()
        weather = find_match(weather, weather_category_lookup.WeatherCategory)
    filename_suffix = [s for s in (route, weather) if s is not None]  # "s" stands for "suffix"
    filename = "_".join([base_name] + filename_suffix + [str(s) for s in extra_suffixes]) + save_as
    return filename


# ====================================================================================================================
""" Get views based on the NR_METEX data """


# Retrieve the TRUST
def merge_schedule8_data(save_as=".pickle"):
    pfpi = get_pfpi()  # Get PfPI (260645, 6)
    incident_record = get_incident_record()  # (233452, 4)
    trust_incident = get_trust_incident(financial_years_06_14=True)  # (192054, 11)
    location = get_location()  # (228851, 6)
    imdm = get_imdm()  # (42, 1)
    incident_reason_info = get_incident_reason_info()  # (393, 7)
    stanox_location = get_stanox_location()  # (7560, 5)
    stanox_section = get_stanox_section()  # (9440, 7)

    # Merge the acquired data sets
    data = pfpi. \
        join(incident_record,  # (260645, 10)
             on='IncidentRecordId', how='inner'). \
        join(trust_incident,  # (260483, 21)
             on='TrustIncidentId', how='inner'). \
        join(stanox_section,  # (260483, 28)
             on='StanoxSectionId', how='inner'). \
        join(location,  # (260470, 34)
             on='LocationId', how='inner', lsuffix='', rsuffix='_Location'). \
        join(stanox_location,  # (260190, 39)
             on='StartStanox', how='inner', lsuffix='_Section', rsuffix=''). \
        join(stanox_location,  # (260140, 44)
             on='EndStanox', how='inner', lsuffix='_Start', rsuffix='_End'). \
        join(incident_reason_info,  # (260140, 51)
             on='IncidentReason', how='inner')  # .\
    # join(imdm, on='IMDM_Location', how='inner')  # (260140, 52)

    """
    There are "errors" in the IMDM data/column of the TrustIncident table.
    Not sure if the information about location id number is correct.
    """

    # Get a ELR-IMDM-Route "dictionary" from vegetation database
    route_du_elr = dbv.get_furlong_location(useful_columns_only=True)[['Route', 'ELR', 'DU']].drop_duplicates()
    route_du_elr.index = range(len(route_du_elr))  # (1276, 3)

    # Further cleaning the data
    data.reset_index(inplace=True)
    temp = data.merge(route_du_elr, how='left', left_on=['ELR_Start', 'IMDM'], right_on=['ELR', 'DU'])  # (260140, 55)
    temp = temp.merge(route_du_elr, how='left', left_on=['ELR_End', 'IMDM'], right_on=['ELR', 'DU'])  # (260140, 58)

    temp[['Route_', 'IMDM_']] = temp[['Route_x', 'DU_x']]
    idx_x = (temp.Route_x.isnull()) & (~temp.Route_y.isnull())
    temp.loc[idx_x, 'Route_'], temp.loc[idx_x, 'IMDM_'] = temp.Route_y.loc[idx_x], temp.DU_y.loc[idx_x]

    idx_y = (temp.Route_x.isnull()) & (temp.Route_y.isnull())
    temp.loc[idx_y, 'IMDM_'] = temp.IMDM_Location.loc[idx_y]
    temp.loc[idx_y, 'Route_'] = temp.loc[idx_y, 'IMDM_'].replace(imdm.to_dict()['Route'])

    temp.drop(labels=['Route_x', 'ELR_x', 'DU_x', 'Route_y', 'ELR_y', 'DU_y',
                      'StanoxSection_Start', 'StanoxSection_End',
                      'IMDM', 'IMDM_Location'], axis=1, inplace=True)  # (260140, 50)

    data = temp.rename(columns={'Location_Start': 'StartLocation',
                                'Location_End': 'EndLocation',
                                'LocationAlias_Start': 'StartLocationAlias',
                                'LocationAlias_End': 'EndLocationAlias',
                                'ELR_Start': 'StartELR',
                                'ELR_End': 'EndELR',
                                'Mileage_Start': 'StartMileage',
                                'Mileage_End': 'EndMileage',
                                'LocationId_Start': 'StartLocationId',
                                'LocationId_End': 'EndLocationId',
                                'LocationId_Section': 'SectionLocationId',
                                'Route_': 'Route',
                                'IMDM_': 'IMDM'})  # (260140, 50)

    idx = data.StartLocation == 'Highbury & Islington (North London Lines)'
    data.loc[idx, ['StartLongitude', 'StartLatitude']] = [-0.1045, 51.5460]
    idx = data.EndLocation == 'Highbury & Islington (North London Lines)'
    data.loc[idx, ['EndLongitude', 'EndLatitude']] = [-0.1045, 51.5460]
    idx = data.StartLocation == 'Dalston Junction (East London Line)'
    data.loc[idx, ['StartLongitude', 'StartLatitude']] = [-0.0751, 51.5461]
    idx = data.EndLocation == 'Dalston Junction (East London Line)'
    data.loc[idx, ['EndLongitude', 'EndLatitude']] = [-0.0751, 51.5461]

    data.EndELR.replace({'STM': 'SDC', 'TIR': 'TLL'}, inplace=True)

    # Sort the merged data frame by index 'PfPIId'
    data = data.set_index('PfPIId').sort_index()  # (260140, 49)

    # Further cleaning the 'IMDM' and 'Route'
    for section in data.StanoxSection.unique():
        temp = data[data.StanoxSection == section]
        # IMDM
        if len(temp.IMDM.unique()) >= 2:
            imdm_temp = data.loc[temp.index].IMDM.value_counts()
            data.loc[temp.index, 'IMDM'] = imdm_temp[imdm_temp == imdm_temp.max()].index[0]
        # Route
        if len(temp.Route.unique()) >= 2:
            route_temp = data.loc[temp.index].Route.value_counts()
            data.loc[temp.index, 'Route'] = route_temp[route_temp == route_temp.max()].index[0]

    if save_as:
        filename = make_filename("Schedule8_details", route=None, weather=None, save_as=save_as)
        save_pickle(data, cdd_metex_db_views(filename))

    # Return the DataFrame
    return data


# Get the TRUST data
def get_schedule8_details(route=None, weather=None, reset_index=False, update=False):
    filename = make_filename("Schedule8_details", route, weather)
    path_to_file = cdd_metex_db_views(filename)

    if os.path.isfile(path_to_file) and not update:
        schedule8_details = load_pickle(path_to_file)
        if reset_index:
            schedule8_details.reset_index(inplace=True)
    else:
        try:
            schedule8_details = merge_schedule8_data(save_as=".pickle")
            schedule8_details = subset(schedule8_details, route, weather, reset_index)
            save_pickle(schedule8_details, path_to_file)
        except Exception as e:
            print("Getting '{}' ... failed due to '{}'.".format(os.path.splitext(filename)[0], e))
            schedule8_details = None

    return schedule8_details


# Essential details about incidents
def get_schedule8_details_pfpi(route=None, weather=None, update=False):
    filename = make_filename("Schedule8_details_pfpi", route, weather)
    path_to_file = cdd_metex_db_views(filename)

    if os.path.isfile(path_to_file) and not update:
        data = load_pickle(path_to_file)
    else:
        try:
            # Get merged data sets
            schedule8_data = get_schedule8_details(route, weather, reset_index=True)
            # Select a list of columns
            selected_features = [
                'PfPIId',  # 260140
                'IncidentRecordId',  # 232978
                'TrustIncidentId',  # 191759
                'IncidentNumber',  # 176370
                'PerformanceEventCode', 'PerformanceEventGroup', 'PerformanceEventName',
                'PfPIMinutes', 'PfPICosts', 'FinancialYear',  # 9
                'IncidentRecordCreateDate',  # 3287
                'StartDate', 'EndDate',
                'IncidentDescription', 'IncidentJPIPCategory',
                'WeatherCategory',  # 10
                'IncidentReason', 'IncidentReasonDescription', 'IncidentCategory', 'IncidentCategoryDescription',
                'IncidentCategoryGroupDescription', 'IncidentFMS', 'IncidentEquipment',
                'WeatherCell',  # 106
                'Route', 'IMDM',
                'StanoxSection', 'StartLocation', 'EndLocation',
                'StartELR', 'StartMileage', 'EndELR', 'EndMileage', 'StartStanox', 'EndStanox',
                'StartLongitude', 'StartLatitude', 'EndLongitude', 'EndLatitude',
                'ApproximateLocation']
            # Acquire the subset (260140, 40)
            data = schedule8_data[selected_features]
            save_pickle(data, path_to_file)
        except Exception as e:
            print("Getting '{}' ... failed due to '{}'.".format(os.path.splitext(filename)[0], e))
            data = None

    return data


# Schedule 8 details combined with weather data
def get_schedule8_details_and_weather(route=None, weather=None, ip_start_hrs=-12, ip_end_hrs=12, update=False):
    """
    :param route:
    :param weather:
    :param ip_start_hrs: [numeric] incident period start time, i.e. hours before the recorded incident start
    :param ip_end_hrs: [numeric] incident period end time, i.e. hours after the recorded incident end time
    :param update: [bool] default, False
    :return:
    """

    filename = make_filename("Schedule8_details_and_weather", route, weather)
    add_suffix = [str(s) for s in (ip_start_hrs, ip_end_hrs)]
    filename = "_".join([filename] + add_suffix) + ".pickle"
    path_to_file = cdd_metex_db_views(filename)

    if os.path.isfile(path_to_file) and not update:
        data = load_pickle(path_to_file)
    else:
        try:
            # Getting Schedule 8 details (i.e. 'Schedule8_details')
            schedule8_details = get_schedule8_details(route, weather, reset_index=False)
            # Truncates "month" and "time" parts from datetime
            schedule8_details['incident_duration'] = \
                schedule8_details.EndDate - schedule8_details.StartDate
            schedule8_details['critical_start'] = \
                schedule8_details.StartDate.apply(datetime_truncate.truncate_hour) + \
                pd.DateOffset(hours=ip_start_hrs)
            schedule8_details['critical_end'] = \
                schedule8_details.EndDate.apply(datetime_truncate.truncate_hour) + \
                pd.DateOffset(hours=ip_end_hrs)
            schedule8_details['critical_period'] = \
                schedule8_details.critical_end - schedule8_details.critical_start
            # Get weather data
            weather_data = get_weather()
            # Merge the two data sets
            data = schedule8_details.join(weather_data, on=['WeatherCell', 'critical_start'], how='inner')
            data.sort_index(inplace=True)  # (257608, 61)
            # Save the merged data
            save_pickle(data, path_to_file)

        except Exception as e:
            print("Getting '{}' ... failed due to '{}'.".format(os.path.splitext(filename)[0], e))
            data = None

    return data


# Get Schedule 8 data by incident location and weather category
def get_schedule8_costs_by_location(route=None, weather=None, update=False):
    """
    :param route: 
    :param weather: 
    :param update: 
    :return: 
    """

    filename = make_filename("Schedule8_costs_by_location", route, weather)
    path_to_file = cdd_metex_db_views(filename)

    if os.path.isfile(path_to_file) and not update:
        data = load_pickle(path_to_file)
    else:
        try:
            # Get Schedule8_details
            schedule8_details = get_schedule8_details(route, weather, reset_index=True)
            # Select columns
            selected_features = [
                'PfPIId',
                # 'TrustIncidentId', 'IncidentRecordCreateDate',
                'WeatherCategory',
                'Route', 'IMDM',
                'StanoxSection',
                'StartLocation', 'EndLocation',
                'StartELR', 'StartMileage', 'EndELR', 'EndMileage',
                'StartStanox', 'EndStanox',
                'StartLongitude', 'StartLatitude', 'EndLongitude', 'EndLatitude',
                'PfPIMinutes', 'PfPICosts']
            schedule8_data = schedule8_details[selected_features]
            data = agg_pfpi_stats(schedule8_data, selected_features)
            save_pickle(data, path_to_file)
        except Exception as e:
            print("Getting '{}' ... failed due to {}.".format(os.path.splitext(filename)[0], e))
            data = None

    return data


# Get Schedule 8 data by datetime and weather category
def get_schedule8_costs_by_datetime(route=None, weather=None, update=False):
    filename = make_filename("Schedule8_costs_by_datetime", route, weather)
    path_to_file = cdd_metex_db_views(filename)

    if os.path.isfile(path_to_file) and not update:
        data = load_pickle(path_to_file)
    else:
        try:
            # Get Schedule8_details
            schedule8_details = get_schedule8_details(route, weather, reset_index=True)
            # Select a list of columns
            selected_features = [
                'PfPIId',
                # 'TrustIncidentId', 'IncidentRecordCreateDate',
                'FinancialYear',
                'StartDate', 'EndDate',
                'WeatherCategory',
                'Route', 'IMDM',
                'WeatherCell',
                'PfPICosts', 'PfPIMinutes']
            schedule8_data = schedule8_details[selected_features]
            data = agg_pfpi_stats(schedule8_data, selected_features, sort_by=['StartDate', 'EndDate'])
            save_pickle(data, path_to_file)
        except Exception as e:
            print("Getting '{}' ... failed due to '{}'.".format(os.path.splitext(filename)[0], e))
            data = None

    return data


# Get Schedule 8 data by datetime and location
def get_schedule8_costs_by_datetime_location(route=None, weather=None, update=False):
    filename = make_filename("Schedule8_costs_by_datetime_location", route, weather)
    path_to_file = cdd_metex_db_views(filename)

    if os.path.isfile(path_to_file) and not update:
        data = load_pickle(path_to_file)
    else:
        try:
            # Get merged data sets
            schedule8_details = get_schedule8_details(route, weather, reset_index=True)
            # Select a list of columns
            selected_features = [
                'PfPIId',
                # 'TrustIncidentId', 'IncidentRecordCreateDate',
                'FinancialYear',
                'StartDate', 'EndDate',
                'WeatherCategory',
                'StanoxSection',
                'Route', 'IMDM',
                'StartLocation', 'EndLocation',
                'StartStanox', 'EndStanox',
                'StartELR', 'StartMileage', 'EndELR', 'EndMileage',
                'StartLongitude', 'StartLatitude', 'EndLongitude', 'EndLatitude',
                'WeatherCell',
                'PfPICosts', 'PfPIMinutes']
            schedule8_data = schedule8_details[selected_features]
            data = agg_pfpi_stats(schedule8_data, selected_features, sort_by=['StartDate', 'EndDate'])
            save_pickle(data, path_to_file)
        except Exception as e:
            print("Getting '{}' ... failed due to '{}'.".format(os.path.splitext(filename)[0], e))
            data = None

    return data


# Get Schedule 8 data by datetime, location and weather
def get_schedule8_costs_by_datetime_location_weather(route=None, weather=None, ip_start=-12, ip_end=12, update=False):
    filename = make_filename("Schedule8_costs_by_datetime_location_weather", route, weather)
    add_suffix = [str(s) for s in (ip_start, ip_end)]
    filename = "_".join([filename] + add_suffix) + ".pickle"
    path_to_file = cdd_metex_db_views(filename)

    if os.path.isfile(path_to_file) and not update:
        data = load_pickle(path_to_file)
    else:
        try:
            # Get Schedule8_costs_by_datetime_location
            schedule8_data = get_schedule8_costs_by_datetime_location(route, weather)
            # Create critical start and end datetimes (truncating "month" and "time" parts from datetime)
            schedule8_data['incident_duration'] = \
                schedule8_data.EndDate - schedule8_data.StartDate
            schedule8_data['critical_start'] = \
                schedule8_data.StartDate.apply(datetime_truncate.truncate_hour) + \
                pd.DateOffset(hours=ip_start)
            schedule8_data['critical_end'] = \
                schedule8_data.EndDate.apply(datetime_truncate.truncate_hour) + \
                pd.DateOffset(hours=ip_end)
            schedule8_data['critical_period'] = \
                schedule8_data.critical_end - schedule8_data.critical_start
            # Get weather data
            weather_data = get_weather()
            # Merge the two data sets
            data = schedule8_data.join(weather_data, on=['WeatherCell', 'critical_start'], how='inner')
            # Save the merged data
            save_pickle(data, path_to_file)
        except Exception as e:
            print("Getting '{}' ... failed due to '{}'.".format(os.path.splitext(filename)[0], e))
            data = None

    return data


# Get Schedule 8 cost by incident reason
def get_schedule8_cost_by_reason(route=None, weather=None, update=False):
    filename = make_filename("Schedule8_costs_by_reason", route, weather)
    path_to_file = cdd_metex_db_views(filename)

    if os.path.isfile(path_to_file) and not update:
        data = load_pickle(path_to_file)
    else:
        try:
            # Get merged data sets
            schedule8_details = get_schedule8_details(route, weather, reset_index=True)
            # Select columns
            selected_features = [
                'PfPIId',
                'FinancialYear',
                'Route',
                # 'IMDM',
                'WeatherCategory',
                'IncidentCategory',
                'IncidentCategoryDescription',
                'IncidentReason',
                'IncidentReasonName',
                'IncidentReasonDescription',
                'PfPIMinutes', 'PfPICosts']
            schedule8_data = schedule8_details[selected_features]
            data = agg_pfpi_stats(schedule8_data, selected_features)
            save_pickle(data, path_to_file)
        except Exception as e:
            print("Getting '{}' ... failed due to '{}'.".format(os.path.splitext(filename)[0], e))
            data = None

    return data


# Get Schedule 8 cost by location and incident reason
def get_schedule8_cost_by_location_reason(route=None, weather=None, update=False):
    filename = make_filename("Schedule8_costs_by_location_reason", route, weather)
    path_to_file = cdd_metex_db_views(filename)

    if os.path.isfile(path_to_file) and not update:
        data = load_pickle(path_to_file)
    else:
        try:
            schedule8_details = get_schedule8_details(route, weather).reset_index()
            selected_features = [
                'PfPIId',
                'FinancialYear',
                'WeatherCategory',
                'Route', 'IMDM',
                'StanoxSection',
                'StartStanox', 'EndStanox',
                'StartLocation', 'EndLocation',
                'StartELR', 'StartMileage', 'EndELR', 'EndMileage',
                'StartLongitude', 'StartLatitude', 'EndLongitude', 'EndLatitude',
                'IncidentReason',
                'IncidentReasonName',
                'IncidentReasonDescription',
                'IncidentCategory',
                'IncidentCategoryDescription',
                'PfPIMinutes', 'PfPICosts']
            schedule8_data = schedule8_details[selected_features]
            data = agg_pfpi_stats(schedule8_data, selected_features)
            save_pickle(data, path_to_file)
        except Exception as e:
            print("Getting '{}' ... failed due to '{}'.".format(os.path.splitext(filename)[0], e))
            data = None

    return data


# Get Schedule 8 cost by datetime, location and incident reason
def get_schedule8_cost_by_datetime_location_reason(route=None, weather=None, update=False):
    filename = make_filename("Schedule8_costs_by_datetime_location_reason", route, weather)
    path_to_file = cdd_metex_db_views(filename)

    if os.path.isfile(path_to_file) and not update:
        data = load_pickle(path_to_file)
    else:
        try:
            schedule8_details = get_schedule8_details(route, weather, reset_index=True)
            selected_features = [
                'PfPIId',
                'FinancialYear',
                'StartDate', 'EndDate',
                'WeatherCategory',
                'WeatherCell',
                'Route', 'IMDM',
                'StanoxSection',
                'StartLocation', 'EndLocation',
                'StartStanox', 'EndStanox',
                'StartELR', 'StartMileage', 'EndELR', 'EndMileage',
                'StartLongitude', 'StartLatitude', 'EndLongitude', 'EndLatitude',
                'IncidentReason',
                'IncidentReasonName',
                'IncidentReasonDescription',
                'IncidentCategory',
                'IncidentCategoryDescription',
                'IncidentCategoryGroupDescription',
                'IncidentJPIPCategory',
                'PfPIMinutes', 'PfPICosts']
            schedule8_data = schedule8_details[selected_features]
            data = agg_pfpi_stats(schedule8_data, selected_features)
            save_pickle(data, path_to_file)
        except Exception as e:
            print("Getting '{}' ... failed due to '{}'.".format(os.path.splitext(filename)[0], e))
            data = None

    return data


# Get Schedule 8 cost by weather category
def get_schedule8_cost_by_weathercategory(route=None, weather=None, update=False):
    filename = make_filename("Schedule8_costs_by_weathercategory", route, weather)
    path_to_file = cdd_metex_db_views(filename)

    if os.path.isfile(path_to_file) and not update:
        data = load_pickle(path_to_file)
    else:
        try:
            schedule8_details = get_schedule8_details(route, weather, reset_index=True)
            selected_features = ['PfPIId', 'FinancialYear',
                                 'Route', 'IMDM', 'WeatherCategory',
                                 'PfPICosts', 'PfPIMinutes']
            schedule8_data = schedule8_details[selected_features]
            data = agg_pfpi_stats(schedule8_data, selected_features)
            save_pickle(data, path_to_file)
        except Exception as e:
            print("Getting '{}' ... failed due to '{}'.".format(os.path.splitext(filename)[0], e))
            data = None

    return data


"""
route = None
weather = None
update = True

get_schedule8_details(route, weather, reset_index=False, update=update)
get_schedule8_details_pfpi(route, weather, update)
get_schedule8_details_and_weather(route, weather, -12, 12, update=update)
get_schedule8_costs_by_location(route, weather, update=update)
get_schedule8_costs_by_datetime(route, weather, update=update)
get_schedule8_costs_by_datetime_location(route, weather, update=update)
get_schedule8_costs_by_datetime_location_weather(route, weather, -12, 12, update=update)
get_schedule8_cost_by_reason(route, weather, update=update)
get_schedule8_cost_by_location_reason(route, weather, update=update)
get_schedule8_cost_by_datetime_location_reason(route, weather, update=update)
get_schedule8_cost_by_weathercategory(route, weather, update=update)
"""
