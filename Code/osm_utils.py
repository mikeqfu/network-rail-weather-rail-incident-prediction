""" OpenStreetMap Data Extracts - Geofabrik downloads """

import glob
import itertools
import os
import re
import shutil
import time
import urllib.error
import urllib.request
import zipfile

import bs4
import fuzzywuzzy.process
import geopandas as gpd
import humanfriendly
import numpy as np
import pandas as pd
import progressbar
import requests
import shapefile
import shapely.geometry

from utils import cdd, save_pickle, load_pickle, save_json

# ====================================================================================================================
""" Change directory """


# Change directory to "Generic\\Data\\osm-util" and sub-directories
def cdd_osm(*directories):
    path = os.path.join(os.path.dirname(os.getcwd()), "Generic\\Data\\osm-pyutils")
    for directory in directories:
        path = os.path.join(path, directory)
    return path


# Change directory to "Generic\\Data\\osm-util\\dat" and sub-directories
def cdd_osm_dat(*directories):
    path = cdd_osm('dat')
    for directory in directories:
        path = os.path.join(path, directory)
    return path


# Change directory to "Generic\\Data\\osm-util\\dat_GeoFabrik" and sub-directories
def cdd_osm_dat_geofabrik(*directories):
    path = cdd_osm('dat_GeoFabrik')
    for directory in directories:
        path = os.path.join(path, directory)
    return path


# ====================================================================================================================
""" Scrape information from the OSM website """


# Get raw directory index (allowing us to see and download older files)
def get_raw_directory_index(url):
    """
    :param url: 
    :return: 
    """
    try:
        raw_directory_index = pd.read_html(url, match='file', header=0, parse_dates=['date'])
        raw_directory_index = pd.DataFrame(pd.concat(raw_directory_index, axis=0, ignore_index=True))
        raw_directory_index.columns = [c.title() for c in raw_directory_index.columns]

        # Clean the DataFrame a little bit
        raw_directory_index.Size = raw_directory_index.Size.apply(humanfriendly.format_size)
        raw_directory_index.sort_values('Date', ascending=False, inplace=True)
        raw_directory_index.index = range(len(raw_directory_index))

        raw_directory_index['FileURL'] = raw_directory_index.File.map(lambda x: urllib.request.urljoin(url, x))

    except (urllib.error.HTTPError, TypeError, ValueError):
        raw_directory_index = None
        if len(urllib.request.urlparse(url).path) <= 1:
            print("The home page does not have a raw directory index.")

    return raw_directory_index


# Get a table for a given URL, which contains all available URLs for each subregion and its file downloading
def get_subregion_url_table(url):
    """
    :param url: 
    :return: 
    """
    try:
        subregion_table = pd.read_html(url, match=re.compile('(Special )?Sub[ \-]Regions?'), skiprows=[0, 1],
                                       encoding='UTF-8')
        subregion_table = pd.DataFrame(pd.concat(subregion_table, axis=0, ignore_index=True))

        # Specify column names
        file_types = ['.osm.pbf', '.shp.zip', '.osm.bz2']
        column_names = ['Subregion'] + file_types
        column_names.insert(2, '.osm.pbf_Size')

        # Add column/names
        if len(subregion_table.columns) == 4:
            subregion_table.insert(2, '.osm.pbf_Size', np.nan)
        subregion_table.columns = column_names

        subregion_table.replace({'.osm.pbf_Size': {re.compile('[()]'): '', re.compile('\xa0'): ' '}}, inplace=True)

        # Get the URLs
        source = requests.get(url)
        soup = bs4.BeautifulSoup(source.text, 'lxml')
        source.close()

        for file_type in file_types:
            text = '[{}]'.format(file_type)
            urls = [urllib.request.urljoin(url, link['href']) for link in soup.find_all(name='a', href=True, text=text)]
            subregion_table.loc[subregion_table[file_type].notnull(), file_type] = urls

        try:
            subregion_urls = [urllib.request.urljoin(url, soup.find('a', text=text)['href'])
                              for text in subregion_table.Subregion]
        except TypeError:
            subregion_urls = [kml['onmouseover'] for kml in soup.find_all('tr', onmouseover=True)]
            subregion_urls = [s[s.find('(') + 1:s.find(')')][1:-1].replace('kml', 'html') for s in subregion_urls]
            subregion_urls = [urllib.request.urljoin(url, sub_url) for sub_url in subregion_urls]
        subregion_table['SubregionURL'] = subregion_urls

        column_names = list(subregion_table.columns)
        column_names.insert(1, column_names.pop(len(column_names) - 1))
        subregion_table = subregion_table[column_names]  # .fillna(value='')

    except (ValueError, TypeError, ConnectionRefusedError, ConnectionError):
        # No more data available for subregions within the region
        print("Checked out \"{}\".".format(url.split('/')[-1].split('.')[0].title()))
        subregion_table = None

    return subregion_table


# Scan through the downloading pages to get a list of available subregion names
def scrape_available_subregion_indices():
    home_url = 'http://download.geofabrik.de/'

    try:
        source = requests.get(home_url)
        soup = bs4.BeautifulSoup(source.text, 'lxml')
        avail_subregions = [td.a.text for td in soup.find_all('td', {'class': 'subregion'})]
        avail_subregion_urls = [urllib.request.urljoin(home_url, td.a['href'])
                                for td in soup.find_all('td', {'class': 'subregion'})]
        avail_subregion_url_tables = [get_subregion_url_table(sub_url) for sub_url in avail_subregion_urls]
        avail_subregion_url_tables = [tbl for tbl in avail_subregion_url_tables if tbl is not None]

        subregion_url_tables = list(avail_subregion_url_tables)

        while subregion_url_tables:

            subregion_url_tables_1 = []

            for subregion_url_table in subregion_url_tables:
                subregions = list(subregion_url_table.Subregion)
                subregion_urls = list(subregion_url_table.SubregionURL)
                subregion_url_tables_0 = [get_subregion_url_table(subregion_url) for subregion_url in subregion_urls]
                subregion_url_tables_1 += [tbl for tbl in subregion_url_tables_0 if tbl is not None]

                # (Note that 'Russian Federation' data is available in both 'Asia' and 'Europe')
                avail_subregions += subregions
                avail_subregion_urls += subregion_urls
                avail_subregion_url_tables += subregion_url_tables_1

            subregion_url_tables = list(subregion_url_tables_1)

        # Save a list of available subregions locally
        save_pickle(avail_subregions, cdd_osm_dat("subregion-index.pickle"))

        # Subregion index - {Subregion: URL}
        subregion_url_index = dict(zip(avail_subregions, avail_subregion_urls))
        # Save subregion_index to local disk
        save_pickle(subregion_url_index, cdd_osm_dat("subregion-url-index.pickle"))
        save_json(subregion_url_index, cdd_osm_dat("subregion-url-index.json"))

        # All available URLs for downloading
        home_subregion_url_table = get_subregion_url_table(home_url)
        avail_subregion_url_tables.append(home_subregion_url_table)
        subregion_downloads_index = pd.DataFrame(pd.concat(avail_subregion_url_tables, ignore_index=True))
        subregion_downloads_index.drop_duplicates(inplace=True)

        # Save subregion_index_downloads to loacal disk
        save_pickle(subregion_downloads_index, cdd_osm_dat("subregion-downloads-index.pickle"))
        subregion_downloads_index.set_index('Subregion').to_json(cdd_osm_dat("subregion-downloads-index.json"))

    except Exception as e:
        print(e)


# Get a list of available subregion names
def get_subregion_index(index_filename="subregion-index", update=False):
    """
    :param index_filename: 
    :param update: 
    :return: 
    """
    available_index = ("subregion-index", "subregion-url-index", "subregion-downloads-index")
    if index_filename not in available_index:
        print("Error: 'index_filename' must be chosen from among {}.".format(available_index))
        index = None
    else:
        indices_filename = ["subregion-index.pickle",
                            "subregion-url-index.pickle", "subregion-url-index.json",
                            "subregion-downloads-index.pickle", "subregion-downloads-index.json"]
        paths_to_files_exist = [os.path.isfile(cdd_osm_dat(f)) for f in indices_filename]
        path_to_index_file = cdd_osm_dat(index_filename + ".pickle")
        if all(paths_to_files_exist) and not update:
            index = load_pickle(path_to_index_file)
        else:
            try:
                scrape_available_subregion_indices()
                index = load_pickle(path_to_index_file)
            except Exception as e:
                print("Update failed due to {}. The existing data file would be loaded instead.".format(e))
                index = None
    return index


# Specify the names of the existing layer
def available_layers():
    layer_names = ['buildings',
                   'landuse',
                   'natural',
                   'places',
                   'pofw',
                   'pois',
                   'railways',
                   'roads',
                   'traffic',
                   'transport',
                   'water',
                   'waterways']
    return layer_names


# Search the OSM directory and its sub-directories to get the path to the file
def fetch_osm_file(subregion, layer, feature=None, file_format=".shp", update=False):
    subregion_index = get_subregion_index("subregion-index", update)
    subregion_name = fuzzywuzzy.process.extractOne(subregion, subregion_index, score_cutoff=10)[0]
    subregion = subregion_name.lower().replace(" ", "-")
    osm_file_path = []

    for dirpath, dirnames, filenames in os.walk(cdd_osm_dat_geofabrik()):
        if feature is None:
            for fname in [f for f in filenames
                          if (layer + "_a" in f or layer + "_free" in f) and f.endswith(file_format)]:
                if subregion in os.path.basename(dirpath) and dirnames == []:
                    osm_file_path.append(os.path.join(dirpath, fname))
        else:
            for fname in [f for f in filenames if layer + "_" + feature in f and f.endswith(file_format)]:
                if subregion not in os.path.dirname(dirpath) and dirnames == []:
                    osm_file_path.append(os.path.join(dirpath, fname))
    # if len(osm_file_path) > 1:
    #     osm_file_path = [p for p in osm_file_path if "_a_" not in p]
    return osm_file_path


# Get download URL
def get_download_url(subregion, file_format=".shp.zip", update=False):
    """
    :param subregion: [str] case-insensitive, e.g. 'Greater London'
    :param file_format: '.osm.pbf', '.shp.zip', '.osm.bz2'
    :param update: [bool]
    :return: 
    """
    # Get a list of available
    subregion_index = get_subregion_index('subregion-index', update=update)
    subregion_name = fuzzywuzzy.process.extractOne(subregion, subregion_index, score_cutoff=10)[0]
    # Get an index of download URLs
    subregion_downloads_index = get_subregion_index('subregion-downloads-index', update=update).set_index('Subregion')
    # Get the URL
    download_url = subregion_downloads_index.loc[subregion_name, file_format]
    return subregion_name, download_url


# Parse the download URL so as to specify a path for storing the downloaded file
def make_file_path(download_url):
    """
    :param download_url: 
    :return: 
    """
    parsed_path = os.path.normpath(urllib.request.urlparse(download_url).path)
    directory = cdd_osm_dat_geofabrik() + os.path.dirname(parsed_path)  # .title()
    filename = os.path.basename(parsed_path)

    if not os.path.exists(directory):
        os.makedirs(directory)
    file_path = os.path.join(directory, filename)

    return filename, file_path


# ====================================================================================================================
""" Download/Remove data """


# Download files
def download_subregion_osm_file(subregion, file_format=".shp.zip", update=False):
    """
    :param subregion: 
    :param file_format: '.osm.pbf', '.shp.zip', '.osm.bz2'
    :param update: 
    :return: 
    """
    available_file_formats = ('.osm.pbf', '.shp.zip', '.osm.bz2')
    if file_format not in available_file_formats:
        print("'file_format' must be chosen from among {}.".format(available_file_formats))
    else:
        # Get download URL
        subregion_name, download_url = get_download_url(subregion, file_format)
        # Download the requested OSM file
        filename, file_path = make_file_path(download_url)

        if os.path.isfile(file_path) and not update:
            print("'{}' is already available for {}.".format(filename, subregion_name))
        else:

            """
            # Make a custom bar to show downloading progress --------------------------
            def make_custom_progressbar():
                widgets = [progressbar.Bar(), ' ', progressbar.Percentage(),
                           ' [', progressbar.Timer(), '] ',
                           progressbar.FileTransferSpeed(),
                           ' (', progressbar.ETA(), ') ']
                progress_bar = progressbar.ProgressBar(widgets=widgets)
                return progress_bar

            pbar = make_custom_progressbar()

            def show_progress(block_count, block_size, total_size):
                if pbar.max_value is None:
                    pbar.max_value = total_size
                    pbar.start()
                pbar.update(min(block_count * block_size, total_size))
            # -------------------------------------------------------------------------
            """
            try:
                urllib.request.urlretrieve(download_url, file_path)
                # urllib.request.urlretrieve(download_url, file_path, reporthook=show_progress)
                # pbar.finish()
                # time.sleep(0.1)
                print("\n'{}' is downloaded for {}.".format(filename, subregion_name))
            except Exception as e:
                print("\nDownload failed due to '{}'.".format(e))


# Remove the downloaded file
def remove_subregion_osm_file(subregion, file_format=".shp.zip"):
    available_file_formats = ('.osm.pbf', '.shp.zip', '.osm.bz2')
    if file_format not in available_file_formats:
        print("'file_format' must be chosen from among {}.".format(available_file_formats))
    else:
        # Get download URL
        subregion_name, download_url = get_download_url(subregion, file_format)
        # Download the requested OSM file
        filename, file_path = make_file_path(download_url)
        if os.path.isfile(file_path):
            os.remove(file_path)
            print("'{}' has been removed.".format(filename))
        else:
            print("The target file, '{}', does not exist.".format(filename))


# Counties which are involved with the Anglia Route
def list_of_relevant_subregions():
    counties = ['Berkshire',
                'Buckinghamshire',
                'Cambridgeshire',
                'East Sussex',
                'Essex',
                'Greater London',
                'Hampshire',
                'Hertfordshire',
                'Kent',
                'Leicestershire',
                'Norfolk',
                'Nottinghamshire',
                'Oxfordshire',
                'Suffolk',
                'Surrey',
                'West Midlands',
                'West Sussex']
    return counties


# ====================================================================================================================
""" Read OSM data """


# Make a dictionary with keys and values being shape_type code (in OSM .shp file) and shapely.geometry, respectively
def shape_type_dict():
    shape_types = {1: shapely.geometry.Point,
                   2: shapely.geometry.MultiPoint,
                   3: shapely.geometry.LineString,
                   4: shapely.geometry.MultiLineString,
                   5: shapely.geometry.Polygon,
                   6: shapely.geometry.MultiPolygon}
    return shape_types


# Merge a set of .shp files (for a given layer)
def merge_shp_files(subregions, layer, update=False):
    """
    Layers include buildings, landuse, natural, places, points, railways, roads and waterways

    Create a .prj projection file for a .shp file: 
    http://geospatialpython.com/2011/02/create-prj-projection-file-for.html

    :param subregions: 
    :param layer:
    :param update: 
    :return:
    """
    # Make sure all the required shapefiles are ready
    subregion_name_and_download_url = [get_download_url(subregion, '.shp.zip') for subregion in subregions]
    # Download the requested OSM file
    filename_and_path = [make_file_path(download_url) for k, download_url in subregion_name_and_download_url]

    info_list = [itertools.chain(*x) for x in zip(subregion_name_and_download_url, filename_and_path)]

    extract_dirs = []
    for subregion_name, download_url, filename, file_path in info_list:
        if not os.path.isfile(file_path) or update:

            # Make a custom bar to show downloading progress
            def make_custom_progressbar():
                widgets = [progressbar.Bar(), ' ',
                           progressbar.Percentage(),
                           ' [', progressbar.Timer(), '] ',
                           progressbar.FileTransferSpeed(),
                           ' (', progressbar.ETA(), ') ']
                progress_bar = progressbar.ProgressBar(widgets=widgets)
                return progress_bar

            pbar = make_custom_progressbar()

            def show_progress(block_count, block_size, total_size):
                if pbar.max_value is None:
                    pbar.max_value = total_size
                    pbar.start()
                pbar.update(block_count * block_size)

            urllib.request.urlretrieve(download_url, file_path, reporthook=show_progress)
            pbar.finish()
            time.sleep(0.01)
            print("\n'{}' is downloaded for {}.".format(filename, subregion_name))

        extract_dir = os.path.splitext(file_path)[0]
        with zipfile.ZipFile(file_path, 'r') as shp_zip:
            shp_zip.extractall(extract_dir)
            shp_zip.close()
        extract_dirs.append(extract_dir)

    # Specify a directory that stores files for the specific layer
    layer_path = cdd('Network\\OpenStreetMap', layer)
    if not os.path.exists(layer_path):
        os.mkdir(layer_path)

    # Copy railways .shp files into Railways folder
    for subregion, p in zip(subregions, extract_dirs):
        for original_filename in glob.glob1(p, "*{}*".format(layer)):
            dest = os.path.join(layer_path, "{}_{}".format(subregion.lower().replace(' ', '-'), original_filename))
            shutil.copyfile(os.path.join(p, original_filename), dest)

    # Resource: http://geospatialpython.com/2011/02/merging-lots-of-shapefiles-quickly.html
    shp_file_paths = glob.glob(os.path.join(layer_path, '*.shp'))
    w = shapefile.Writer()
    for f in shp_file_paths:
        readf = shapefile.Reader(f)
        w.shapes().extend(readf.shapes())
        w.records.extend(readf.records())
        w.fields = list(readf.fields)
    w.save(os.path.join(layer_path, layer))


# Read .shp file (Alternative to geopandas.read_file())
def read_shp_file(path_to_shp):
    """
    :param path_to_shp:
    :return:

    len(shp.records()) == shp.numRecords
    len(shp.shapes()) == shp.numRecords
    shp.bbox  # boundaries

    """

    # Read .shp file using shapefile.Reader()
    shp_reader = shapefile.Reader(path_to_shp)

    # Transform the data to a DataFrame
    filed_names = [field[0] for field in shp_reader.fields[1:]]
    shp_data = pd.DataFrame(shp_reader.records(), columns=filed_names)

    # shp_data['name'] = shp_data.name.str.encode('utf-8').str.decode('utf-8')  # Clean data
    shape_info = pd.DataFrame([(s.points, s.shapeType) for s in shp_reader.iterShapes()],
                              index=shp_data.index, columns=['coords', 'shape_type'])
    shp_data = shp_data.join(shape_info)

    return shp_data


# Make a path to pickle file for OSM data, given layer and feature
def make_osm_pickle_file_path(extract_dir, layer, feature, suffix='shp'):
    subregion_name = os.path.basename(extract_dir).split('-')[0]
    filename = "-".join((s for s in [subregion_name, layer, feature, suffix] if s is not None)) + ".pickle"
    path_to_file = os.path.join(extract_dir, filename)
    return path_to_file


# Read .shp.zip
def read_shp_zip(subregion, layer, feature=None, update=False, keep_extracts=True):
    """
    :param subregion: 
    :param layer: 
    :param feature: 
    :param update: 
    :param keep_extracts: 
    :return: 
    """
    _, download_url = get_download_url(subregion, file_format=".shp.zip")
    _, file_path = make_file_path(download_url)

    extract_dir = os.path.splitext(file_path)[0]

    path_to_shp_pickle = make_osm_pickle_file_path(extract_dir, layer, feature)

    if os.path.isfile(path_to_shp_pickle) and not update:
        shp_data = load_pickle(path_to_shp_pickle)
    else:
        if not os.path.exists(extract_dir) or glob.glob(os.path.join(extract_dir, '*{}*.shp'.format(layer))) == [] or \
                update:

            if not os.path.isfile(file_path) or update:
                # Download the requested OSM file urlretrieve(download_url, file_path)
                download_subregion_osm_file(subregion, file_format='.shp.zip', update=update)

            with zipfile.ZipFile(file_path, 'r') as shp_zip:
                members = [f.filename for f in shp_zip.filelist if layer in f.filename]
                shp_zip.extractall(extract_dir, members)
                shp_zip.close()

        path_to_shp = glob.glob(os.path.join(extract_dir, "*{}*.shp".format(layer)))
        if len(path_to_shp) > 1:
            if feature is not None:
                path_to_shp_feature = [p for p in path_to_shp if layer + "_" + feature not in p]
                if len(path_to_shp_feature) == 1:  # The "a_*.shp" file does not exist
                    path_to_shp_feature = path_to_shp_feature[0]
                    shp_data = gpd.read_file(path_to_shp_feature)
                    shp_data = shp_data[shp_data.fclass == feature]
                    shp_data.crs = {'no_defs': True, 'ellps': 'WGS84', 'datum': 'WGS84', 'proj': 'longlat'}
                    shp_data.to_file(path_to_shp_feature.replace(layer, layer + "_" + feature), driver='ESRI Shapefile')
                else:   # An old .shp for feature is available, but an "a_" file also exists
                    shp_data = [gpd.read_file(p) for p in path_to_shp_feature]
                    shp_data = [dat[dat.fclass == feature] for dat in shp_data]
            else:  # feature is None
                path_to_orig_shp = [p for p in path_to_shp if layer + '_a' in p or layer + '_free' in p]
                if len(path_to_orig_shp) > 1:  # An "a_*.shp" file does not exist
                    shp_data = [gpd.read_file(p) for p in path_to_shp]
                    # shp_data = pd.concat([read_shp_file(p) for p in path_to_shp], axis=0, ignore_index=True)
                else:  # The "a_*.shp" file does not exist
                    shp_data = gpd.read_file(path_to_orig_shp[0])
        else:
            path_to_shp = path_to_shp[0]
            shp_data = gpd.read_file(path_to_shp)  # gpd.GeoDataFrame(read_shp_file(path_to_shp))
            if feature is not None:
                shp_data = gpd.GeoDataFrame(shp_data[shp_data.fclass == feature])
                path_to_shp_feature = path_to_shp.replace(layer, layer + "_" + feature)
                shp_data = shp_data[shp_data.fclass == feature]
                shp_data.crs = {'no_defs': True, 'ellps': 'WGS84', 'datum': 'WGS84', 'proj': 'longlat'}
                shp_data.to_file(path_to_shp_feature, driver='ESRI Shapefile')

        if not keep_extracts:
            # import shutil
            # shutil.rmtree(extract_dir)
            for f in glob.glob(os.path.join(extract_dir, "gis.osm*")):
                # if layer not in f:
                os.remove(f)

        save_pickle(shp_data, path_to_shp_pickle)

    return shp_data
