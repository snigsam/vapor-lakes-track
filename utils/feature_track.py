"""
    Functions to create a database of binned lake contours with their properties derived from MERRA2 Data
    Also another database storing the geometry features in a dataframe ...
"""
# Import necessary libraries
from shapely import geometry
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import fiona
import os
from shapely.geometry import LineString
from shapely.geometry import Point
from shapely.ops import nearest_points
import pyproj
from glob import glob
import xarray as xr
import pandas as pd
import geopandas as gpd
from datetime import datetime, timedelta, timezone
import time
import dask
import warnings
warnings.filterwarnings('ignore')
import s3fs
import requests
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.ndimage import distance_transform_edt, binary_erosion, label
from shapely.geometry import Polygon, MultiPolygon, Point
from skimage.measure import find_contours
import flox
from flox.xarray import xarray_reduce
from IPython.display import display
import ipywidgets as widgets

# Get S3 credentials and expiration information
def get_s3_credentials():
    gesdisc_s3 = "https://data.gesdisc.earthdata.nasa.gov/s3credentials"
    response = requests.get(gesdisc_s3).json()

    # Check if 'expiration' key is in the response
    if 'expiration' not in response:
        print("Error: 'expiration' key not found in the response.")
        print("Response received:", response)
        raise KeyError("'expiration' key not found in the response.")

    expiration_time = datetime.strptime(response['expiration'], '%Y-%m-%d %H:%M:%S%z')
    fs = s3fs.S3FileSystem(key=response['accessKeyId'],
                           secret=response['secretAccessKey'],
                           token=response['sessionToken'],
                           client_kwargs={'region_name':'us-west-2'})
    return fs, expiration_time

def generate_s3_urls_for_date_slv(fs, date):
    """
    Tuple of strings: S3 URLs for diag, flux, and slv data.
    """
    year = date.strftime('%Y')
    month = date.strftime('%m')
    day = date.strftime('%d')
    date_pattern = f'{year}{month}{day}'

    base_path_slv = 'gesdisc-cumulus-prod-protected/MERRA2/M2T1NXSLV.5.12.4/'    

    # Construct the full path for each data type
    full_path_slv = f'{base_path_slv}{year}/{month}/'
    
    # Use fs.ls to list files in each directory
    files_slv = fs.ls(full_path_slv)
    
    # Filter files by date
    url_slv = next((f's3://{file}' for file in files_slv if date_pattern in file), None)

    return url_slv

# Generate S3 URLs for a specific date
def generate_s3_urls_for_date(fs, date):
    year = date.strftime('%Y')
    month = date.strftime('%m')
    day = date.strftime('%d')
    date_pattern = f'{year}{month}{day}'
    
    # Base paths for different types of data
    base_path_mrad = 'gesdisc-cumulus-prod-protected/MERRA2/M2T3NVRAD.5.12.4/'
    base_path_srad = 'gesdisc-cumulus-prod-protected/MERRA2/M2T1NXRAD.5.12.4/'
    base_path_asmv = 'gesdisc-cumulus-prod-protected/MERRA2/M2T3NVASM.5.12.4/'
    base_path_diag = 'gesdisc-cumulus-prod-protected/MERRA2/M2T1NXINT.5.12.4/'
    base_path_flux = 'gesdisc-cumulus-prod-protected/MERRA2/M2T1NXFLX.5.12.4/'
    base_path_slv = 'gesdisc-cumulus-prod-protected/MERRA2/M2T1NXSLV.5.12.4/'  
    
    # Construct the full path for each data type
    full_path_mrad = f'{base_path_mrad}{year}/{month}/'
    full_path_srad = f'{base_path_srad}{year}/{month}/'
    full_path_asmv = f'{base_path_asmv}{year}/{month}/'
    full_path_diag = f'{base_path_diag}{year}/{month}/'
    full_path_flux = f'{base_path_flux}{year}/{month}/'
    full_path_slv = f'{base_path_slv}{year}/{month}/'
        
    # Use fs.ls to list files in each directory
    files_mrad = fs.ls(full_path_mrad)
    files_srad = fs.ls(full_path_srad)
    files_asmv = fs.ls(full_path_asmv)
    files_diag = fs.ls(full_path_diag)
    files_flux = fs.ls(full_path_flux)
    files_slv = fs.ls(full_path_slv)

    # Filter files by date
    url_mrad = next((f's3://{file}' for file in files_mrad if date_pattern in file), None)
    url_srad = next((f's3://{file}' for file in files_srad if date_pattern in file), None)
    url_asmv = next((f's3://{file}' for file in files_asmv if date_pattern in file), None)
    url_diag = next((f's3://{file}' for file in files_diag if date_pattern in file), None)
    url_flux = next((f's3://{file}' for file in files_flux if date_pattern in file), None)
    url_slv = next((f's3://{file}' for file in files_slv if date_pattern in file), None)
    
    return url_mrad, url_srad, url_asmv, url_diag, url_flux, url_slv

# Generate date range between start_date and end_date.
def generate_date_range(start_date, end_date):
    start = datetime.strptime(start_date, '%Y-%m-%d').date()
    end = datetime.strptime(end_date, '%Y-%m-%d').date()
    delta = end - start
    return [start + timedelta(days=i) for i in range(delta.days + 1)]

# Create a grid of points from latitude and longitude arrays
def create_points_grid(latitudes, longitudes):
    lat2d, lon2d = np.meshgrid(latitudes, longitudes)
    points = gpd.GeoSeries([Point(x, y) for x, y in zip(lon2d.ravel(), lat2d.ravel())])
    return points.set_crs(epsg=4326), lat2d, lon2d

# Calculate nonzero distance from points to a polygon
def calculate_nonzero_distance(point, polygon):
    nearest_point = nearest_points(point, polygon.exterior)[1] # Get the nearest point on the polygon's boundary
    nonzero_distance = point.distance(nearest_point)
    return nonzero_distance

# Calculate directions to centroid for a batch of points
def calculate_directions_to_centroid_batch(lon2d, lat2d, centlon, centlat):
    geod = pyproj.Geod(ellps='WGS84')
    lon_flat, lat_flat = lon2d.ravel(), lat2d.ravel()
    fwd_azimuths, _, _ = geod.inv(lon_flat, lat_flat, np.full_like(lon_flat, centlon), np.full_like(lat_flat, centlat))
    adjusted_azimuths = (fwd_azimuths - 180) % 360
    return adjusted_azimuths

def calculate_distance_from_centroid(point, polygon):
    # Calculate the centroid of the polygon
    centroid = polygon.centroid
    # Calculate the Euclidean distance from the point to the centroid
    return point.distance(centroid)

def calculate_distances_from_centroid(points, polygon):
    # Apply the new function to calculate distance from centroid for each point
    return points.apply(lambda point: calculate_distance_from_centroid(point, polygon))


# Calculate perimeter to area ratio
def perimeter_area_ratio(perimeter, area):
    return (np.pi*area) / (perimeter**2) # Polsby-Popper (PP) measure (polsby & Popper, 1991)

def gdf_from_contours_scipy(lon, lat, tqv, conlevel):
    """
    Extract contours using scipy and plot them without closing the boundaries.
    """
    lon_grid, lat_grid = np.meshgrid(lon, lat)

    if hasattr(tqv, 'values'):
        tqv = tqv.values

    # Use scipy to find contours at the specified level
    contours = find_contours(tqv, conlevel)

    closed_polygons = []
    open_linestrings = []

    # Loop over each contour and classify as closed or open
    for contour in contours:
        contour_lon = np.interp(contour[:, 1], np.arange(lon_grid.shape[1]), lon_grid[0, :])
        contour_lat = np.interp(contour[:, 0], np.arange(lat_grid.shape[0]), lat_grid[:, 0])
        points = list(zip(contour_lon, contour_lat))
        line = LineString(points)
        
        # Check if the LineString is closed (start and end points are the same)
        if line.is_ring:
            closed_polygons.append(Polygon(points))  # Convert closed contour to Polygon
        else:
            open_linestrings.append(line)  # Keep open LineStrings

    return closed_polygons, open_linestrings

def filter_single_point_lines(open_linestrings):
    """
    Remove single-point LineStrings.
    """
    return [line for line in open_linestrings if len(line.coords) > 1]

def sort_boundary_points(open_linestrings, lon_bound, lat_bound):
    """
    Extract start and end points of LineStrings and sort them anticlockwise along the bounding box.
    """
    boundary_points = []
    
    for line in open_linestrings:
        start, end = Point(line.coords[0]), Point(line.coords[-1])
        boundary_points.append(start)
        boundary_points.append(end)
    
    # Sort the points by their positions in the bounding box in anticlockwise order
    sorted_points = sorted(boundary_points, key=lambda p: (p.x, p.y))  # Anticlockwise order
    
    return sorted_points

def check_tqv_along_line(line, lon, lat, tqv, conlevel=45):
    """
    Check the average TQV values along a line between two points.
    Returns True if the average TQV value along the line is above the threshold.
    """
    num_points = 10
    line_points = [line.interpolate(i / num_points, normalized=True) for i in range(num_points + 1)]
    
    tqv_values = []
    for point in line_points:
        lon_value, lat_value = point.x, point.y
        tqv_value = tqv_at_point(lon_value, lat_value, lon, lat, tqv)
        tqv_values.append(tqv_value)
    
    avg_tqv_value = np.mean(tqv_values)
    return avg_tqv_value >= conlevel

def tqv_at_point(lon_value, lat_value, lon, lat, tqv):
    lon_idx = np.argmin(np.abs(lon - lon_value))
    lat_idx = np.argmin(np.abs(lat - lat_value))
    return tqv[lat_idx, lon_idx]

def join_linestrings(sorted_points, open_linestrings, lon, lat, tqv, conlevel):
    """
    Join linestrings based on sorted points and TQV check.
    Extend LineStrings by connecting valid points and check if they form closed loops.
    """
    joined_lines = open_linestrings.copy()
    i = 0

    while i < len(sorted_points) - 1:
        point1 = sorted_points[i]
        point2 = sorted_points[i + 1]

        # Check if the TQV values along the boundary line between point1 and point2 are valid
        boundary_line = LineString([point1, point2])
        if check_tqv_along_line(boundary_line, lon, lat, tqv, conlevel):
            # Find the LineStrings associated with point1 and point2
            line1, line2 = None, None
            for line in joined_lines:
                if Point(line.coords[0]).equals(point1) or Point(line.coords[-1]).equals(point1):
                    line1 = line
                if Point(line.coords[0]).equals(point2) or Point(line.coords[-1]).equals(point2):
                    line2 = line

            # Case 1: Both points belong to different LineStrings, so we merge them
            if line1 and line2 and line1 != line2:
                if Point(line1.coords[-1]).equals(point1):
                    line1_coords = list(line1.coords)
                else:
                    line1_coords = list(line1.coords[::-1])  # Reverse the coordinates

                if Point(line2.coords[0]).equals(point2):
                    line2_coords = list(line2.coords)
                else:
                    line2_coords = list(line2.coords[::-1])

                # Merge the two LineStrings and the boundary line
                new_coords = line1_coords + [point2] + line2_coords
                new_line = LineString(new_coords)
                joined_lines.remove(line1)
                joined_lines.remove(line2)
                joined_lines.append(new_line)

            # Case 2: Both points belong to the same LineString, we try to close the loop
            elif line1 == line2 and line1:
                if Point(line1.coords[-1]).equals(point1):
                    line1_coords = list(line1.coords)
                else:
                    line1_coords = list(line1.coords[::-1])

                # Close the loop by adding the points to form a ring
                new_coords = line1_coords + [point2, line1_coords[0]]
                new_line = LineString(new_coords)
                joined_lines.remove(line1)
                joined_lines.append(new_line)

        i += 2  # Move to the next pair

    return joined_lines


def create_closed_polygons(closed_polygons, joined_lines):
    """
    Create closed polygons by joining boundary points and open lines.
    Convert LineStrings into polygons if they form closed loops.
    """
    # Initialize the GeoDataFrame for the final polygons
    gdf_polygons = gpd.GeoDataFrame(columns=['geometry', 'touches_boundary'], geometry='geometry')

    final_polygons = closed_polygons.copy()
    
    # Add closed polygons with 'touches_boundary' set to False initially
    for polygon in closed_polygons:
        gdf_polygons = gdf_polygons._append({
            'geometry': polygon,
            'touches_boundary': False
        }, ignore_index=True)
    
    # Iterate over joined lines and convert valid closed LineStrings to polygons
    for line in joined_lines:
        # Extract points from the LineString and check if the first and last points match
        points = list(line.coords)
        if len(points) >= 3 and points[0] == points[-1]:
            # Convert to polygon only if it's a closed ring
            polygon = Polygon(points)
            if polygon.is_valid:
                gdf_polygons = gdf_polygons._append({
                    'geometry': polygon,
                    'touches_boundary': True
                }, ignore_index=True)

    return gdf_polygons
def remove_nested_polygons(gdf_polygons):
    """
    Remove polygons that are completely inside another polygon in the GeoDataFrame.
    """
    # List to store the final polygons
    final_polygons = []
    
    # Iterate over each polygon
    for i, poly in gdf_polygons.iterrows():
        is_contained = False
        # Check if the current polygon is contained within any other polygon
        for j, other_poly in gdf_polygons.iterrows():
            if i != j:  # Don't compare the polygon with itself
                if other_poly['geometry'].contains(poly['geometry']):
                    is_contained = True
                    break
        # If not contained in any other polygon, add it to final_polygons
        if not is_contained:
            final_polygons.append(poly)
    
    # Create a new GeoDataFrame with the filtered polygons
    filtered_gdf = gpd.GeoDataFrame(final_polygons, columns=gdf_polygons.columns)
    
    return filtered_gdf


# Load dataset from URL and cache it
def load_dataset(url, cache, date_key,fs):
    if date_key not in cache:
        cache[date_key] = xr.open_dataset(fs.open(url), decode_cf=True)
    return cache[date_key]

# RFR.py

import geopandas as gpd

def add_east_africa_touch_column(polygon):

    #Adds a column to `gdf_polygons` indicating if each polygon touches or overlaps
    #with any of the specified East African countries.

    # Default East African countries including Madagascar
    east_african_countries = ['Kenya', 'Tanzania', 'Somalia', 'Madagascar', 'Mozambique']
    
    url="https://d2ad6b4ur7yvpq.cloudfront.net/naturalearth-3.3.0/ne_50m_admin_0_countries.geojson"

    gdf = gpd.read_file(url)
    # Load Natural Earth data for countries and filter for the specified East African countries
    #world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    east_africa = gdf[gdf['name'].isin(east_african_countries)]
    
    # Initialize the `touches_east_africa` column based on intersection with East Africa
    return east_africa['geometry'].intersects(polygon)

    #return east_africa.intersects(polygon).any()
def handling_FeatureCollection(polygon_data):
    processed_data = []
    for entry in polygon_data:
        # Check if the geometry is a FeatureCollection and handle accordingly
        if 'geometry' in entry:
            if entry['geometry'].iloc[0].geom_type == 'FeatureCollection':
                # Extract polygons within the FeatureCollection
                features = entry['geometry'].iloc[0]['features']
                # Convert each feature to a shapely geometry
                geometries = [shape(feature['geometry']) for feature in features if feature['geometry']['type'] in ['Polygon', 'MultiPolygon']]
                
                # Use only the first Polygon if multiple are found (modify as needed)
                if geometries:
                    entry['geometry'] = geometries[0]  # Replace with first valid Polygon
            else:
                entry['geometry'] = entry['geometry'].iloc[0]  # Directly extract if not a FeatureCollection
        
        processed_data.append(entry)  # Add processed entry to the list

    return processed_data

# Calculate haversine distance between two points
def haversine(lat1, lon1, lat2, lon2):
    R = 6371000  # Earth radius in meters
    phi1, phi2 = np.deg2rad(lat1), np.deg2rad(lat2)
    dphi = np.deg2rad(lat2 - lat1)
    dlambda = np.deg2rad(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c

# Calculate geodesic distances for a 2D grid
def calculate_geodesic_distance(lat2d, lon2d):
    R = 6371000  # Earth radius in meters
    dlat = np.empty(lat2d.shape)
    dlon = np.empty(lon2d.shape)
    for i in range(lat2d.shape[0] - 1):
        dlat[i, :] = haversine(lat2d[i, 0], lon2d[i, 0], lat2d[i + 1, 0], lon2d[i + 1, 0])
    for j in range(lon2d.shape[1] - 1):
        dlon[:, j] = haversine(lat2d[0, j], lon2d[0, j], lat2d[0, j + 1], lon2d[0, j + 1])
    return dlat, dlon

# Process a single time step
def process_time_step(inow, cwv, url_mrad, url_srad, url_asmv, url_diag, url_flux, url_slv, data_cache, path_lake_data, path_lake_binned, pressure_levels, min_pressure_level, lat_bound, lon_bound, tqv_conlevel, var_mrad, var_srad, var_asmv, var_diag, var_flux, var_slv,MINSIZE,all_gdflakes_date,fs,fs_out):
    try:
        gdf_now = gdf_from_contours_scipy(cwv.lon, cwv.lat, cwv[inow], tqv_conlevel, lon_bound, lat_bound)
    except Exception as e:
        print(f"Failed to generate contours for time step {inow}: {e}")
        return

    yyyymmddhh = cwv.time[inow].values
    timestring = pd.to_datetime(str(yyyymmddhh)).strftime('%H:%M')
    datestring = pd.to_datetime(str(yyyymmddhh)).strftime('%Y_%m_%d')
    datetime_str = pd.to_datetime(datestring.replace('_', '-') + ' ' + timestring)
    
    gdf_now['Time'] = timestring 
    gdf_now['Date'] = datestring
    gdf_now['DateTime'] = datetime_str
    if gdf_now.crs is None:
        gdf_now.set_crs(epsg=4326, inplace=True)

    numlakes = gdf_now.count()[0]
    i = 0
    while i < numlakes:
        single_row_df = gdf_now.iloc[[i]]
        lake = gpd.GeoDataFrame(single_row_df, geometry='geometry')
        if (lake.geometry.area.values[0] > MINSIZE):
            lake['Perimeter'] = lake.geometry.length
            lake['Area'] = lake.geometry.area
            lake['Effective_Radius'] = np.sqrt(lake['Area'] / np.pi)
            lake['Centroidlat'] = lake.geometry.centroid.y
            lake['Centroidlon'] = lake.geometry.centroid.x
            lake['Centroid_is_inside'] = lake.geometry.contains(lake.geometry.centroid)
            lake['Maxlon'] = lake.geometry.bounds.maxx
            lake['Minlon'] = lake.geometry.bounds.minx
            lake['Maxlat'] = lake.geometry.bounds.maxy
            lake['Minlat'] = lake.geometry.bounds.miny
            lake['Perimeter_Area_Ratio'] = lake['Perimeter'] / lake['Area']
            lake['lake_idx'] = i
            all_gdflakes_date.append(lake)
            
            points, lat2d , lon2d = create_points_grid(cwv.lat.values, cwv.lon.values)
            nonzero_distances = points.apply(lambda point: calculate_nonzero_distance(point, lake))
            dist = np.array(nonzero_distances).reshape(len(cwv.lon), len(cwv.lat))
            isin = points.geometry.apply(lambda g: lake.contains(g)).values.reshape(len(cwv.lon),len(cwv.lat))
            dist *= (-2)*(isin-0.5)  # make SIGNED distance from boundary, positive is exterior 
            dist = np.transpose(dist)
            
            dir_to = calculate_directions_to_centroid_batch(lon2d, lat2d, lake.geometry.centroid.x, lake.geometry.centroid.y)
            dir_to = dir_to.reshape(len(cwv.lon),len(cwv.lat))
            dir_to = np.transpose(dir_to)
            lake['distance'] = [{'data': dist, 'dims': ('lat', 'lon'), 'coords': {'lat': cwv.lat.values, 'lon': cwv.lon.values}}]
            lake['dir_to'] = [{'data': dir_to, 'dims': ('lat', 'lon'), 'coords': {'lat': cwv.lat.values, 'lon': cwv.lon.values}}]

            ds_mrad = load_dataset(url_mrad, data_cache, f'mrad_{datestring}',fs)
            ds_srad = load_dataset(url_srad, data_cache, f'srad_{datestring}',fs)
            ds_asmv = load_dataset(url_asmv, data_cache, f'asmv_{datestring}',fs)
            ds_diag = load_dataset(url_diag, data_cache, f'diag_{datestring}',fs)
            ds_flux = load_dataset(url_flux, data_cache, f'flux_{datestring}',fs)
            ds_slv = load_dataset(url_slv, data_cache, f'slv_{datestring}',fs)

            relevant_levels = pressure_levels >= min_pressure_level
            relevant_pressure_levels = pressure_levels[relevant_levels]
            pressure_dict = {lev: pressure for lev, pressure in enumerate(relevant_pressure_levels, start=1)}

            ds_mrad_select = ds_mrad.sel(time=yyyymmddhh, lev=relevant_levels).sel(lat=slice(lat_bound[0], lat_bound[1]), lon=slice(lon_bound[0], lon_bound[1]))[var_mrad]
            ds_srad_select = ds_srad.sel(time=yyyymmddhh).sel(lat=slice(lat_bound[0], lat_bound[1]), lon=slice(lon_bound[0], lon_bound[1]))[var_srad]
            ds_asmv_select = ds_asmv.sel(time=yyyymmddhh, lev=relevant_levels).sel(lat=slice(lat_bound[0], lat_bound[1]), lon=slice(lon_bound[0], lon_bound[1]))[var_asmv]
            ds_diag_select = ds_diag.sel(time=yyyymmddhh).sel(lat=slice(lat_bound[0], lat_bound[1]), lon=slice(lon_bound[0], lon_bound[1]))[var_diag]
            ds_flux_select = ds_flux.sel(time=yyyymmddhh).sel(lat=slice(lat_bound[0], lat_bound[1]), lon=slice(lon_bound[0], lon_bound[1]))[var_flux]
            ds_slv_select = ds_slv.sel(time=yyyymmddhh).sel(lat=slice(lat_bound[0], lat_bound[1]), lon=slice(lon_bound[0], lon_bound[1]))[var_slv]

            merged_merra2 = xr.merge([ds_mrad_select, ds_srad_select, ds_asmv_select, ds_diag_select, ds_flux_select, ds_slv_select], compat='override')

            merged_merra2 = merged_merra2.assign_coords({'pressure': ('lev', list(pressure_dict.values()))})
            
            # Convert degrees to meters
            dlat_2d, dlon_2d = calculate_geodesic_distance(lat2d, lon2d)
            
            dlat_2d = np.transpose(dlat_2d)
            dlon_2d = np.transpose(dlon_2d)
            
            UFLXQV = merged_merra2['UFLXQV']
            VFLXQV = merged_merra2['VFLXQV']  
            Omega = merged_merra2['OMEGA']
            Qv = merged_merra2['QV']

            # Compute the partial derivatives
            dUFLXQV_dx = UFLXQV.differentiate('lon') / dlon_2d
            dVFLXQV_dy = VFLXQV.differentiate('lat') / dlat_2d
            
            dVFLXQV_dx = VFLXQV.differentiate('lon') / dlat_2d
            dUFLXQV_dy = UFLXQV.differentiate('lat') / dlon_2d

            # Compute the curl (moisture vorticity)
            Moisture_vorticity = dVFLXQV_dx - dUFLXQV_dy
            
            Integrated_moisture_convergence = -(dUFLXQV_dx + dVFLXQV_dy)

            # Calculate wind profile
            U = merged_merra2['U']
            V = merged_merra2['V']
            
            dV_dx = V.differentiate('lon') / dlon_2d
            dU_dy = U.differentiate('lat') / dlat_2d
            vertical_vorticity = dV_dx - dU_dy
            
            
            # Calculate horizontal shear vorticity profile
            dU_dz = U.differentiate('pressure')
            dV_dz = V.differentiate('pressure')
            horizontal_vorticity = dU_dz + dV_dz

            merged_merra2['integrated_moisture_convergence'] = Integrated_moisture_convergence
            merged_merra2['integrated_moisture_vorticity'] = Moisture_vorticity
            merged_merra2['vertical_vorticity'] = vertical_vorticity
            merged_merra2['horizontal_vorticity'] = horizontal_vorticity

            distance_data = lake['distance'].iloc[0]['data']
            distance_dims = lake['distance'].iloc[0]['dims']
            distance_coords = lake['distance'].iloc[0]['coords']
            distance_da = xr.DataArray(data=distance_data, dims=distance_dims, coords=distance_coords, name='DIST')
    
            dir_to_data = lake['dir_to'].iloc[0]['data']
            dir_to_dims = lake['dir_to'].iloc[0]['dims']
            dir_to_coords = lake['dir_to'].iloc[0]['coords']
            dir_to_da = xr.DataArray(data=dir_to_data, dims=dir_to_dims, coords=dir_to_coords, name='DIREC')
    
            merged_merra2 = xr.merge([merged_merra2, distance_da, dir_to_da])
            merged_merra2 = merged_merra2.set_coords(['DIST', 'DIREC'])
            
            merged_merra2 = merged_merra2.assign_coords({'pressure': ('lev', list(pressure_dict.values()))})

            file_path_local_merged_merra2 = path_lake_data + str(datetime_str) + '_' + str(i) +'lake.nc'
            merged_merra2.to_netcdf(file_path_local_merged_merra2)
            s3_path_merged_merra2 = 'openscapeshub-scratch/test/'+ str(datetime_str) + '_' + str(i) +'lake.nc'

            fs_out.put(file_path_local_merged_merra2, f's3://{s3_path_merged_merra2}')
            
            distance_bins = np.arange(-5.0, 10.0, 0.2)
            direction_bins = np.linspace(0, 360, 37)

            reduced_ds = xarray_reduce(
                            merged_merra2,
                           "DIST",
                            "DIREC",
                            func = "mean",  
                            expected_groups=(
                                pd.IntervalIndex.from_breaks(distance_bins),
                                pd.IntervalIndex.from_breaks(direction_bins),
                            ),
                            isbin = True,
                            fill_value=np.nan
                            )
            
            distance_intervals = pd.IntervalIndex.from_breaks(distance_bins)
            distance_midpoints = distance_intervals.mid.values
            
            direction_intervals = pd.IntervalIndex.from_breaks(direction_bins)
            direction_midpoints = direction_intervals.mid.values

            reduced_ds = reduced_ds.assign_coords(DIST_bins=('DIST_bins', distance_midpoints), DIREC_bins=('DIREC_bins', direction_midpoints))
            reduced_ds = reduced_ds.rename({'DIST_bins': 'DIST_midpoints', 'DIREC_bins': 'DIREC_midpoints'})

            file_path_local = path_lake_binned + str(datetime_str) + '_' + str(i) +'binned_lake.nc'
            reduced_ds.to_netcdf(file_path_local)
            s3_path = 'openscapeshub-scratch/test/'+ str(datetime_str) + '_' + str(i) +'binned_lake.nc'

            fs_out.put(file_path_local, f's3://{s3_path}')
            
            all_gdflakes_date.append(lake)
        else:
            print('out of bounds or too small ', i)
        
        i = i + 1
    return

def run_process(start_date, end_date, lat_bound, lon_bound, min_pressure_level, pressure_levels, path_lake_contours_pd,path_lake_data, path_lake_binned, var_mrad, var_srad, var_asmv, var_diag, var_flux, var_slv, tqv_conlevel, MINSIZE):
    date_range = generate_date_range(start_date, end_date)        
    start_time = time.time()
    all_gdflakes = []
    all_data = []
    
    fs, expiration_time = get_s3_credentials()
    fs_out = s3fs.S3FileSystem(anon=False)

    print(f"Expiration time: {expiration_time.strftime('%Y-%m-%d %H:%M:%S')}")
    data_cache = {}
    buffer_time = timedelta(minutes=5)
    progress = widgets.IntProgress(min=0, max=len(date_range), description='Processing:')
    display(progress)
    
    for date in date_range:
        print(f"Processing date: {date.strftime('%Y-%m-%d')}")
        current_time = datetime.utcnow().replace(tzinfo=timezone.utc)
        if current_time > (expiration_time - buffer_time):
        #if datetime.utcnow() > (expiration_time - buffer_time):
            # Refresh S3 credentials
            fs, expiration_time = get_s3_credentials()
            fs_out = s3fs.S3FileSystem(anon=False)
            print(f"Credentials refreshed at {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}, new expiration time: {expiration_time.strftime('%Y-%m-%d %H:%M:%S')}")
            
        all_gdflakes_date = []
        url_mrad, url_srad, url_asmv, url_diag, url_flux, url_slv = generate_s3_urls_for_date(fs, date)
        cwv_data = xr.open_dataset(fs.open(url_slv))
        cwv_WEIO = cwv_data.sel(lat=slice(lat_bound[0], lat_bound[1]), lon=slice(lon_bound[0], lon_bound[1]))
        cwv_WEIO_sub_3hr = cwv_WEIO.isel(time=slice(1, None, 3))['TQV']
        n_time_steps = len(cwv_WEIO_sub_3hr.time)
        for inow in range(n_time_steps):
            process_time_step(inow, cwv_WEIO_sub_3hr, url_mrad, url_srad, url_asmv, url_diag, url_flux, url_slv, data_cache, path_lake_data, path_lake_binned, pressure_levels, min_pressure_level, lat_bound, lon_bound, tqv_conlevel, var_mrad, var_srad, var_asmv, var_diag, var_flux, var_slv,MINSIZE,all_gdflakes_date,fs,fs_out)
        if all_gdflakes_date:  
            all_lakes = pd.concat(all_gdflakes_date).reset_index(drop=True) 
            local_file_path = path_lake_contours_pd + 'LakeContours_' + str(date) + '.csv'
            all_lakes.to_csv(local_file_path, index=False) 
            s3_path = 'openscapeshub-scratch/test/' + 'LakeContours_' + str(date) + '.csv'
            fs_out.put(local_file_path, s3_path)

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time} seconds")
