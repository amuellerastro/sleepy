#!/usr/bin/env python3
################################################################################
##
## @file sleepy.py
## @brief query openstreetmap and compute distance from features in a predefined grid.
## @author Andre Mueller (<amuellerastro@gmail.com>)
##
## ### SYNOPSIS
##
##     sleepy.py [-h]
##
## ### DESCRIPTION
##
##      Query openstreetmap through overpass API. Compute distance from features in a predefined grid.
##      Compute gradient, slope inside the search areaa.
##
## ### OPTIONS
##
## positional arguments:
##
##    coordinates           Longitude, Latitude [deg]
##
##  optional arguments:
##    -h, --help               Show this help message and exit.
##    --area_name              Name of the area. Has no influence on the provided coordinates. This paramter is just added to the file name of the output files.
##    --radius_km              Search radius around the provided coordinates in kilometer (default=0.8).
##    --res_m                  The spatial resolution of the search grid in meter (default=100).
##    --cmap                   The name of the color map used to plot the contour map (default='seismic')
##    --min_dist               The minimum distance from buildings, roads, hunting stands to be considered acceptable in meter (default=300).
##    --gpx                    Load a .gpx file. Track will be overplotted. Default: no gpx file is loaded.
##    --topo                   If True (default=False) the elevation of the defined grid of coordinates is queried via an API from a local server.
##
## ### EXAMPLE
##
##  Simple casee:
##    $> sleepy.py 8.123 50.5
##
################################################################################
# https://towardsdatascience.com/loading-data-from-openstreetmap-with-python-and-the-overpass-api-513882a27fd0

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
# import branca
import folium
from folium import plugins
import geojsoncontour
import requests
import argparse
import gpxpy
import gpxpy.gpx
from tqdm import trange
import sys
# from progress.bar import ChargingBar
# import pdb; pdb.set_trace()
# from astropy import units as u
# from astropy.constants import R_earth


def create_parser():
    parser = argparse.ArgumentParser(description='Query openstreetmap for features around a given location.')

    parser.add_argument('coordinates', type=float, default=[8.5, 48.5], nargs='+', help="Longitude and Latitude in degrees of the area of interest.")
    parser.add_argument('--area_name', type=str, default=[], help="Name of the area of interest")
    parser.add_argument('--radius_km', type=float, default=0.8, help="Search radius around the provided coordinates in kilometer.")
    parser.add_argument('--res_m', type=float, default=100, help="The spatial resolution of the search grid in meter.")
    parser.add_argument('--cmap', type=str, default='seismic', help="The name of the color map used to plot the contour map")
    parser.add_argument('--min_dist', type=float, default=300, help="The minimum distance from buildings, roads, hunting stands to be considered acceptable in meter.")
    parser.add_argument('--gpx', default=False, help="Load a .gpx file. Track will be overplotted.")
    parser.add_argument('--topo', default=False, help="The elevation of the defined grid of coordinates is queried via an API from a local server.")

    return parser


def circular_segment(radius):
    r_earth = 6378.1  # R_earth.to(u.km).value from astropy.constants
    return np.degrees(radius / r_earth)


def define_search_area(lon, lat, radius):
    # circular segment
    alpha = circular_segment(radius)
    # correction for geographical latitude
    lat_corr = (np.cos(np.radians(lat)))
    corner_area = [lat - alpha, lon - alpha / lat_corr, lat + alpha, lon + alpha / lat_corr]
    return corner_area


def query_overpass(corner_area):
    overpass_url = "http://overpass-api.de/api/interpreter"
    overpass_query = f"""
    [out:json];
    (
      way["highway"]({corner_area[0]}, {corner_area[1]}, {corner_area[2]}, {corner_area[3]});
      node["amenity"="hunting_stand"]({corner_area[0]}, {corner_area[1]}, {corner_area[2]}, {corner_area[3]});
      way["amenity"="hunting_stand"]({corner_area[0]}, {corner_area[1]}, {corner_area[2]}, {corner_area[3]});
      relation["amenity"="hunting_stand"]({corner_area[0]}, {corner_area[1]}, {corner_area[2]}, {corner_area[3]});
      node["amenity"="shelter"]({corner_area[0]}, {corner_area[1]}, {corner_area[2]}, {corner_area[3]});
      way["amenity"="shelter"]({corner_area[0]}, {corner_area[1]}, {corner_area[2]}, {corner_area[3]});
      relation["amenity"="shelter"]({corner_area[0]}, {corner_area[1]}, {corner_area[2]}, {corner_area[3]});
      way["building"="hut"]({corner_area[0]}, {corner_area[1]}, {corner_area[2]}, {corner_area[3]});

    );
    out body;
    >;
    out skel qt;
    """
    response = requests.get(overpass_url,
                            params={'data': overpass_query})

    data_overpass = response.json()
    # with open('data.json', 'w') as f:
    #    json.dump(data_overpass, f, indent=2)

    coordinates_query = coordinates_from_overpass(data_overpass)

    return coordinates_query


def coordinates_from_overpass(data):
    # Collect coords into list
    all_coords = []
    for element in data['elements']:
        if element['type'] == 'node':
            lon = element['lon']
            lat = element['lat']
            all_coords.append((lon, lat))

    return all_coords


def define_search_grid(corner_area, grid_resolution):
    # create a 2D array regularly gridded, based on the required resulotion, e.g. 20m
    # at the moment take the center latitude to compute the correction factor
    # for longitude because the dimension of the search box are small

    # following computation of X, Y based on
    # https://stackoverflow.com/questions/32208359/is-there-a-multi-dimensional-version-of-arange-linspace-in-numpy

    # circular segment from grid resolution
    alpha = circular_segment(grid_resolution)

    # add and subtract alpha_grid to give some margin at the edges when creating the grid
    return np.mgrid[(corner_area[1] - alpha):(corner_area[3] + alpha):alpha,
           corner_area[0] - alpha:corner_area[2] + alpha:alpha]


def spherical_dist(pos1, pos2, r=6378.1):
    pos1 = pos1 * np.pi / 180
    pos2 = pos2 * np.pi / 180
    cos_lat1 = np.cos(pos1[..., 0])
    cos_lat2 = np.cos(pos2[..., 0])
    cos_lat_d = np.cos(pos1[..., 0] - pos2[..., 0])
    cos_lon_d = np.cos(pos1[..., 1] - pos2[..., 1])
    return r * np.arccos(cos_lat_d - cos_lat1 * cos_lat2 * (1 - cos_lon_d))


def compute_distance_from_features(grid_coordinates, feature_coordinates, nlon, nlat, n_average):
    # https://stackoverflow.com/questions/19413259/efficient-way-to-calculate-distance-matrix-given-latitude-and-longitude-data-in
    result = spherical_dist(grid_coordinates[:, None], np.array(feature_coordinates))
    result.sort()

    distance_km = []
    for idx in range(0, len(result)):
        distance_km.append(np.mean(result[idx][0:n_average]))

    distance_km = np.array(distance_km)
    z = distance_km.reshape(nlon, nlat)
    z_meter = z * 1e3

    return z_meter


def get_zoom_level(radius):
    if radius <= 0.5:
        zoom_level = 15
    elif 0.5 < radius <= 1.5:
        zoom_level = 14
    elif 1.5 < radius <= 2.5:
        zoom_level = 13
    elif 2.5 < radius <= 5:
        zoom_level = 12
    else:
        zoom_level = 11

    return zoom_level


def generate_base_map(radius, location=[50.0, 8.0], tiles="OpenStreetMap"):
    zoom_start = get_zoom_level(radius)
    new_map = folium.Map(location=location, control_scale=True, zoom_start=zoom_start, tiles=tiles)
    # Fullscreen mode
    plugins.Fullscreen(position='topright', force_separate_button=True).add_to(new_map)
    folium.TileLayer('openstreetmap').add_to(new_map)
    folium.TileLayer('Stamen Terrain').add_to(new_map)
    return new_map


def generate_layer_distance(tmp_map, X, Y, Z, ll=0, ul=300, cmap_name='seismic', levels=30):
    contourf = plt.contourf(X, Y, Z, levels, cmap=cmap_name, alpha=1, vmin=ll,
                            vmax=ul, linestyles='dashed')
    # Convert matplotlib contourf to geojson
    geojson = geojsoncontour.contourf_to_geojson(
        contourf=contourf,
        stroke_width=0)

    fg_dist = folium.FeatureGroup(name="Distance", show=False)
    tmp_map.add_child(fg_dist)

    # Plot the contour plot on folium
    folium.GeoJson(
        geojson,
        style_function=lambda x: {
            # 'weight': x['properties']['stroke-width'],
            'fillColor': x['properties']['fill'],
            'opacity': 0.5,  # does not work
            'weight': 0.4
        }).add_to(fg_dist)

    return tmp_map


def generate_layer_features(tmp_map, feature_coordinates):
    fg_feat = folium.FeatureGroup(name="Features", show=False)
    tmp_map.add_child(fg_feat)
    for lon, lat in feature_coordinates:
        folium.CircleMarker(
            [lat, lon],
            radius=1,
            fill=True,
            fill_color='#3186cc',
            fill_opacity=0.4,
            parse_html=False).add_to(fg_feat)

    return tmp_map


def add_central_marker(tmp_map, lon, lat, radius):
    url_google = "https://www.google.com/maps/@" + str(lat) + "," + str(lon) + "," + str(
        get_zoom_level(radius)) + "z/data=!3m1!1e3"
    link_text_google = "Google Maps"
    url_topo = "https://opentopomap.org/#map=" + str(get_zoom_level(radius)) + "/" + str(lat) + "/" + str(lon)
    link_text_topo = "Topo Map"

    label = folium.Html('<a href="' + url_google + '"target="_blank">' + link_text_google + '</a>' + ' / ' +
                        '<a href="' + url_topo + '"target="_blank">' + link_text_topo + '</a>', script=True)
    popup = folium.Popup(label, max_width=500, parse_html=True)
    fg_mark = folium.FeatureGroup(name="Central Marker", show=False)
    tmp_map.add_child(fg_mark)
    folium.Marker(
        [lat, lon],
        popup=popup).add_to(fg_mark)
    return tmp_map


def add_gpx_track(tmp_map, gpx):
    gpx_file = open(gpx, 'r')
    gpx_data = gpxpy.parse(gpx_file)
    points = []
    for track in gpx_data.tracks:
        for segment in track.segments:
            for point in segment.points:
                points.append([point.latitude, point.longitude])

    fg_gpx = folium.FeatureGroup(name="GPX Track", show=False)
    tmp_map.add_child(fg_gpx)

    for gpx_coord in points:
        folium.CircleMarker(gpx_coord,
                            radius=1,
                            color='black',
                            fill_color='black',
                            fill=True,
                            parse_html=False).add_to(fg_gpx)

    return tmp_map


def generate_layer_gradient_slope(base_map, X, Y, ll=0, ul=20, cmap_name='Reds', levels=30):
    # compute gradient
    X_1d = X.flatten()
    Y_1d = Y.flatten()
    elevation = []
    for i in trange(0, len(X_1d)):
        # https://www.opentopodata.org/api/
        topo_url = f"http://localhost:5000/v1/eudem25m?locations={Y_1d[i]},{X_1d[i]}&interpolation=cubic"
        response = requests.get(topo_url)
        data_topo = response.json()
        elevation.append(data_topo['results'][0]['elevation'])

    # https://medium.com/ai-in-plain-english/introduction-to-digital-elevation-map-processing-visualization-in-python-4bb7aa65f2b1
    elevation_array = np.array(elevation)
    elevation_2d = elevation_array.reshape(np.unique(X_1d).size, np.unique(Y_1d).size)
    dx, dy = np.gradient(elevation_2d)
    grad_tot = np.hypot(dx, dy)

    contourf_gradient = plt.contourf(X, Y, grad_tot, levels, cmap=cmap_name, alpha=1, vmin=ll,
                                     vmax=ul, linestyles='dashed')
    # Convert matplotlib contourf to geojson
    geojson_gradient = geojsoncontour.contourf_to_geojson(
        contourf=contourf_gradient,
        stroke_width=0)

    fg_grad = folium.FeatureGroup(name="Gradient", show=False)
    base_map.add_child(fg_grad)

    folium.GeoJson(
        geojson_gradient,
        style_function=lambda x: {
            # 'weight': x['properties']['stroke-width'],
            'fillColor': x['properties']['fill'],
            'opacity': 0.5,  # does not work
            'weight': 0.4
        }).add_to(fg_grad)

    # compute slope

    slopes = np.degrees(np.arctan(np.hypot(dy, dx)))
    min_slope = np.ceil(np.min(slopes))
    max_slope = np.ceil(np.max(slopes))
    contourf_slopes = plt.contourf(X, Y, slopes, levels, cmap=cmap_name, alpha=1, vmin=min_slope,
                                   vmax=max_slope, linestyles='dashed')
    # Convert matplotlib contourf to geojson
    geojson_slopes = geojsoncontour.contourf_to_geojson(
        contourf=contourf_slopes,
        stroke_width=0)

    fg_slope = folium.FeatureGroup(name="Slopes [deg]", show=False)
    base_map.add_child(fg_slope)

    folium.GeoJson(
        geojson_slopes,
        style_function=lambda x: {
            # 'weight': x['properties']['stroke-width'],
            'fillColor': x['properties']['fill'],
            'opacity': 0.5,  # does not work
            'weight': 0.4
        }).add_to(fg_slope)

    return base_map


def get_hex_values(cmap_name, levels):
    cmap = cm.get_cmap(cmap_name, levels)  # PiYG
    hex_values = []
    for i in range(cmap.N):
        rgba = cmap(i)
        # rgb2hex accepts rgb or rgba
        # print(matplotlib.colors.rgb2hex(rgba))
        hex_values.append(matplotlib.colors.rgb2hex(rgba))
    return hex_values


def process(args):
    lon_center = args.coordinates[0]
    lat_center = args.coordinates[1]
    search_radius = args.radius_km

    # define geographical coordinates of corners of search area
    search_area = define_search_area(lon_center, lat_center, search_radius)

    # query overpass to get coordinates of roads, hunting stands, etc.
    feature_coordinates = query_overpass(search_area)

    # define even grid of coordinates inside search area
    X, Y = define_search_grid(search_area, args.res_m / 1000.0)
    grid_coordinates = np.vstack((X.flatten(), Y.flatten())).T
    n_lon, n_lat = X.shape

    # compute distance for each grid point from feature coordinates
    # n_average: average over n computed distances and do not rely on the closest feature
    Z = compute_distance_from_features(grid_coordinates, feature_coordinates, n_lon, n_lat, n_average=5)

    # generate basic map
    base_map = generate_base_map(search_radius, location=[lat_center, lon_center], tiles="OpenStreetMap")

    # add external websites
    base_map = add_central_marker(base_map, lon_center, lat_center, search_radius)

    # generate layer of found features
    base_map = generate_layer_features(base_map, feature_coordinates)

    # generate layer of distance contour plot
    ul = 300  # everything above 300m is considered to be OK anyway
    base_map = generate_layer_distance(base_map, X, Y, Z, ll=0, ul=ul)

    if args.topo:
        base_map = generate_layer_gradient_slope(base_map, X, Y, ll=0, ul=20)

    # add coordinates of a gpx file if provided
    if args.gpx:
        base_map = add_gpx_track(base_map, args.gpx)

    folium.LayerControl(collapsed=False).add_to(base_map)

    # Save the map
    fout_tmp = f'_lon{np.round(lon_center, 5)}_lat{np.round(lat_center, 5)}' \
               f'_radius{np.round(search_radius, 3)}_res{np.round(args.res_m, 3)}_dist{np.round(ul, 3)}.html'
    if not args.area_name:
        fout_name = 'map' + fout_tmp
    else:
        fout_name = f'{args.area_name}' + fout_tmp
    base_map.save(fout_name)


def main(args):
    process(args)
    sys.exit(0)


if __name__ == '__main__':
    main(create_parser().parse_args())


# import pdb; pdb.set_trace()
########################################################

# The color bar issue has to be adressed later


# color names
# https://matplotlib.org/3.1.0/gallery/color/named_colors.html

# color maps
# https://matplotlib.org/3.1.0/tutorials/colors/colormaps.html

# colors = ['#d7191c', '#fdae61', '#ffffbf', '#abdda4', '#2b83ba']
# colors = ['maroon','red','darkorange','orange','yellow',
#          'greenyellow','limegreen','forestgreen']

# colors = ['blue','royalblue', 'navy','pink',  'mediumpurple',  'darkorchid',  'plum',  'm', 'mediumvioletred', 'palevioletred', 'crimson',
#         'magenta','pink','red','yellow','orange', 'brown','green', 'darkgreen']

# levels = len(colors) # without -1 the display would not be correct
# levels = 30  # np.linspace(0, 1000, 100)


# # workaround to setup color bar with right colors in folium
# hex_values = get_hex_values(cmap_name, levels)
# # create color map
# cm_dist = branca.colormap.LinearColormap(hex_values, vmin=0, vmax=vmax).to_step(levels)
# # Add the colormap to the folium map
# cm_dist.caption = 'Distance [meter]'
# geomap.add_child(cm_dist)
