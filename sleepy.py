# https://towardsdatascience.com/loading-data-from-openstreetmap-with-python-and-the-overpass-api-513882a27fd0

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
import branca
from astropy import units as u
from astropy.constants import R_earth, c
import folium
from folium import plugins
import geojsoncontour
import requests
import argparse
import gpxpy
import gpxpy.gpx


def generate_base_map(location=[0, 0], zoom_start=12, tiles="OpenStreetMap"):
    base_map = folium.Map(location=location, control_scale=True, zoom_start=zoom_start, tiles=tiles)
    return base_map


def query_overpass(search_box):
    overpass_url = "http://overpass-api.de/api/interpreter"
    overpass_query = f"""
    [out:json];
    (
      way["highway"]({search_box[0]}, {search_box[1]}, {search_box[2]}, {search_box[3]});
      node["amenity"="hunting_stand"]({search_box[0]}, {search_box[1]}, {search_box[2]}, {search_box[3]});
      way["amenity"="hunting_stand"]({search_box[0]}, {search_box[1]}, {search_box[2]}, {search_box[3]});
      relation["amenity"="hunting_stand"]({search_box[0]}, {search_box[1]}, {search_box[2]}, {search_box[3]});
      node["amenity"="shelter"]({search_box[0]}, {search_box[1]}, {search_box[2]}, {search_box[3]});
      way["amenity"="shelter"]({search_box[0]}, {search_box[1]}, {search_box[2]}, {search_box[3]});
      relation["amenity"="shelter"]({search_box[0]}, {search_box[1]}, {search_box[2]}, {search_box[3]});
      way["building"="hut"]({search_box[0]}, {search_box[1]}, {search_box[2]}, {search_box[3]});

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


def coordinates_from_overpass(data_overpass):
    # Collect coords into list
    all_coords = []
    for element in data_overpass['elements']:
        if element['type'] == 'node':
            lon = element['lon']
            lat = element['lat']
            all_coords.append((lon, lat))

    return all_coords


def spherical_dist(pos1, pos2, r=3958.75):
    pos1 = pos1 * np.pi / 180
    pos2 = pos2 * np.pi / 180
    cos_lat1 = np.cos(pos1[..., 0])
    cos_lat2 = np.cos(pos2[..., 0])
    cos_lat_d = np.cos(pos1[..., 0] - pos2[..., 0])
    cos_lon_d = np.cos(pos1[..., 1] - pos2[..., 1])
    return r * np.arccos(cos_lat_d - cos_lat1 * cos_lat2 * (1 - cos_lon_d))


def get_hex_values(cmap_name, levels):
    cmap = cm.get_cmap(cmap_name, levels)  # PiYG
    hex_values = []
    for i in range(cmap.N):
        rgba = cmap(i)
        # rgb2hex accepts rgb or rgba
        # print(matplotlib.colors.rgb2hex(rgba))
        hex_values.append(matplotlib.colors.rgb2hex(rgba))
    return hex_values


def add_gpx_track(gpx, geomap):
    gpx_file = open(gpx, 'r')

    gpx = gpxpy.parse(gpx_file)
    points = []
    for track in gpx.tracks:
        for segment in track.segments:
            for point in segment.points:
                points.append([point.latitude, point.longitude])

    for gpx_coord in points:
        folium.CircleMarker(gpx_coord,
                            radius=1,
                            color='blue',
                            fill_color='blue',
                            fill=True,
                            parse_html=False).add_to(geomap)

    return geomap


def add_googlemaps(nplaces, geomap, X, Y, Z_meter):
    X_1d = X.flatten()
    Y_1d = Y.flatten()
    Z_1d = Z_meter.flatten()
    idxs = sorted(range(len(Z_1d)), key=lambda k: Z_1d[k], reverse=True)

    for idx in range(0,nplaces):
        url = "https://www.google.com/maps/@"+str(Y_1d[idxs[idx]])+","+str(X_1d[idxs[idx]])+","+str(zoom_level)+"z/data=!3m1!1e3"
        link_text = "Google Maps"
        tmp_dist = round(Z_1d[idxs[idx]])
        label = folium.Html('<a href="' + url + '"target="_blank">' + link_text + '</a>', script=True)
        popup = folium.Popup(label, parse_html=True)
        folium.Marker(
            [Y_1d[idxs[idx]], X_1d[idxs[idx]]],
            popup=popup).add_to(geomap)
    return geomap


def generate_tmp_map(coords, all_coords, zoom_level):
    base_map = generate_base_map(location=[coords[1], coords[0]], zoom_start = zoom_level)
    # Marker of the coordinates provided by user
    label = folium.Popup("User Coordinates Lon: {} Lat: {}".format(round(coords[0],2), round(coords[1],2)))
    folium.CircleMarker(
        [coords[1], coords[0]],
        popup=label,
        radius=5,
        fill=True,
        color='red',
        fill_color='red',
        fill_opacity=1).add_to(base_map)

    for lon, lat in all_coords:
        #label = folium.Popup('{} ({}): {} - Cluster {}'.format(bor, post, poi, cluster), parse_html=True)
        folium.CircleMarker(
            [lat, lon],
            radius=1,
            #popup=label,
            #color='blue',
            fill=True,
            fill_color='#3186cc',
            fill_opacity=0.4,
            parse_html=False).add_to(base_map)

    base_map.save('tmp.html')



parser = argparse.ArgumentParser()
parser.add_argument('-lon', default=8.266133)
parser.add_argument('-lat', default=48.443269)
parser.add_argument('-radius_km', default=0.8)
parser.add_argument('-res_m', default=100)
parser.add_argument('-cmap', default='seismic')
parser.add_argument('-min_dist', default=300)
parser.add_argument('-gpx', default=False)
parser.add_argument('-nplaces', default=0)

args = parser.parse_args()
coords = [float(args.lon), float(args.lat)]
search_radius = float(args.radius_km)
grid_resolution_meter = float(args.res_m)
cmap_name = str(args.cmap)
vmax = float(args.min_dist)
gpx = args.gpx
nplaces = int(args.nplaces)

# coordinates of area of interest
# coords = [8.266133, 48.443269] #longitude, latitude
# coords = [15.585038, 78.202115]
# coords = [12.118435, 50.529979]
# coords = [-38, 69]

# search radius in km
# search_radius = 1.5

# resolution of the grid in meter
# grid_resolution_meter = 100

# color map
# cmap_name = 'gist_gray' #'seismic' 'RdYlGn'

vmin = 0
# vmax = 300

grid_resolution = grid_resolution_meter / 1000

# define initial zoom level of map

if search_radius <= 0.5:
    zoom_level = 15
elif search_radius > 0.5 and search_radius <= 1.5:
    zoom_level = 14
elif search_radius > 1.5 and search_radius <= 2.5:
    zoom_level = 13
elif search_radius > 2.5 and search_radius <= 5:
    zoom_level = 12
else:
    zoom_level = 11

# circular segment
alpha = np.degrees(search_radius / R_earth.to(u.km).value)

# correction for geographical latitude
lat_corr = (np.cos(np.radians(coords[1])))

search_box = [coords[1] - alpha, coords[0] - alpha / lat_corr, coords[1] + alpha, coords[0] + alpha / lat_corr]

# query overpass to get coordinates of roads and hunting stands
all_coords = query_overpass(search_box)

# map showing result from query and central coordinates
generate_tmp_map(coords, all_coords, zoom_level)

# create a 2D array regularly gridded, based on the required resulotion, e.g. 20m
# at the moment take the center latitude to compute the correction factor
# for longitude because the dimension of the search box are small

# circular segment from grid resolution
alpha_grid = np.degrees(grid_resolution / R_earth.to(u.km).value)

# add and subtract alpha_grid to give some margin at the edges when creating the grid
# following computation of X, Y based on
# https://stackoverflow.com/questions/32208359/is-there-a-multi-dimensional-version-of-arange-linspace-in-numpy
# grid = np.mgrid[search_box[1]-alpha_grid:search_box[3]+alpha_grid:alpha_grid,
#                 search_box[0]-alpha_grid:search_box[2]+alpha_grid:alpha_grid].reshape(2,-1).T
X, Y = np.mgrid[(search_box[1] - alpha_grid):(search_box[3] + alpha_grid):alpha_grid,
       search_box[0] - alpha_grid:search_box[2] + alpha_grid:alpha_grid]
grid = np.vstack((X.flatten(), Y.flatten())).T
n_lon, n_lat = X.shape

# for lon, lat in grid:
#     folium.CircleMarker(
#         [lat, lon],
#         radius=1,
#         #popup=label,
#         color='white',
#         fill=True,
#         fill_color='white',
#         fill_opacity=0.4,
#         parse_html=False).add_to(base_map)
# base_map

# compute distance
# https://stackoverflow.com/questions/19413259/efficient-way-to-calculate-distance-matrix-given-latitude-and-longitude-data-in
result = spherical_dist(grid[:, None], np.array(all_coords), r=R_earth.to(u.km).value)
# print(np.array(all_coords))
result.sort()

dist_km = []
for idx in range(0, len(result)):
    dist_km.append(np.mean(result[idx][0:5]))

dist_km = np.array(dist_km)

Z = dist_km.reshape(n_lon, n_lat)
Z_meter = Z * 1e3

#color names
# https://matplotlib.org/3.1.0/gallery/color/named_colors.html

# color maps
# https://matplotlib.org/3.1.0/tutorials/colors/colormaps.html

# colors = ['#d7191c', '#fdae61', '#ffffbf', '#abdda4', '#2b83ba']
# colors = ['maroon','red','darkorange','orange','yellow',
#          'greenyellow','limegreen','forestgreen']

# colors = ['blue','royalblue', 'navy','pink',  'mediumpurple',  'darkorchid',  'plum',  'm', 'mediumvioletred', 'palevioletred', 'crimson',
#         'magenta','pink','red','yellow','orange', 'brown','green', 'darkgreen']

# hex values from colorbar:
# import matplotlib
# import matplotlib.cm
# cmap = cm.get_cmap('seismic', 5)    # PiYG
# for i in range(cmap.N):
#     rgba = cmap(i)
#     # rgb2hex accepts rgb or rgba
#     print(matplotlib.colors.rgb2hex(rgba))

# levels = len(colors) # without -1 the display would not be correct
levels = 30  # np.linspace(0, 1000, 100)

# workaround to setup color bar with right colors in folium
hex_values = get_hex_values(cmap_name, levels)

# create color map
cm = branca.colormap.LinearColormap(hex_values, vmin=vmin, vmax=vmax).to_step(levels)

# x_mesh, y_mesh = np.meshgrid(x_orig, y_orig)
# z_mesh = griddata((x_orig, y_orig), z_orig, (x_mesh, y_mesh), method='linear')

# Gaussian filter the grid to make it smoother
# sigma = [2, 2]
# z_mesh = sp.ndimage.filters.gaussian_filter(z_mesh, sigma, mode='constant')

# Create the contour
# plt.figure(figsize=(10,8))
contourf = plt.contourf(X, Y, Z_meter, levels, cmap=cmap_name, alpha=1, vmin=0,
                        vmax=vmax, linestyles='dashed')  # , colors=colors
# plt.colorbar();


# Convert matplotlib contourf to geojson
geojson = geojsoncontour.contourf_to_geojson(
    contourf=contourf,
    # min_angle_deg=3.0,
    # ndigits=5,
    stroke_width=0)
# fill_opacity=0.1)

# Set up the folium plot
# geomap = folium.Map([df.latitude.mean(), df.longitude.mean()], zoom_start=10, tiles="cartodbpositron")
geomap = generate_base_map(location=[coords[1], coords[0]], zoom_start=zoom_level)

# Plot the contour plot on folium
folium.GeoJson(
    geojson,
    style_function=lambda x: {
        # 'color': x['properties']['stroke'],
        'weight': x['properties']['stroke-width'],
        'fillColor': x['properties']['fill'],
        'opacity': 0.5,  # does not work
        'weight': 0.4
    }).add_to(geomap)

# Add the colormap to the folium map
cm.caption = 'Distance [meter]'
geomap.add_child(cm)

# Fullscreen mode
plugins.Fullscreen(position='topright', force_separate_button=True).add_to(geomap)

# add makers with link to google maps satellite images
if nplaces > 0:
    geomap = add_googlemaps(nplaces, geomap, X, Y, Z_meter)

# load GPX file
if gpx:
    geomap = add_gpx_track(gpx, geomap)

# Plot the data
geomap.save(
    f'folium_contour_map_lon{np.round(coords[0], 5)}_lat{np.round(coords[1], 5)}'
    f'_radius{np.round(search_radius, 3)}_res{np.round(grid_resolution_meter, 3)}'
    f'_dist{np.round(vmax, 3)}.html')
# geomap
