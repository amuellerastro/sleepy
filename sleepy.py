#!/usr/bin/env python3
# https://towardsdatascience.com/loading-data-from-openstreetmap-with-python-and-the-overpass-api-513882a27fd0

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
import branca
from astropy import units as u
from astropy.constants import R_earth
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
        url_topo =  "https://opentopomap.org/#map=" + str(zoom_level) + "/" + str(Y_1d[idxs[idx]]) + "/" + str(X_1d[idxs[idx]])
        link_text_topo = "Topo Map"

        tmp_dist = str(round(Z_1d[idxs[idx]]))
        tmp_dist_str = f'Dist. {tmp_dist} m'
        label = folium.Html(tmp_dist_str+' / '+'<a href="' + url + '"target="_blank">' + link_text + '</a>' +' / '+
                            '<a href="' + url_topo + '"target="_blank">' + link_text_topo + '</a>', script=True)
        popup = folium.Popup(label, max_width=500, parse_html=True)
        folium.Marker(
            [Y_1d[idxs[idx]], X_1d[idxs[idx]]],
            popup=popup).add_to(geomap)
    return geomap


def generate_tmp_map(coords, all_coords, zoom_level, fout_name):
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

    base_map.save(fout_name)


parser = argparse.ArgumentParser()
parser.add_argument('-area_name', default='')
parser.add_argument('-lon', default=8.266133)
parser.add_argument('-lat', default=48.443269)
parser.add_argument('-radius_km', default=0.8)
parser.add_argument('-res_m', default=100)
parser.add_argument('-cmap', default='seismic')
parser.add_argument('-min_dist', default=300)
parser.add_argument('-gpx', default=False)
parser.add_argument('-nplaces', default=0)

args = parser.parse_args()
area_name = str(args.area_name)
coords = [float(args.lon), float(args.lat)]
search_radius = float(args.radius_km)
grid_resolution_meter = float(args.res_m)
cmap_name = str(args.cmap)
vmax = float(args.min_dist)
gpx = args.gpx
nplaces = int(args.nplaces)




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
R_earth = 6378.1  #R_earth.to(u.km).value from astropy.constants
alpha = np.degrees(search_radius / R_earth)

# correction for geographical latitude
lat_corr = (np.cos(np.radians(coords[1])))

search_box = [coords[1] - alpha, coords[0] - alpha / lat_corr, coords[1] + alpha, coords[0] + alpha / lat_corr]

# query overpass to get coordinates of roads and hunting stands
all_coords = query_overpass(search_box)

# map showing result from query and central coordinates
if area_name:
    fout_name_tmp = f'{area_name}_lon{np.round(coords[0], 5)}_lat{np.round(coords[1], 5)}' \
            f'_radius{np.round(search_radius, 3)}_res{np.round(grid_resolution_meter, 3)}_dist{np.round(vmax, 3)}_tmp.html'
else:
    fout_name_tmp = f'map_lon{np.round(coords[0], 5)}_lat{np.round(coords[1], 5)}' \
    f'_radius{np.round(search_radius, 3)}_res{np.round(grid_resolution_meter, 3)}_dist{np.round(vmax, 3)}_tmp.html'
generate_tmp_map(coords, all_coords, zoom_level, fout_name_tmp)

# create a 2D array regularly gridded, based on the required resulotion, e.g. 20m
# at the moment take the center latitude to compute the correction factor
# for longitude because the dimension of the search box are small

# circular segment from grid resolution
alpha_grid = np.degrees(grid_resolution / R_earth)

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
result = spherical_dist(grid[:, None], np.array(all_coords), r=R_earth)
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
cm = branca.colormap.LinearColormap(hex_values, vmin=0, vmax=vmax).to_step(levels)

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

##################################################
##################################################
##################################################

from tqdm import trange
import pandas as pd


# https://stackoverflow.com/questions/55827778/elevation-xyz-data-to-slope-gradient-map-using-python
def window3x3(arr, shape=(3, 3)):
    r_win = np.floor(shape[0] / 2).astype(int)
    c_win = np.floor(shape[1] / 2).astype(int)
    x, y = arr.shape
    for i in range(x):
        xmin = max(0, i - r_win)
        xmax = min(x, i + r_win + 1)
        for j in range(y):
            ymin = max(0, j - c_win)
            ymax = min(y, j + c_win + 1)
            yield arr[xmin:xmax, ymin:ymax]


def gradient(XYZ_file, min=0, max=15, figsize=(15, 10), **kwargs):

    """

    :param XYZ_file: XYZ file in the following format: x,y,z (inlcuding headers)
    :param min: color bar minimum range.
    :param max: color bar maximum range.
    :param figsize: figure size.
    :param kwargs:
           plot: to plot a gradient map. Default is True.
    :return: returns an array with the shape of the grid with the computed slopes


    The algorithm calculates the gradient using a first-order forward or backward difference on the corner points, first
    order central differences at the boarder points, and a 3x3 moving window for every cell with 8 surrounding cells (in
    the middle of the grid) using a third-order finite difference weighted by reciprocal of squared distance

    Assumed 3x3 window:

                        -------------------------
                        |   a   |   b   |   c   |
                        -------------------------
                        |   d   |   e   |   f   |
                        -------------------------
                        |   g   |   h   |   i   |
                        -------------------------


    """

    kwargs.setdefault('plot', True)

    grid = XYZ_file.to_numpy()

    nx = XYZ_file['x'].unique().size
    ny = XYZ_file['y'].unique().size
#     nx = len(X)
#     ny = len(Y)

    xs = grid[:, 0].reshape(ny, nx, order='F')
    ys = grid[:, 1].reshape(ny, nx, order='F')
    zs = grid[:, 2].reshape(ny, nx, order='F')
    # xs = X.reshape(ny, nx, order='F')
    # ys = Y.reshape(ny, nx, order='F')
    # zs = Z.reshape(ny, nx, order='F')
    dx = abs((xs[:, 1:] - xs[:, :-1]).mean())
    dy = abs((ys[1:, :] - ys[:-1, :]).mean())

    gen = window3x3(zs)
    windows_3x3 = np.asarray(list(gen))
    windows_3x3 = windows_3x3.reshape(ny, nx)

    dzdx = np.empty((ny, nx))
    dzdy = np.empty((ny, nx))
    loc_string = np.empty((ny, nx), dtype="S25")

    for ax_y in trange(ny):
        for ax_x in range(nx):

            # corner points
            if ax_x == 0 and ax_y == 0:  # top left corner
                dzdx[ax_y, ax_x] = (windows_3x3[ax_y, ax_x][0][1] - windows_3x3[ax_y, ax_x][0][0]) / dx
                dzdy[ax_y, ax_x] = (windows_3x3[ax_y, ax_x][1][0] - windows_3x3[ax_y, ax_x][0][0]) / dy
                loc_string[ax_y, ax_x] = 'top left corner'

            elif ax_x == nx - 1 and ax_y == 0:  # top right corner
                dzdx[ax_y, ax_x] = (windows_3x3[ax_y, ax_x][0][1] - windows_3x3[ax_y, ax_x][0][0]) / dx
                dzdy[ax_y, ax_x] = (windows_3x3[ax_y, ax_x][1][1] - windows_3x3[ax_y, ax_x][0][1]) / dy
                loc_string[ax_y, ax_x] = 'top right corner'

            elif ax_x == 0 and ax_y == ny - 1:  # bottom left corner
                dzdx[ax_y, ax_x] = (windows_3x3[ax_y, ax_x][1][1] - windows_3x3[ax_y, ax_x][1][0]) / dx
                dzdy[ax_y, ax_x] = (windows_3x3[ax_y, ax_x][1][0] - windows_3x3[ax_y, ax_x][0][0]) / dy
                loc_string[ax_y, ax_x] = 'bottom left corner'

            elif ax_x == nx - 1 and ax_y == ny - 1:  # bottom right corner
                dzdx[ax_y, ax_x] = (windows_3x3[ax_y, ax_x][1][1] - windows_3x3[ax_y, ax_x][1][0]) / dx
                dzdy[ax_y, ax_x] = (windows_3x3[ax_y, ax_x][1][1] - windows_3x3[ax_y, ax_x][0][1]) / dy
                loc_string[ax_y, ax_x] = 'bottom right corner'

            # top boarder
            elif (ax_y == 0) and (ax_x != 0 and ax_x != nx - 1):
                dzdx[ax_y, ax_x] = (windows_3x3[ax_y, ax_x][0][-1] - windows_3x3[ax_y, ax_x][0][0]) / (2 * dx)
                dzdy[ax_y, ax_x] = (windows_3x3[ax_y, ax_x][1][1] - windows_3x3[ax_y, ax_x][0][1]) / dy
                loc_string[ax_y, ax_x] = 'top boarder'

            # bottom boarder
            elif ax_y == ny - 1 and (ax_x != 0 and ax_x != nx - 1):
                dzdx[ax_y, ax_x] = (windows_3x3[ax_y, ax_x][1][-1] - windows_3x3[ax_y, ax_x][1][0]) / (2 * dx)
                dzdy[ax_y, ax_x] = (windows_3x3[ax_y, ax_x][1][1] - windows_3x3[ax_y, ax_x][0][1]) / dy
                loc_string[ax_y, ax_x] = 'bottom boarder'

            # left boarder
            elif ax_x == 0 and (ax_y != 0 and ax_y != ny - 1):
                dzdx[ax_y, ax_x] = (windows_3x3[ax_y, ax_x][1][1] - windows_3x3[ax_y, ax_x][1][0]) / dx
                dzdy[ax_y, ax_x] = (windows_3x3[ax_y, ax_x][-1][0] - windows_3x3[ax_y, ax_x][0][0]) / (2 * dy)
                loc_string[ax_y, ax_x] = 'left boarder'

            # right boarder
            elif ax_x == nx - 1 and (ax_y != 0 and ax_y != ny - 1):
                dzdx[ax_y, ax_x] = (windows_3x3[ax_y, ax_x][1][1] - windows_3x3[ax_y, ax_x][1][0]) / dx
                dzdy[ax_y, ax_x] = (windows_3x3[ax_y, ax_x][-1][-1] - windows_3x3[ax_y, ax_x][0][-1]) / (2 * dy)
                loc_string[ax_y, ax_x] = 'right boarder'

            # middle grid
            else:
                a = windows_3x3[ax_y, ax_x][0][0]
                b = windows_3x3[ax_y, ax_x][0][1]
                c = windows_3x3[ax_y, ax_x][0][-1]
                d = windows_3x3[ax_y, ax_x][1][0]
                f = windows_3x3[ax_y, ax_x][1][-1]
                g = windows_3x3[ax_y, ax_x][-1][0]
                h = windows_3x3[ax_y, ax_x][-1][1]
                i = windows_3x3[ax_y, ax_x][-1][-1]

                dzdx[ax_y, ax_x] = ((c + 2 * f + i) - (a + 2 * d + g)) / (8 * dx)
                dzdy[ax_y, ax_x] = ((g + 2 * h + i) - (a + 2 * b + c)) / (8 * dy)
                loc_string[ax_y, ax_x] = 'middle grid'

    hpot = np.hypot(abs(dzdy), abs(dzdx))
    slopes_angle = np.degrees(np.arctan(hpot))
    if kwargs['plot']:
        slopes_angle[(slopes_angle < min) | (slopes_angle > max)]

        plt.figure(figsize=figsize)
        # plt.pcolormesh(xs, ys, slopes_angle, cmap=plt.cm.gist_yarg, vmax=max, vmin=min)
        plt.pcolormesh(xs, ys, slopes_angle, cmap=plt.cm.gist_yarg, vmax=np.max(slopes_angle), vmin=np.min(slopes_angle))

        plt.colorbar()
        plt.tight_layout()
        plt.show()

    return slopes_angle


# elevation
# https://www.opentopodata.org/api/
X_1d = X.flatten()
Y_1d = Y.flatten()

elevation = []
for i in range(0,len(X_1d)):
    #print(i, len(X_1d))
    topo_url = f"http://localhost:5000/v1/eudem25m?locations={Y_1d[i]},{X_1d[i]}"
    response = requests.get(topo_url)
    data_topo = response.json()
    #print(data_topo['results'][0]['elevation'])
    elevation.append(data_topo['results'][0]['elevation'])

#np.savetxt("foo.csv", np.transpose(np.array([X_1d, Y_1d, elevation])), delimiter=",")
XYZ = pd.read_csv('foo.csv')
slopes = gradient(XYZ)
#slopes = gradient(X_1d, Y_1d, elevation)

##################################################
##################################################
##################################################


# add makers with link to google maps satellite images
if nplaces > 0:
    geomap = add_googlemaps(nplaces, geomap, X, Y, Z_meter)

# load GPX file
if gpx:
    geomap = add_gpx_track(gpx, geomap)

if not area_name:
    fout_name  = f'map_lon{np.round(coords[0], 5)}_lat{np.round(coords[1], 5)}' \
    f'_radius{np.round(search_radius, 3)}_res{np.round(grid_resolution_meter, 3)}_dist{np.round(vmax, 3)}.html'
# Save the map
else:
    fout_name = f'{area_name}_lon{np.round(coords[0], 5)}_lat{np.round(coords[1], 5)}' \
            f'_radius{np.round(search_radius, 3)}_res{np.round(grid_resolution_meter, 3)}_dist{np.round(vmax, 3)}.html'
geomap.save(fout_name)
# geomap
