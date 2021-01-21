# To Do:
# add color bar, check if vmax checks out
# contour opacity - no solution yet
# load gpx file and overlay?

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
import json



def generateBaseMap(location=[0, 0], zoom_start=12, tiles="OpenStreetMap"):
    base_map = folium.Map(location=location, control_scale=True, zoom_start=zoom_start, tiles=tiles)
    return base_map

# coordinates of area of interest
coords = [8.266133, 48.443269] #longitude, latitude
# coords = [15.585038, 78.202115]
# coords = [12.118435, 50.529979]
# coords = [-38, 69]

# search radius in km
search_radius = 1.5

# resolution of the grid in meter
grid_resolution_meter = 100

grid_resolution = grid_resolution_meter/1000

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

# define search box based on inputs and correct for latitude

# circular segment
alpha = np.degrees(search_radius / R_earth.to(u.km).value)

# correction for geographical latitude
lat_corr = (np.cos(np.radians(coords[1])))

search_box = [coords[1]-alpha, coords[0]-alpha/lat_corr, coords[1]+alpha, coords[0]+alpha/lat_corr]

# print(search_box)


# define query

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
# out center;

# print(overpass_query)

response = requests.get(overpass_url,
                        params={'data': overpass_query})
data_overpass = response.json()

# for element in data['elements']:
#    print(element['id'], element['center']['lat'])

# with open('data.json', 'w') as f:
#    json.dump(data_overpass, f, indent=2)

# count = 0
# for element in data['elements']:
#    count += 1
# print(f"Found {count} points.")

# print(len(data['elements']))
# print(len(data.keys()))


# Collect coords into list
all_coords = []
for element in data_overpass['elements']:
    if element['type'] == 'node':
        lon = element['lon']
        lat = element['lat']
        all_coords.append((lon, lat))
#  elif 'center' in element:
#    lon = element['center']['lon']
#    lat = element['center']['lat']
#    coords.append((lon, lat))

# Convert coordinates into numpy array
found_points = np.array(all_coords)
plt.figure(figsize=(10, 10))
plt.plot(found_points[:, 0] * lat_corr, found_points[:, 1], 'o')
plt.title('Nodes of Roads')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.axis('equal')
plt.show()

# display found nodes on map


# OpenStreetMap
# Stamen Terrain
# cartodbpositron

# base_map = folium.Map(location = [coords[1], coords[0]] , zoom_start = zoom_level,  control_scale=True)
base_map = generateBaseMap(location=[coords[1], coords[0]], zoom_start = zoom_level)

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

# base_map

# create a 2D array regularly gridded, based on the required resulotion, e.g. 20m
# at the moment take the center latitude to compute the correction factor
# for longitude because the dimension of the search box are small

# circular segment from grid resolution
alpha_grid = np.degrees(grid_resolution / R_earth.to(u.km).value)
#print(alpha_grid)

# add and subtract alpha_grid to give some margin at the edges when creating the grid
# following computation of X, Y based on
# https://stackoverflow.com/questions/32208359/is-there-a-multi-dimensional-version-of-arange-linspace-in-numpy
# grid = np.mgrid[search_box[1]-alpha_grid:search_box[3]+alpha_grid:alpha_grid,
#                 search_box[0]-alpha_grid:search_box[2]+alpha_grid:alpha_grid].reshape(2,-1).T
X,Y = np.mgrid[(search_box[1]-alpha_grid):(search_box[3]+alpha_grid):alpha_grid,
               search_box[0]-alpha_grid:search_box[2]+alpha_grid:alpha_grid]
grid = np.vstack((X.flatten(), Y.flatten())).T
n_lon, n_lat = X.shape

# print(search_box)
# print(len(grid))

#plt.figure(figsize=(10,10))
#plt.plot(grid[:, 0], grid[:, 1], 'o')
#plt.title('Grid')
#plt.xlabel('Longitude')
#plt.ylabel('Latitude')
#plt.axis('equal')
#plt.show()

for lon, lat in grid:
    #label = folium.Popup('{} ({}): {} - Cluster {}'.format(bor, post, poi, cluster), parse_html=True)
    folium.CircleMarker(
        [lat, lon],
        radius=1,
        #popup=label,
        color='white',
        fill=True,
        fill_color='white',
        fill_opacity=0.4,
        parse_html=False).add_to(base_map)
base_map


# compute distance

# https://stackoverflow.com/questions/19413259/efficient-way-to-calculate-distance-matrix-given-latitude-and-longitude-data-in

def spherical_dist(pos1, pos2, r=3958.75):

    pos1 = pos1 * np.pi / 180
    pos2 = pos2 * np.pi / 180
    cos_lat1 = np.cos(pos1[..., 0])
    cos_lat2 = np.cos(pos2[..., 0])
    cos_lat_d = np.cos(pos1[..., 0] - pos2[..., 0])
    cos_lon_d = np.cos(pos1[..., 1] - pos2[..., 1])
    return r * np.arccos(cos_lat_d - cos_lat1 * cos_lat2 * (1 - cos_lon_d))


result = spherical_dist(grid[:,None], np.array(all_coords), r=R_earth.to(u.km).value)
#print(np.array(all_coords))
result.sort()

dist_km = []
for idx in range(0,len(result)):
    dist_km.append(np.mean(result[idx][0:5]))

#dist_km.sort()
#plt.figure(figsize=(10,5))
#plt.plot(dist_km, 'o')
#plt.show()
dist_km = np.array(dist_km)

Z = dist_km.reshape(n_lon, n_lat)

Z_meter = Z * 1e3

# color names
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
#colors = ['#a50026', '#a70226', '#a90426', '#ab0626', '#ad0826', '#af0926', '#b10b26', '#b30d26', '#b50f26', '#b71126', '#b91326', '#bb1526', '#bd1726', '#be1827', '#c01a27', '#c21c27', '#c41e27', '#c62027', '#c82227', '#ca2427', '#cc2627', '#ce2827', '#d02927', '#d22b27', '#d42d27', '#d62f27', '#d83128', '#d93429', '#da362a', '#db382b', '#dc3b2c', '#dd3d2d', '#de402e', '#e0422f', '#e14430', '#e24731', '#e34933', '#e44c34', '#e54e35', '#e65036', '#e75337', '#e95538', '#ea5739', '#eb5a3a', '#ec5c3b', '#ed5f3c', '#ee613e', '#ef633f', '#f16640', '#f26841', '#f36b42', '#f46d43', '#f47044', '#f57245', '#f57547', '#f57748', '#f67a49', '#f67c4a', '#f67f4b', '#f7814c', '#f7844e', '#f8864f', '#f88950', '#f88c51', '#f98e52', '#f99153', '#f99355', '#fa9656', '#fa9857', '#fa9b58', '#fb9d59', '#fba05b', '#fba35c', '#fca55d', '#fca85e', '#fcaa5f', '#fdad60', '#fdaf62', '#fdb163', '#fdb365', '#fdb567', '#fdb768', '#fdb96a', '#fdbb6c', '#fdbd6d', '#fdbf6f', '#fdc171', '#fdc372', '#fdc574', '#fdc776', '#fec877', '#feca79', '#fecc7b', '#fece7c', '#fed07e', '#fed27f', '#fed481', '#fed683', '#fed884', '#feda86', '#fedc88', '#fede89', '#fee08b', '#fee18d', '#fee28f', '#fee491', '#fee593', '#fee695', '#fee797', '#fee999', '#feea9b', '#feeb9d', '#feec9f', '#feeda1', '#feefa3', '#fff0a6', '#fff1a8', '#fff2aa', '#fff3ac', '#fff5ae', '#fff6b0', '#fff7b2', '#fff8b4', '#fffab6', '#fffbb8', '#fffcba', '#fffdbc', '#fffebe', '#feffbe', '#fdfebc', '#fbfdba', '#fafdb8', '#f8fcb6', '#f7fcb4', '#f5fbb2', '#f4fab0', '#f2faae', '#f1f9ac', '#eff8aa', '#eef8a8', '#ecf7a6', '#ebf7a3', '#e9f6a1', '#e8f59f', '#e6f59d', '#e5f49b', '#e3f399', '#e2f397', '#e0f295', '#dff293', '#ddf191', '#dcf08f', '#daf08d', '#d9ef8b', '#d7ee8a', '#d5ed88', '#d3ec87', '#d1ec86', '#cfeb85', '#cdea83', '#cbe982', '#c9e881', '#c7e77f', '#c5e67e', '#c3e67d', '#c1e57b', '#bfe47a', '#bde379', '#bbe278', '#b9e176', '#b7e075', '#b5df74', '#b3df72', '#b1de71', '#afdd70', '#addc6f', '#abdb6d', '#a9da6c', '#a7d96b', '#a5d86a', '#a2d76a', '#a0d669', '#9dd569', '#9bd469', '#98d368', '#96d268', '#93d168', '#91d068', '#8ecf67', '#8ccd67', '#89cc67', '#87cb67', '#84ca66', '#82c966', '#7fc866', '#7dc765', '#7ac665', '#78c565', '#75c465', '#73c264', '#70c164', '#6ec064', '#6bbf64', '#69be63', '#66bd63', '#63bc62', '#60ba62', '#5db961', '#5ab760', '#57b65f', '#54b45f', '#51b35e', '#4eb15d', '#4bb05c', '#48ae5c', '#45ad5b', '#42ac5a', '#3faa59', '#3ca959', '#39a758', '#36a657', '#33a456', '#30a356', '#2da155', '#2aa054', '#279f53', '#249d53', '#219c52', '#1e9a51', '#1b9950', '#199750', '#18954f', '#17934e', '#16914d', '#15904c', '#148e4b', '#138c4a', '#128a49', '#118848', '#108647', '#0f8446', '#0e8245', '#0d8044', '#0c7f43', '#0b7d42', '#0a7b41', '#097940', '#08773f', '#07753e', '#06733d', '#05713c', '#04703b', '#036e3a', '#026c39', '#016a38', '#006837'
#]


cmap_name = 'gist_gray' #'seismic' 'RdYlGn'

# workaround to setup color bar with right colors in folium
cmap = cm.get_cmap(cmap_name, 30)    # PiYG
hex_values = []
for i in range(cmap.N):
    rgba = cmap(i)
    # rgb2hex accepts rgb or rgba
    # print(matplotlib.colors.rgb2hex(rgba))
    hex_values.append(matplotlib.colors.rgb2hex(rgba))


vmin = 0 #np.min(Z_meter)
vmax = 300 #np.max(Z_meter)
#levels = len(colors) # without -1 the display would not be correct
levels = 30 #np.linspace(0, 1000, 100)
levels2 = np.linspace(0, np.max(Z_meter), 30)

cm = branca.colormap.LinearColormap(hex_values, vmin=vmin, vmax=vmax).to_step(levels)

# x_mesh, y_mesh = np.meshgrid(x_orig, y_orig)
# z_mesh = griddata((x_orig, y_orig), z_orig, (x_mesh, y_mesh), method='linear')

# Gaussian filter the grid to make it smoother
#sigma = [2, 2]
#z_mesh = sp.ndimage.filters.gaussian_filter(z_mesh, sigma, mode='constant')

# Create the contour
plt.figure(figsize=(10,8))
contourf = plt.contourf(X, Y, Z_meter, levels, cmap=cmap_name, alpha=1, vmin=0,
                        vmax=vmax, linestyles='dashed') #, colors=colors
plt.colorbar();

# Convert matplotlib contourf to geojson
geojson = geojsoncontour.contourf_to_geojson(
    contourf=contourf,
    #min_angle_deg=3.0,
    #ndigits=5,
    stroke_width=0)
    #fill_opacity=0.1)

# Set up the folium plot
#geomap = folium.Map([df.latitude.mean(), df.longitude.mean()], zoom_start=10, tiles="cartodbpositron")
geomap = generateBaseMap(location=[coords[1], coords[0]], zoom_start = zoom_level)

# Plot the contour plot on folium
folium.GeoJson(
    geojson,
    style_function=lambda x: {
        #'color': x['properties']['stroke'],
        'weight': x['properties']['stroke-width'],
        'fillColor': x['properties']['fill'],
        'opacity': 0.5,  #does not work
        'weight': 0.4
    }).add_to(geomap)


# def style_function_opa(self):
#     return {'fillopacity': 0.0}

# folium.GeoJson(geojson,
#     style_function=style_function_opa).add_to(geomap)

# Add the colormap to the folium map
cm.caption = 'Distance [meter]'
geomap.add_child(cm)

# Fullscreen mode
plugins.Fullscreen(position='topright', force_separate_button=True).add_to(geomap)

# Plot the data
geomap.save(f'folium_contour_map.html')
geomap
