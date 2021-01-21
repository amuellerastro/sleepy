sleepy
------

A small script to find isolated places in the forest. Its query relies on openstreetmap data, which might be incomplete
or wrong or outdated, especially in the area inside a forest. This is a first approximation only since roads are defined
by individual nodes which can be separated by many meters.

Author: Andre Mueller

Usage
-----

The user has to prove initial coordinates (longitude, latitude) in degrees, the search radius in kilometer, and the 
spatial resolution of the search grid.

Output
------

The final output is a html file called folium_contour_map.html. It shows an interactive map centered around the user
coordinates. A contour plot shows for each location inside the user defined area the approximate distance to roads, 
paths, tracks, and hunting stands.

![Example output](https://github.com/amuellerastro/sleepy/blob/main/example.png?raw=true)


Dependencies
------------

python-numpy

python-matplotlib

python-branca

python-astropy

python-folium

python-geojsoncontour

python-requests

