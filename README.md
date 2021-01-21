sleepy
------

A small script to find isolated places in the forest.

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


Dependencies
------------

python-numpy

python-matplotlib

python-branca
python-astropy
python-folium
python-geojsoncontour
python-requests

