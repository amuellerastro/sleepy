sleepy
------

A script to find isolated places in the forest. Its query relies on openstreetmap data, which might be incomplete
or wrong or outdated, especially in the area inside a forest. This is a first approximation only since roads are defined
by individual nodes which can be separated by many meters.

Author: Andre Mueller

Usage
-----

The user has to provide initial coordinates (longitude, latitude) in degrees, the search radius in kilometer, and the 
spatial resolution of the search grid. A value for the color map and a minimum distance to the objects can be provided
as well.

```
python sleepy.py -lon <deg> -lat <deg>
```

The available options are:

    -area_name     Name of the area. Has no influence on the provided coordinates. 
                    This paramter is just added to the file name of the output files.
    -radius_km     Search radius around the provided coordinates in kilometer (default=0.8).
    -res_m         The spatial resolution of the search grid in meter (default=100).
    -cmap          The name of the color map used to plot the contour map (default='gist_gray')
    -min_dist      The minimum distance from buildings, roads, hunting stands to be considered 
                    acceptable in meter (default=300).
    -gpx           Load a .gpx file. Track will be overplotted. Default: no gpx file is loaded.
    -topo          If True (default=False) the elevation of the defined grid of coordinates is queried
                    via an API from a local setup server and data set 
                    (see https://www.opentopodata.org/ for details). Depending on the grid size and resolution
                    this query can take up several minutes.


Output
------

The final output is a html file called "{map or area_name}_lon{}_lat{}_radius{}_res_{}_dist{}.html" with the main used 
parameters in the filename. It shows an interactive map centered around the provided
coordinates. If "area_name" is set this value will appear at the beginning of the file name. Otherwise, the file name start 
with "map". On the right-hand side of the map different markers, layers, and contours can be plotted over the map.
E.g. The contour plot 'Distance' shows for each location inside the user defined area the approximate distance to roads, 
paths, tracks, and hunting stands. If 'Features' is selected the coordinates of identified features inside the search area 
are displayed on the map.

![Example output](https://github.com/amuellerastro/sleepy/blob/main/example.png?raw=true)


Dependencies
------------

   1. python-numpy
   1. python-matplotlib
   1. python-branca
   1. python-folium
   1. python-geojsoncontour
   1. python-requests
   1. python-argparse
   1. python-gpxpy
   1. python-tqdm
