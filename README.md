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

    -area_name     Name of the area. Has no influence on the provided coordinates. This paramter is just added to the 
                    file name of the output files.
    -radius_km     Search radius around the provided coordinates in kilometer (default=0.8).
    -res_m         The spatial resolution of the search grid in meter (default=100).
    -cmap          The name of the color map used to plot the contour map (default='gist_gray')
    -min_dist      The minimum distance from buildings, roads, hunting stands to be considered acceptable in meter 
                    (default=300).
    -gpx           Load a .gpx file. Track will be overplotted. Default: no gpx file is loaded.
    -nplaces       If >0 then n markers at position of largest distance are placed with popus containing a link to 
                    google maps satellite images. The zoom level is identical to the original folium map zoom level.
                    Default=0.


Output
------

The final output is a html file called "{map or area_name}_lon{}_lat{}_radius{}_res_{}_dist{}.html" with the main used 
parameters in the filename. It shows an interactive map centered around the user
coordinates. If "area_name" is set this value will appear at the beginning of the file name. Otherwise, the file name start 
with "map". A contour plot shows for each location inside the user defined area the approximate distance to roads, 
paths, tracks, and hunting stands. In addition, a map with the identified objects and the provided user coordinates is 
provided. The file name of this map is identical to the one of the final map but with "_tmp.html" added to the end.

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
