from datetime import datetime
import os
import re

import coda
import harp
import numpy as np
import pyproj
import requests
from scipy.interpolate import griddata

from . import vis

from .vis import Plot, MapPlot, MapPlot3D

_UNPREFERED_PATTERNS = [
    "index",
    "collocation_index",
    "orbit_index",
    ".*subindex",
    "scan_direction_type",
    "datetime.*",
    "sensor_.*",
    ".*validity",
    ".*_uncertainty.*",
    ".*_apriori.*",
    ".*_amf.*",
    "wavenumber$",
    "wavelength$",
    ".*latitude.*",
    ".*longitude.*",
    ".*altitude",
    ".*geoid_height",
    ".*geopotential",
    ".*pressure",
    ".*_angle"
]

"""
Atmosphere Virtual Lab

A toolkit for interactive plotting of atmospheric data.

Given a Harp product and variable name, it extracts data as well as meta-data
to automatically produce an annotated data trace.

The following types of data traces are currently supported:

- Scatter
- Line
- Histogram
- Heatmap
- Geo
- Geo3D
- Volume

There are three types of plots that can also be individually instantiated, and
populated with (compatible) data traces:

- Plot
- MapPlot
- MapPlot3D

Data traces are in themselves also plots (with a single data trace), so
can be shown interactively without requiring a separate plot.

Example usage:

    avl.Scatter(product, 'variable name')

Additional keyword arguments are passed to the underlying plot:

    avl.Geo(product, 'var1', centerlat=90)

Combining data traces:

    plot = avl.MapPlot()
    plot.add(avl.Geo(product, 'var1'))
    plot.add(avl.Geo(product, 'var2'))
    plot

"""


def download(files, target_directory="."):
    """
    Download file(s) from `atmospherevirtuallab.org`, skipping files
    that already exist.

    Arguments:
    files -- file name or list/tuple of file names
    target_directory -- path where to store files (default '.')
    """
    if isinstance(files, (list, tuple)):
        return [download(file) for file in files]
    filename = os.path.basename(files)
    targetpath = os.path.join(target_directory, filename)
    if not os.path.exists(targetpath):
        url = "https://atmospherevirtuallab.org/files/" + filename
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(os.path.join(target_directory, filename), 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
    return filename


class _objdict(dict):
    def __getattr__(self, name):
        if name in self:
            return self[name]
        else:
            raise AttributeError("No such attribute: " + name)

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        if name in self:
            del self[name]
        else:
            raise AttributeError("No such attribute: " + name)


def _get_prefered_value(values, unprefered_patterns=_UNPREFERED_PATTERNS):
    if unprefered_patterns:
        non_matching = [v for v in values if not re.match(unprefered_patterns[0], v)]
        value = _get_prefered_value(non_matching, unprefered_patterns[1:])
        if value is not None:
            return value
    if values:
        return values[0]
    return None


def _get_attributes(product):
    # we return all scalars and 1D (time dependent) variables
    def attr_value(value, variable):
        if hasattr(variable, "unit"):
            if " since " in variable.unit:
                # this is a time value
                base, epoch = variable.unit.split(" since ")
                if base in ["s", "seconds", "days"]:
                    if base == "days":
                        value *= 86400
                    formats = ("yyyy-MM-dd HH:mm:ss.SSSSSS|"
                               "yyyy-MM-dd HH:mm:ss|"
                               "yyyy-MM-dd")
                    value = value + coda.time_string_to_double(formats, epoch)
                    return coda.time_to_string(value)
            return "%s [%s]" % (str(value), variable.unit)
        return str(value)

    attr = {}
    for name in list(product):
        if len(product[name].dimension) == 0:
            attr[name] = attr_value(product[name].data, product[name])
        elif (len(product[name].dimension) == 1 and
              product[name].dimension[0] == 'time'):
            attr[name] = [attr_value(value, product[name]) for value in product[name].data]

    return attr


def _get_midpoint_axis_from_bounds(bounds_variable, log=False):
    if (bounds_variable.data.shape[-1] != 2 or
            bounds_variable.dimension[-1] is not None):
        raise ValueError("bounds variable should end with independent"
                         "dimension of length 2")
    if log:
        data = np.exp((np.log(bounds_variable.data[..., 0]) +
                       np.log(bounds_variable.data[..., 1])) / 2.0)
    else:
        data = ((bounds_variable.data[..., 0] +
                 bounds_variable.data[..., 1]) / 2.0)
    return harp.Variable(data, bounds_variable.dimension[:-1], bounds_variable.unit)


def _plot_data(product, value=None, average=False):
    if not isinstance(product, harp.Product):
        raise TypeError("Expecting a HARP product")

    variable_names = []

    for name in list(product):
        if len(product[name].dimension) == 0 or len(product[name].dimension) > 2:
            continue
        if not isinstance(product[name].data, (np.ndarray, np.generic)):
            continue
        if product[name].data.dtype.char in ['O', 'S']:
            continue
        if product[name].dimension[0] != 'time':
            continue
        if (len(product[name].dimension) == 2 and
                product[name].dimension[1] not in ['spectral', 'vertical']):
            continue
        variable_names += [name]

    if value is not None:
        if value not in list(product):
            raise ValueError("product variable does not exist ('%s')" % value)
        if value not in variable_names:
            raise ValueError("product variable is not plottable ('%s')" % value)
    else:
        value = _get_prefered_value(variable_names)

    if value is None:
        raise ValueError("HARP product is not plotable")

    xdata = None
    ydata = product[value]
    attr = {}
    location = []
    prop = {'title': value.replace('_', ' '), 'name': value}

    if len(product[value].dimension) == 2:
        if product[value].dimension[1] == 'spectral':
            if 'wavelength' in product:
                xdata = product['wavelength']
            elif 'wavenumber' in product:
                xdata = product['wavenumber']
            if xdata is None:
                raise ValueError("Could not determine x-axis for spectral"
                                 " data ('%s')" % value)
        else:
            # product[value].dimension[1] == 'vertical'
            # swap axis
            xdata = ydata
            ydata = None
            if 'altitude' in product:
                ydata = product['altitude']
            elif 'altitude_bounds' in product:
                ydata = _get_midpoint_axis_from_bounds(product['altitude_bounds'])
            elif 'pressure' in product:
                ydata = product['pressure']
                prop["ylog"] = True
            elif 'pressure_bounds' in product:
                ydata = _get_midpoint_axis_from_bounds(product['pressure_bounds'], log=True)
                prop["ylog"] = True
            if ydata is None:
                raise ValueError("Could not determine y-axis for vertical"
                                 " profile data ('%s')" % value)
        attr = _get_attributes(product)
    else:
        if 'datetime' in product:
            xdata = product['datetime']
        elif 'datetime_start' in product:
            xdata = product['datetime_start']
        elif 'datetime_stop' in product:
            xdata = product['datetime_stop']
        if xdata is None:
            raise ValueError("Could not determine x-axis for time-series"
                             " data ('%s')" % value)

    if 'latitude_bounds' in product and 'longitude_bounds' in product:
        location = [product.latitude_bounds.data, product.longitude_bounds.data]
    elif 'latitude' in product and 'longitude' in product:
        location = [product.latitude.data, product.longitude.data]

    try:
        xunit = xdata.unit
    except AttributeError:
        xunit = None
    try:
        yunit = ydata.unit
    except AttributeError:
        yunit = None

    xlabel = None
    ylabel = None
    xerror = None
    yerror = None
    coords = None
    colorlabel = None

    if xdata.dimension == ['time', 'vertical']:  # TODO generalize
        xlabel = 'time'
        ylabel = 'altitude (km)'
        colorlabel = xunit

        xdata_dt = np.empty(len(product.datetime.data), dtype='datetime64[s]')  # TODO ns?
        offset = (datetime(2000, 1, 1) - datetime(1970, 1, 1)).total_seconds()
        xdata_dt[:] = (product.datetime.data * 24 * 60 * 60) + offset

        coords = (xdata_dt, product.altitude.data)

    xdata = xdata.data
    ydata = ydata.data

    if xunit is not None:
        if xunit == 'seconds since 2010-01-01':  # TODO generalize (start of epoch.. more formats?)
            xdata_dt = np.empty(len(xdata), dtype='datetime64[s]')  # TODO ns?
            offset = (datetime(2010, 1, 1) - datetime(1970, 1, 1)).total_seconds()
            xdata_dt[:] = xdata + offset
            xdata = xdata_dt
        if xunit == 'days since 2000-01-01':
            xdata_dt = np.empty(len(product.datetime.data), dtype='datetime64[s]')  # TODO ns?
            offset = (datetime(2000, 1, 1) - datetime(1970, 1, 1)).total_seconds()
            xdata_dt[:] = (product.datetime.data * 24 * 60 * 60) + offset
            xdata = xdata_dt
        elif xlabel is None:
            xlabel = xunit

    if yunit is not None and ylabel is None:
        ylabel = yunit

    # average along time dim # TODO checks, y, nanaverage?
    if average:
        xerror = np.nanstd(xdata, 0)
        xdata = np.nanmean(xdata, 0)
        xlabel = xunit

    if value + '_uncertainty' in list(product):
        yerror = product[value + '_uncertainty'].data

    return _objdict(**{
        'xdata': xdata,
        'xerror': xerror,
        'yerror': yerror,
        'ydata': ydata,
        'xlabel': xlabel,
        'ylabel': ylabel,
        'colorlabel': colorlabel,
        'title': prop.get('title'),
        'coords': coords,
    })


kPointData = 0
kSwathData = 1
kGridData = 2


def _mapplot_data(product, value=None, locationOnly=False):
    if not isinstance(product, harp.Product):
        raise TypeError("Expecting a HARP product")

    data_type = kPointData
    latitude = None
    longitude = None

    if ('latitude_bounds' in list(product) and
            'longitude_bounds' in list(product)):

        latitude_bounds = product['latitude_bounds']
        longitude_bounds = product['longitude_bounds']

        if len(latitude_bounds.dimension) != len(longitude_bounds.dimension):
            raise ValueError("latitude and longitude bounds should have same"
                             " number of dimensions")

        if (latitude_bounds.dimension[-1] is not None or
                longitude_bounds.dimension[-1] is not None):
            raise ValueError("last dimension for latitude and longitude bounds"
                             " should be independent")

        if (len(latitude_bounds.dimension) > 1 and
                latitude_bounds.dimension[-2] == 'latitude' and
                longitude_bounds.dimension[-2] == 'longitude'):

            if len(latitude_bounds.dimension) == 3:
                if (latitude_bounds.dimension[0] != 'time' or
                        longitude_bounds.dimension[0] != 'time'):
                    raise ValueError("first dimension for latitude and longitude"
                                     " bounds should be the time dimension")
            else:
                if len(latitude_bounds.dimension) != 2:
                    raise ValueError("latitude and longitude bounds should be"
                                     " two or three dimensional for gridded data")

            if (latitude_bounds.data.shape[-1] != 2 or
                    longitude_bounds.data.shape[-1] != 2):
                raise ValueError("independent dimension of latitude and longitude"
                                 " bounds should have length 2 for gridded data")
            data_type = kGridData
            latitude = _get_midpoint_axis_from_bounds(latitude_bounds)
            longitude = _get_midpoint_axis_from_bounds(longitude_bounds)

        elif not locationOnly:
            if len(latitude_bounds.dimension) != 2:
                raise ValueError("latitude and longitude bounds should be two"
                                 " dimensional for non-gridded data")

            if (latitude_bounds.dimension[0] != 'time' or
                    longitude_bounds.dimension[0] != 'time'):
                raise ValueError("first dimension for latitude and longitude"
                                 " bounds should be the time dimension")

            if latitude_bounds.data.shape != longitude_bounds.data.shape:
                raise ValueError("latitude and longitude bounds should have"
                                 " the same dimension lengths")

            data_type = kSwathData
            latitude = latitude_bounds
            longitude = longitude_bounds

    if (data_type != kSwathData and
            'latitude' in list(product) and
            'longitude' in list(product)):

        latitude = product['latitude']
        longitude = product['longitude']

        if len(latitude.dimension) != len(longitude.dimension):
            raise ValueError("latitude and longitude should have same number"
                             " of dimensions")

        if (len(latitude.dimension) > 0 and
                latitude.dimension[-1] == 'latitude' and
                longitude.dimension[-1] == 'longitude'):

            # if we have both lat/lon center and lat/lon bounds then use the center lat/lon for the grid axis
            if len(latitude.dimension) == 2:
                if (latitude.dimension[0] != 'time' or
                        longitude.dimension[0] != 'time'):
                    raise ValueError("first dimension for latitude and longitude"
                                     " should be the time dimension")
            else:
                if len(latitude.dimension) != 1:
                    raise ValueError("latitude and longitude should be one or two"
                                     " dimensional")
            data_type = kGridData

        else:
            if len(latitude.dimension) != 1:
                raise ValueError("latitude and longitude should be one"
                                 " dimensional")

            if (latitude.dimension[0] != 'time' or
                    longitude.dimension[0] != 'time'):
                raise ValueError("first dimension for latitude and longitude"
                                 " should be the time dimension")
            data_type = kPointData

    if latitude is None or longitude is None:
        raise ValueError("HARP product has no latitude/longitude information")

    if locationOnly:
        value = None
    else:
        variable_names = []

        for name in list(product):
            if not isinstance(product[name].data, (np.ndarray, np.generic)):
                continue
            if product[name].data.dtype.char in ['O', 'S']:
                continue
            if data_type == kGridData:
                if len(product[name].dimension) < 2 or len(product[name].dimension) > 3:
                    continue
                if product[name].dimension[-2] != 'latitude':
                    continue
                if product[name].dimension[-1] != 'longitude':
                    continue
                if len(product[name].dimension) == 3 and product[name].dimension[0] != 'time':
                    continue
            else:
                if len(product[name].dimension) != 1:
                    continue
                if product[name].dimension[0] != 'time':
                    continue
            variable_names += [name]

        if value is not None:
            if value not in list(product):
                raise ValueError("product variable does not exist ('%s')" % value)
            if value not in variable_names:
                raise ValueError("product variable is not plottable ('%s')" % value)
        else:
            value = _get_prefered_value(variable_names)

    data = None
    attr = {}
    prop = {}

    if data_type == kGridData:
        attr = _get_attributes(product)
        if value is None:
            raise ValueError("HARP product has no variable to use for gridded plot")

    if value is not None:
        data = product[value].data
        prop['name'] = value
        prop['colorbartitle'] = value.replace('_', ' ')
        try:
            prop['colorbartitle'] += ' [' + product[value].unit + ']'
        except AttributeError:
            pass

    return _objdict(**{
        'data': data,
        'longitude': longitude.data,
        'latitude': latitude.data,
        'data_type': data_type,
    })


def volume_data(product, value, spherical=False, **kwargs):
    if not isinstance(product, harp.Product):
        raise TypeError("Expecting a HARP product")

    data = product[value].data

    if spherical:  # TODO add earth somehow
        source_crs = pyproj.CRS('epsg:4326')
        target_crs = pyproj.crs.GeocentricCRS('epsg:6326')

        trans = pyproj.Transformer.from_crs(source_crs, target_crs)

        points = []
        values = []

        lons = product.longitude.data  # TODO what if we only have *_bounds? (eg after rebin)
        lats = product.latitude.data
        alts = product.altitude.data

        altmin = min(alts)
        altmax = max(alts)

        for ilat, lat in enumerate(lats):
            for ilon, lon in enumerate(lons):
                for ialt, alt in enumerate(alts):
                    value = data[ilat][ilon][ialt]  # TODO assuming (lat,lon,alt) dimension (order)
                    point = list(trans.itransform([[lat, lon, ((alt - altmin) / (altmax - altmin)) * 3e6]]))[0]
                    points.append(point)
                    values.append(value)

        # TODO better solution (add outer 'nan' layers to data? or arg to interp engine?)
        # add inner layer, to make 'nearest' not fill up earth..
        for ilon in np.arange(0, 360):
            for ilat in np.arange(0, 180):
                value = np.nan
                point = list(trans.itransform([[ilat - 90, ilon, -1e5]]))[0]
                points.append(point)
                values.append(value)

        # outer layer
        for ilon in np.arange(0, 360):
            for ilat in np.arange(0, 180):
                value = np.nan
                point = list(trans.itransform([[ilat - 90, ilon, 3.1e6]]))[0]  # TODO
                points.append(point)
                values.append(value)

        grid_x, grid_y, grid_z = np.mgrid[-1e7:1e7:100j, -1e7:1e7:100j, -1e7:1e7:100j]  # TODO configurable
        data = griddata(points, values, (grid_x, grid_y, grid_z), method='nearest')

    return _objdict(**{
        'data': data,
    })


# TODO is 'value' the right name for a Harp variable?

def Volume(product, value, **kwargs):
    """
    Return a Volume data trace for the given Harp variable.

    Nan-values are converted to the lowest non-nan value.

    Volume data traces cannot currently be combined in a single plot!

    Arguments:
    product -- Harp product
    value -- Harp variable name
    spherical -- Project data to a sphere (default False)

    """
    data = volume_data(product, value, **kwargs)
    return vis.VolumePlot(**data, **kwargs)


def histogram_data(product, value, **kwargs):
    if not isinstance(product, harp.Product):
        raise TypeError("Expecting a HARP product")

    var = product[value]

    return {
        'data': var.data,
        'title': value.replace('_', ' '),
        'xlabel': var.unit,
        'ylabel': var.unit,
    }


def Histogram(product, value, **kwargs):
    """
    Return a Histogram data trace for the given Harp variable.

    Compatible plot type: `Plot`

    Arguments:
    product -- Harp product
    value -- Harp variable name
    bins -- Number of bins

    """
    data = histogram_data(product, value)
    return vis.Histogram(**data, **kwargs)


def scatter_data(product, value, average=False, **kwargs):
    return _plot_data(product, value, average=average)


def Scatter(product, value, **kwargs):
    """
    Return a Scatter data trace for the given Harp variable.

    Compatible plot type: `Plot`

    Arguments:
    product -- Harp product
    value -- Harp variable name

    """
    data = scatter_data(product, value)
    return vis.Scatter(**data, **kwargs)


def line_data(product, value, average=False, **kwargs):
    return _plot_data(product, value, average=average)


def Line(product, value, **kwargs):
    """
    Return a Line data trace for the given Harp variable.

    Compatible plot type: `Plot`

    Arguments:
    product -- Harp product
    value -- Harp variable name

    """
    data = line_data(product, value)
    return vis.Line(**data, **kwargs)


def heatmap_data(product, value, **kwargs):
    data = _plot_data(product, value)

    data['data'] = data['xdata']
    del data['xdata']

    return data


def Heatmap(product, value, **kwargs):
    """
    Return a Heatmap data trace for the given Harp variable.

    Compatible plot type: `Plot`

    Arguments:
    product -- Harp product
    value -- Harp variable name
    colormap -- Colormap name (matplotlib) or list of (x,r,g,b,a) values (0..1)
    gap_threshold -- Add gaps when larger (np.timedelta, default 24h)

    """
    data = heatmap_data(product, value)
    return vis.Heatmap(**data, **kwargs)


def geo_data(product, value, **kwargs):
    return _mapplot_data(product, value)


def Geo(product, value, **kwargs):
    """
    Return a Geo data trace for the given Harp variable.

    Compatible plot type: `MapPlot`

    Arguments:
    product -- Harp product
    value -- Harp variable name

    """
    data = geo_data(product, value)
    return vis.Geo(**data, **kwargs)


def geo3d_data(product, value, **kwargs):
    return _mapplot_data(product, value)


def Geo3D(product, value, **kwargs):
    """
    Return a Geo3D data trace for the given Harp variable.

    Compatible plot type: `MapPlot3D`

    Arguments:
    product -- Harp product
    value -- Harp variable name

    """
    data = geo3d_data(product, value)
    return vis.Geo3D(**data, **kwargs)
