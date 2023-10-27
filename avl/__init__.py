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
- Curtain
- Geo
- Geo3D
- Volume

There are three types of plots that can also be individually instantiated, and
populated with (compatible) data traces:

- Plot (Histogram, Scatter, Line, Heatmap and Curtain)
- MapPlot (Geo)
- MapPlot3D (Geo3D)

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


def _get_prefered_value(values, unprefered_patterns=_UNPREFERED_PATTERNS):
    if unprefered_patterns:
        non_matching = [v for v in values if not re.match(unprefered_patterns[0], v)]
        value = _get_prefered_value(non_matching, unprefered_patterns[1:])
        if value is not None:
            return value
    if values:
        return values[0]
    return None


def _get_product_value(product, value, dims=(1, 2)):
    if not isinstance(product, harp.Product):
        raise TypeError("Expecting a HARP product")

    variable_names = []

    for name in list(product):
        if len(product[name].dimension) not in dims:
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

    return value


def _get_datetime(product):

    if 'datetime' in product:
        output = product['datetime']
    elif 'datetime_start' in product:
        output = product['datetime_start']
    elif 'datetime_stop' in product:
        output = product['datetime_stop']
    if output is None:
        raise ValueError("Could not determine x-axis for time-series"
                         " data.")

    return output


def _process_product_data(product, value, prop):

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
            xdata = product[value]
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
    else:
        ydata = product[value]
        xdata = _get_datetime(product)

    if 'latitude_bounds' in product and 'longitude_bounds' in product:
        location = [product.latitude_bounds.data, product.longitude_bounds.data]
    elif 'latitude' in product and 'longitude' in product:
        location = [product.latitude.data, product.longitude.data]

    if xdata.dimension == ['time', 'vertical']:  # TODO generalize
        xlabel = 'time'
        ylabel = 'altitude (%s)' % ydata.unit
        datetime = _get_datetime(product)
        # colorlabel = xunit
        xdata_dt = get_timestamps(datetime)
        coords = (xdata_dt, ydata.data)
    else:
        xlabel = xdata.unit
        ylabel = ydata.unit
        coords = None

    if value + '_uncertainty' in list(product):
        yerror = product[value + '_uncertainty'].data
    else:
        yerror = None

    return xdata, ydata, xlabel, ylabel, coords, yerror


def _plot_data(arc1, arc2, average=False):

    xlabel = None
    ylabel = None
    xerror = None
    yerror = None
    coords = None
    colorlabel = None

    if isinstance(arc1, harp.Variable) and arc2 is None:
        ydata = arc1
        xdata = np.arange(len(ydata.data))
        prop = {"title": f"{ydata.description}"}

    elif all(isinstance(x, harp.Variable) for x in [arc1, arc2]):

        ydata = arc1
        xdata = arc2
        if len(ydata.data) != len(xdata.data):
            raise ValueError("Both variables need the same data length")
        try:
            prop = {"title": f"{ydata.unit} vs {xdata.unit} cross-correlation"}
        except AttributeError:
            prop = {"title": "Harp variable cross-correlation"}

    elif isinstance(arc1, harp.Product) and isinstance(arc2, str):
        product = arc1
        value = arc2

        value = _get_product_value(product, value)
        prop = {'title': value.replace('_', ' '), 'name': value}
        xdata, ydata, xlabel, ylabel, coords, yerror = _process_product_data(product, value, prop)

    else:
        raise ValueError("Inputs must either be one or two Harp variables or a "
                         "Harp product with a value of the product as a string.")

    try:
        xunit = xdata.unit
        colorlabel = xunit
    except AttributeError:
        xunit = None
    try:
        yunit = ydata.unit
    except AttributeError:
        yunit = None

    if xunit is not None:
        if " since " in xunit:  # TODO check dimension instead
            xdata = get_timestamps(xdata)
        else:
            if xlabel is None:
                xlabel = xunit
            xdata = xdata.data
    else:
        xdata = xdata.data

    ydata = ydata.data

    if yunit is not None and ylabel is None:
        ylabel = yunit

    # average along time dim # TODO checks, y, nanaverage?
    if average:
        xerror = np.nanstd(xdata, 0)
        xdata = np.nanmean(xdata, 0)
        xlabel = xunit

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


def _mapplot_data(arc1, arc2, arc3, locationOnly=False):

    latitude_bounds = None
    latitude = None
    longitude = None
    longitude_bounds = None
    attr = {}
    prop = {}

    if isinstance(arc1, harp.Product) and isinstance(arc2, str):
        product = arc1
        value = arc2
        plot_data = product[value]
        if 'latitude_bounds' in list(product):
            latitude_bounds = product['latitude_bounds']
        if 'longitude_bounds' in list(product):
            longitude_bounds = product['longitude_bounds']
        if "latitude" in list(product):
            latitude = product["latitude"]
        if "longitude" in list(product):
            longitude = product["longitude"]

        if value is not None:
            plot_data = product[value]
            prop['name'] = value
            prop['colorbartitle'] = value.replace('_', ' ')
            try:
                prop['colorbartitle'] += ' [' + product[value].unit + ']'
            except AttributeError:
                pass

    elif all(isinstance(x, harp.Variable) for x in [arc1, arc2, arc3]):
        plot_data = arc3
        if "latitude" in arc1.description:
            if "corners" in arc1.description:
                latitude_bounds = arc1
            else:
                latitude = arc1
        else:
            raise ValueError("When using three HARP variables, first variable"
                             " (arc1) should contain a latitude component.")
        if "longitude" in arc2.description:
            if "corners" in arc2.description:
                longitude_bounds = arc2
            else:
                longitude = arc2
        else:
            raise ValueError("When using three HARP variables, second variable (arc2)"
                             " should contain a longitude component.")

        prop['name'] = "Harp Variable"
        prop['colorbartitle'] = f"Harp Variable - [ {arc3.unit} ]"
    else:
        raise TypeError("Expecting a HARP product with a valid string or three HARP "
                        "variables for the latitude component, longitudinal component and data to plot.")

    data_type = kPointData

    if (latitude_bounds is not None and
            longitude_bounds is not None):

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
            latitude is not None and
            longitude is not None):

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
        raise ValueError("Input combination of HARP product or HARP variable"
                         " is missing a latitude or a longitude component")

    if locationOnly:
        variable_names = []
        value = _get_prefered_value(variable_names)
    else:
        plotable = True
        if not isinstance(plot_data.data, (np.ndarray, np.generic)):
            plotable = False
        if plot_data.data.dtype.char in ['O', 'S']:
            plotable = False
        if data_type == kGridData:
            if len(plot_data.dimension) < 2 or len(plot_data.dimension) > 3:
                plotable = False
            if plot_data.dimension[-2] != 'latitude':
                plotable = False
            if plot_data.dimension[-1] != 'longitude':
                plotable = False
            if len(plot_data.dimension) == 3 and plot_data.dimension[0] != 'time':
                plotable = False
        else:
            if len(plot_data.dimension) != 1:
                plotable = False
            if plot_data.dimension[0] != 'time':
                plotable = False
            # variable_names += [name]
        if not plotable:
            raise ValueError("Combination of input data is not plotable")

    return _objdict(**{
        'data': plot_data.data,
        'longitude': longitude.data,
        'latitude': latitude.data,
        'colorlabel': prop.get('colorbartitle'),
    })


def volume_data(arc1, arc2, arc3, arc4, spherical=False):
    if isinstance(arc1, harp.Product) and isinstance(arc2, str):

        data = arc1[arc2].data
        product = arc1
        if spherical:
            lons = product.longitude.data  # TODO what if we only have *_bounds? (eg after rebin)
            lats = product.latitude.data
            alts = product.altitude.data

    elif all(isinstance(x, harp.Variable) for x in [arc1, arc2, arc3, arc4]) and spherical:
        data = arc1.data
        for item in [arc2, arc3, arc4]:
            if "latitude" in item.description:
                lats = item
            if "longitude" in item.description:
                lons = item
            if "altitude" in item.description:
                alts = item

    elif isinstance(arc1, harp.Variable):
        data = arc1.data
        if spherical:
            raise ValueError("Cannot plot a spherical Volume plot without four harp variables"
                             " for the following attributes: Data, Latitude, Longitude & Altitude")
    else:
        raise TypeError("Expecting either a HARP product with a valid string, a single HARP"
                        " variable or four HARP variables in the case of a spherical Volume plot.")

    if spherical:  # TODO add earth somehow
        source_crs = pyproj.CRS('epsg:4326')
        target_crs = pyproj.crs.GeocentricCRS('epsg:6326')

        trans = pyproj.Transformer.from_crs(source_crs, target_crs)

        points = []
        values = []

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

def Volume(arc1, arc2=None, arc3=None, arc4=None, spherical=False, **kwargs):
    """
    Return a Volume data trace for the given Harp variable.

    Nan-values are converted to the lowest non-nan value.

    Volume data traces cannot currently be combined in a single plot!

    Arguments:
    arc1 - HARP Product or a HARP variable representing a data array
    arc2 - String or a HARP variable reprensenting a latitude array
    arc3 - Third HARP variable representing a longitude array
    arc3 - Third HARP variable representing an altitude array
    spherical -- Project data to a sphere (default False)

    """
    data = volume_data(arc1, arc2, arc3, arc4, spherical)
    return vis.VolumePlot(**data, **kwargs)


def histogram_data(arc1, arc2, **kwargs):
    if isinstance(arc1, harp.Variable):
        var = arc1
        title = "Histogram of HARP Variable"
    elif isinstance(arc1, harp.Product) and isinstance(arc2, str):

        var = arc1[arc2]
        title = arc2.replace('_', ' ')

    else:
        raise TypeError("Expecting a single HARP variable or a HARP product with a value.")

    return {
        'data': var.data,
        'title': title,
        'ylabel': var.unit,
    }


def Histogram(arc1, arc2=None, **kwargs):
    """
    Return a Histogram data trace for the given inputs. Inputs can exist of a HARP product together
    with a string, representing a variable, or one HARP variable.
    Compatible plot type: `Plot`

    Arguments:
    arc1 - HARP Product or a HARP variable
    arc2 - String or second HARP Variale
    bins -- Number of bins

    """
    data = histogram_data(arc1, arc2)
    return vis.Histogram(**data, **kwargs)


def scatter_data(arc1, arc2, average=False, **kwargs):
    return _plot_data(arc1, arc2, average=average)


def Scatter(arc1, arc2=None, **kwargs):
    """
    Return a Scatter data trace for the given inputs. Inputs can exist of a HARP product together
    with a string, representing a variable, or one or two HARP variable.

    Compatible plot type: `Plot`

    Arguments:
    arc1 - HARP Product or a HARP variable
    arc2 - String or second HARP Variale

    """
    data = scatter_data(arc1, arc2)
    return vis.Scatter(**data, **kwargs)


def line_data(arc1, arc2, average=False, **kwargs):
    return _plot_data(arc1, arc2, average=average)


def Line(arc1, arc2=None, **kwargs):
    """
    Return a Line data trace for the given inputs. Inputs can exist of a HARP product together
    with a string, representing a variable, or one or two HARP variable.

    Compatible plot type: `Plot`

    Arguments:
    arc1 - HARP Product or a HARP variable
    arc2 - String or second HARP Variale

    """
    data = line_data(arc1, arc2)
    return vis.Line(**data, **kwargs)


def heatmap_data(arc1, arc2, arc3=None, **kwargs):  # TODO disentangle from _plot_data, avoid 'del' below in any case

    if isinstance(arc1, harp.Product) and isinstance(arc2, str):
        plot_data = _plot_data(arc1, arc2)
        data = {}
        print(plot_data, type(plot_data))
        data['data'] = plot_data['xdata']
        for item in plot_data:
            print(item)
            if item not in ["xdata", "ydata", "xerror", "yerror"]:
                data[item] = plot_data[item]

    elif all(isinstance(x, harp.Variable) for x in [arc1, arc2, arc3]):
        ydata = arc1
        time_data = arc2
        altitude_data = arc3
        if "since" not in arc2.unit:
            raise ValueError("When using three HARP variables, second input (arc2) should contain a time component.")
        if arc3.unit not in ["Pa", "m"]:
            raise ValueError("When using three HARP variables, third input (arc3) "
                             "should contain an altitude component.")

        xlabel = 'time'
        ylabel = 'altitude (%s)' % altitude_data.unit

        colorlabel = ydata.unit

        xdata_dt = get_timestamps(time_data)
        coords = (xdata_dt, altitude_data.data)

        data = {
            'data': ydata.data,
            'xlabel': xlabel,
            'ylabel': ylabel,
            'colorlabel': colorlabel,
            'title': ydata.unit,
            'coords': coords,
        }
        print(data, coords)
    else:
        raise TypeError("Expecting a HARP product and a value string or three "
                        "harp variables for the data, time and altitude axii")

    return data


def _calc_bounds(a):
    # calculate center points (z(i)+z(i+1))/2
    centers = (a[:, 1:] + a[:, :-1]) / 2

    # extrapolate and add outer boundaries
    lower_bound = ((3 * a[:, 0]) - a[:, 1]) / 2
    upper_bound = ((3 * a[:, -1]) - a[:, -2]) / 2

    lower = np.insert(centers, 0, lower_bound, 1)
    upper = np.insert(centers, centers.shape[1], upper_bound, 1)

    # stack to create new [lower, upper] dimension
    return np.stack([lower, upper], axis=2)


def get_timestamps(input):
    """
    Return a formatted datetime array based on the input's time series

    Arguments:
    input -- Harp product, Harp variable or Python dict containing an array of time and units

    Python dictionaries can be created and used as input which must contain a "data" and "unit" entry.

    """
    if type(input) is harp.Product:
        input = _get_datetime(input)

    if type(input) is dict:
        values = input["data"]
        unit = input["unit"]
    elif type(input) is harp.Variable:
        values = input.data
        unit = input.unit
    else:
        raise ValueError("Input must be a Harp product, Harp variable or a python dict")

    if " since " not in unit:
        raise ValueError("unsupported unit: %s" % unit)

    base, epoch = unit.split(" since ")

    if base not in ["s", "seconds", "days"]:
        raise ValueError("unsupported unit: %s" % unit)

    xdata_dt = np.empty(len(values), dtype='datetime64[us]')

    # TODO avoid dep on coda
    formats = ("yyyy-MM-dd HH:mm:ss.SSSSSS|"
               "yyyy-MM-dd HH:mm:ss|"
               "yyyy-MM-dd")
    offset = coda.time_string_to_double(formats, epoch) + (datetime(2000, 1, 1) - datetime(1970, 1, 1)).total_seconds()

    if base == "days":
        xdata_dt[:] = (values * 86400 + offset) * 1e6
    else:
        xdata_dt[:] = (values + offset) * 1e6

    return xdata_dt


def curtain_data(arc1, arc2, arc3, arc4, **kwargs):
    if isinstance(arc1, harp.Product) and isinstance(arc2, str):
        product = arc1
        value = _get_product_value(product, arc2, dims=(2,))
        data = product[value].data
        dimensions = product[value].dimension
        product_values = list(product)
        colorlabel = product[value].unit
        title = value.replace('_', ' ')
        invert_yaxis = False

        # derive datetime_start
        x_start = x_stop = None

        if 'datetime_start' in product_values:
            x_start = product.datetime_start.data
            x_unit = product.datetime_start.unit
        elif 'datetime_stop' in product_values and 'datetime' in product_values:
            x_start = product.datetime.data - (product.datetime_stop.data - product.datetime.data)
            x_unit = product.datetime_stop.unit  # TODO error if units don't match
        elif 'datetime_stop' in product_values and 'datetime_length' in product_values:
            x_start = product.datetime_stop.data - product.datetime_length.data
            x_unit = product.datetime_stop.unit
        elif 'datetime_stop' in product_values:
            x_stop = product.datetime_stop.data
            x_start = np.append((3 * x_stop[:1] - x_stop[1:2]) / 2, x_stop[:-1])  # TODO to helper func, also below
            x_unit = product.datetime_stop.unit
        elif 'datetime' in product_values and 'datetime_length' in product_values:
            x_start = product.datetime.data - (product.datetime_length.data / 2)
            x_unit = product.datetime.unit

        # derive datetime_stop
        if 'datetime_stop' in product_values:
            x_stop = product.datetime_stop.data
        elif 'datetime_start' in product_values and 'datetime' in product_values:
            x_stop = product.datetime.data + (product.datetime.data - product.datetime_start.data)
        elif 'datetime_start' in product_values and 'datetime_length' in product_values:
            x_stop = product.datetime_start.data + product.datetime_length.data
        elif 'datetime_start' in product_values:
            x_stop = np.append(x_start[1:], (3*x_start[-1:] - x_start[-2:-1])/2)
        elif 'datetime' in product_values and 'datetime_length' in product_values:
            x_stop = product.datetime.data + (product.datetime_length.data / 2)

        if x_start is None or x_stop is None:
            if 'datetime' in product_values:
                dts = product.datetime.data
                midpoints = dts[:-1] + (dts[1:] - dts[:-1]) / 2
                x_start = np.append((3 * dts[:1] - dts[1:2]) / 2, midpoints)
                x_stop = np.append(midpoints, (3 * dts[-1:] - dts[-2:-1]) / 2)
                x_unit = product.datetime.unit
            else:
                raise ValueError('cannot determine time boundaries')

        # derive bounds for 'vertical' dimension
        if dimensions[1] == 'vertical':
            for val in ('altitude_bounds', 'pressure_bounds', 'geopotential_height_bounds'):
                if val in product_values:
                    y = product[val].data
                    ylabel = '%s (%s)' % (val[:-7], product[val].unit)
                    if val == 'pressure_bounds':
                        invert_yaxis = True
                    break
            else:
                for val in ('altitude', 'pressure', 'geopotential_height'):
                    if val in product_values:
                        y = _calc_bounds(product[val].data)
                        ylabel = '%s (%s)' % (val, product[val].unit)
                        if val == 'pressure':
                            invert_yaxis = True
                        break
                else:
                    raise ValueError('cannot determine vertical boundaries')

        # derive bounds for 'spectral' dimension
        else:  # spectral
            for val in ('wavelength_bounds', 'wavenumber_bounds', 'frequency_bounds'):
                if val in product_values:
                    y = product[val].data
                    ylabel = '%s (%s)' % (val[:-7], product[val].unit)
                    break
            else:
                for val in ('wavelength', 'wavenumber', 'frequency'):
                    if val in product_values:
                        y = _calc_bounds(product[val].data)
                        ylabel = '%s (%s)' % (val, product[val].unit)
                        break
                else:
                    raise ValueError('cannot determine spectral boundaries')
    elif all(isinstance(x, harp.Variable) for x in [arc1, arc2, arc3, arc4]):
        title = "Curtain Diagram of Harp Variable"

        data = arc1.data
        colorlabel = arc1.unit
        date_start = None
        date_stop = None
        date_time = None
        date_length = None
        x_start = None
        x_stop = None
        y = None
        invert_yaxis = False
        ylabel = arc2.unit
        if "bounds" in arc2.description:
            y = arc2.data
        else:
            y = _calc_bounds(arc2.data)

        for item in [arc3, arc4]:

            if "start time" in item.dimension:
                date_start = item
            elif "stop time" in item.description:
                date_stop = item
            elif "duration" in item.description:
                date_length = item
            elif "time" in item.description:
                date_time = item

        if date_start is not None:
            x_start = date_start.data
            x_unit = date_start.unit
            if date_time is not None:
                x_stop = date_time.data + (date_time.data - date_start.data)

        if date_stop is not None:
            x_stop = date_stop.data
            x_unit = date_start.unit
            if date_time is not None:
                x_start = date_time.data - (date_stop.data - date_time.data)

        if date_length is not None:
            if date_start is not None:
                x_stop = date_start.data + date_length.data
            if date_stop is not None:
                x_start = date_stop.data - date_length.data
            if date_time is not None:
                x_unit = date_time.unit
                x_start = date_time.data - (date_length.data / 2)
                x_stop = date_time.data + (date_length.data / 2)

        if not all(x is not None for x in [y, x_start, x_stop]):
            raise ValueError(
                "Incorrect HARP variable inputs."
                " Inputs require one data set, two variables for determining the time range"
                " and one variable for a vertical / spectral component."
                )

    else:
        raise ValueError("Input must either be a HARP product wiht a value or four separate HARP variables.")
    # correct time stamps via unit

    x_data_start = {"data": x_start, "unit": x_unit}
    x_data_stop = {"data": x_stop, "unit": x_unit}
    xdata_start = get_timestamps(x_data_start)
    xdata_stop = get_timestamps(x_data_stop)

    # make x same shape as y (more flexible)  # TODO faster/nicer
    x = np.column_stack([xdata_start, xdata_stop])
    x = x.reshape((x.shape[0], 1, x.shape[1]))
    x = x.repeat(y.shape[1], 1)

    # return results
    return {
        'title': title,
        'xdata': x,
        'ydata': y,
        'data': data,
        'xlabel': 'time',
        'ylabel': ylabel,
        'colorlabel': colorlabel,
        'invert_yaxis': invert_yaxis,
    }


def Heatmap(arc1, arc2, arc3=None, colormap='viridis', colorrange=None, **kwargs):
    """
    Return a Heatmap data trace for the given input combination.
    This function can take a HARP product together with a string or three HARP variables
    representing a data, time and altitude array, respectively, can manually be inputted.

    Compatible plot type: `Plot`

    Arguments:
    arc1 - HARP product or a HARP variable representing a data array
    arc2 - String reference a variable in the HARP product or or a HARP variable representing a time array
    arc3 - A HARP variable representing an altitude array
    colormap -- Colormap name (matplotlib) or list of (r,g,b), (r,g,b,a)
                or (x,r,g,b,a) values (ranging from 0..1)
    colorrange -- Color range to use (default min, max of data)
    gap_threshold -- Add gaps when larger (np.timedelta64)

    """
    data = heatmap_data(arc1, arc2, arc3)
    return vis.Heatmap(
        colormap=colormap,
        colorrange=colorrange,
        **data,
        **kwargs
    )


def Curtain(arc1, arc2, arc3=None, arc4=None, colormap='viridis', colorrange=None, **kwargs):
    """
    Return a Curtain data trace for the given Harp variable.

    Compatible plot type: `Plot`

    Arguments:
    arc1 - HARP product or a HARP Latitude variable
    arc2 - String reference a variable in the HARP product or a HARP Longitude variable
    arc3 - Optional: A HARP data variable, only applicable when using a HARP Latitude and Longitude variable
    arc4 -
    colormap -- Colormap name (matplotlib) or list of (r,g,b), (r,g,b,a)
                or (x,r,g,b,a) values (ranging from 0..1)
    colorrange -- Color range to use (default min, max of data)

    """
    data = curtain_data(arc1, arc2, arc3, arc4)

    return vis.Curtain(
        colormap=colormap,
        colorrange=colorrange,
        **data,
        **kwargs
    )


def geo_data(arc1, arc2, arc3, **kwargs):
    return _mapplot_data(arc1, arc2, arc3)


def Geo(arc1, arc2, arc3=None, colormap='viridis', colorrange=None, opacity=0.6,
        pointsize=2, showcolorbar=True, **kwargs):
    """
    Return a Geo data trace for the given input.

    This function can take a HARP product together with a valid string or can use three
    HARP variables as input in the following order:

    Latitude variable
    Longitude variable
    Data variable

    Compatible plot type: `MapPlot`

    Arguments:
    arc1 - HARP product or a HARP Latitude variable
    arc2 - String reference a variable in the HARP product or a HARP Longitude variable
    arc3 - Optional: A HARP data variable, only applicable when using a HARP Latitude and Longitude variable
    colormap -- Colormap name (matplotlib) or list of (r,g,b), (r,g,b,a)
                or (x,r,g,b,a) values (ranging from 0..1)
    colorrange -- Color range to use (default min, max of data)
    opacity -- Opacity (default 0.6)
    pointsize -- Point size (for point data)

    """
    data = geo_data(arc1, arc2, arc3)

    return vis.Geo(
        colormap=colormap,
        colorrange=colorrange,
        opacity=opacity,
        pointsize=pointsize,
        showcolorbar=showcolorbar,
        **data,
        **kwargs
    )


def geo3d_data(arc1, arc2, arc3, **kwargs):
    return _mapplot_data(arc1, arc2, arc3)


def Geo3D(arc1, arc2, arc3=None, colormap='viridis', colorrange=None, heightfactor=None,
          opacity=0.6, pointsize=None, showcolorbar=True, **kwargs):
    """
    Return a Geo3D data trace for the given Harp variable.

    Compatible plot type: `MapPlot3D`

    Arguments:
    product -- Harp product
    value -- Harp variable name
    colormap -- Colormap name (matplotlib) or list of (r,g,b), (r,g,b,a)
                or (x,r,g,b,a) values (ranging from 0..1)
    colorrange -- Color range to use (default min, max of data)
    heightfactor -- Scale height
    opacity -- Opacity (default 0.6)
    pointsize -- Point size
    showcolorbar -- Show colorbar (default True)

    """
    data = geo3d_data(arc1, arc2, arc3)

    return vis.Geo3D(
        colormap=colormap,
        colorrange=colorrange,
        heightfactor=heightfactor,
        opacity=opacity,
        pointsize=pointsize,
        showcolorbar=showcolorbar,
        **data,
        **kwargs
    )
