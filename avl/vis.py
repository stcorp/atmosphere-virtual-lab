import datetime
from importlib_resources import files, as_file
import os

from IPython.display import display
from ipywidgets import Layout
import ipyleaflet
import matplotlib  # TODO access colormaps differently? use separate project?
import numpy as np
import panel as pn
import plotly.graph_objects as go
import pyproj
import vtk
import io
from vtk.numpy_interface.dataset_adapter import numpyTovtkDataArray
import vtk.util.numpy_support  # TODO use newer numpy_interface?

try:
    from ipyleaflet_gl_vector_layer import IpyleafletGlVectorLayer, IpyleafletGlVectorLayerWrapper
except ImportError:
    pass

import avl

# register extensions to be loaded when calling pn.extension
# TODO under discussion with panel developer about correct approach for panel wrapper lib
import panel.models.plotly
import panel.models.vtk

_kPointData = 0
_kSwathData = 1
_kGridData = 2


def _data_type(latitude, longitude, data):  # TODO check that shapes match
    if data is not None and data.ndim == 2:
        if (latitude.ndim, longitude.ndim) == (1, 1):
            data_type = _kGridData

    elif (len(latitude.shape) == 2 and latitude.shape[-1] == 4 and
          len(longitude.shape) == 2 and longitude.shape[-1] == 4):
        if data is not None and data.ndim == 1:
            data_type = _kSwathData

    elif latitude.ndim == 1 and longitude.ndim == 1:
        if data is not None and data.ndim == 1:
            data_type = _kPointData

    if data_type is None:
        raise Exception('invalid lat/lon/data dimensions')

    return data_type


# TODO add BasePlot, merge add methods?

class Plot:  # TODO
    """2D Plot type
    """

    def __init__(self, layout=None, **kwargs):  # TODO name individual kwargs
        self._fig = go.Figure()
        self._traces = []
        self._layout = layout
        if layout is not None:
            self._fig.update_layout(layout)

    def add(self, obj):
        """Add data trace of the same plot type.

        Arguments:
        obj -- Data trace
        """
        if isinstance(obj, Plot):
            traces = obj._traces
            if self._layout is None:
                self._layout = obj._layout
                self._fig.update_layout(obj._layout)
        elif isinstance(obj, Trace):
            traces = [obj]
        else:
            assert False

        for trace in traces:
            if trace.type_ == 'scatter':
                self._fig.add_trace(go.Scatter(**trace.kwargs))
            elif trace.type_ == 'heatmap':
                self._fig.add_trace(go.Heatmap(**trace.kwargs))
            elif trace.type_ == 'histogram':
                self._fig.add_trace(go.Histogram(**trace.kwargs))
            self._traces.append(trace)

    def _ipython_display_(self):
        pn.extension(sizing_mode='stretch_width')
        display(pn.pane.Plotly(self._fig))


class MapPlot:
    """2D Map Plot type
    """

    def __init__(self, data_type, centerlat=0.0, centerlon=0.0, colorrange=None, opacity=0.6,
                 pointsize=2, zoom=1, size=(800, 400), colormap=None, **kwargs):
        """
        Arguments:
        centerlon -- Center longitude (default 0)
        centerlat -- Center latitude (default 0)
        colormap -- Colormap name (matplotlib) or list of (x,r,g,b,a) values (0..1)
        colorrange -- Color range to use (default min, max of data)
        opacity -- Opacity (default 0.6)
        pointsize -- Point size
        size -- Plot size in pixels (default (640, 480))
        zoom -- Zoom factor

        """
        layout = Layout(width=str(size[0]) + 'px', height=str(size[1]) + 'px')
        self._map = ipyleaflet.Map(center=[centerlat, centerlon], zoom=zoom,
                                   scroll_wheel_zoom=True, layout=layout)
        self._traces = []
        self._pointsize = pointsize
        self._opacity = opacity
        self._data_type = data_type  # TODO use _data_type instead!
        self._colorrange = colorrange
        self._colormap = colormap

    def getMap(self):
        return self._map

    def add(self, obj):
        """Add data trace of the same plot type.

        Arguments:
        obj -- Data trace
        """
        if isinstance(obj, MapPlot):
            traces = obj._traces
        elif isinstance(obj, Trace):
            traces = [obj]
        else:
            assert False

        for trace in traces:
            kwargs = trace.kwargs
            opacity = kwargs.get('opacity')

            plot_types = dict([(0, 'points'), (1, 'swath'), (2, 'grid')])
            plot_type = None
            if(self._data_type is not None):
                plot_type = plot_types[self._data_type]

            featureGlWrapper = IpyleafletGlVectorLayerWrapper()
            args = {
                "lat": kwargs['latitude'],
                "lon": kwargs['longitude'],
                "data": kwargs['data'],
                "colorrange": list(self._colorrange) if self._colorrange else None,
                "plot_type": plot_type,
                "pointsize": self._pointsize,
                "opacity": opacity if opacity is not None else self._opacity,
                "colormap": self._colormap
            }
            featureGlLayer = IpyleafletGlVectorLayer(**args)
            self._map.add_layer(featureGlWrapper)
            featureGlWrapper.add_layer(featureGlLayer)

            self._traces.append(trace)

    def _ipython_display_(self):
        pn.extension(sizing_mode='stretch_width')
        display(pn.pane.ipywidget.IPyLeaflet(self._map))


class MapPlot3D:
    """3D Map Plot type
    """

    # TODO pointsize, colormap, colorrange.. settable per-trace
    def __init__(self, showcolorbar=True, colorrange=None, size=(640, 480),
                 centerlon=0, centerlat=0, opacity=0.6, pointsize=None,
                 heightfactor=None, zoom=None, colormap=None, **kwargs):
        """
        Arguments:
        centerlon -- Center longitude (default 0)
        centerlat -- Center latitude (default 0)
        colormap -- Colormap name (matplotlib) or list of (x,r,g,b,a) values (0..1)
        colorrange -- Color range to use (default min, max of data)
        heightfactor -- Scale height
        opacity -- Opacity (default 0.6)
        pointsize -- Point size
        showcolorbar -- Show colorbar (default True)
        size -- Plot size in pixels (default (640, 480))
        zoom -- Zoom factor

        """
        self.showcolorbar = showcolorbar
        self.colorrange = colorrange
        self.size = size
        self.centerlon = centerlon
        self.centerlat = centerlat
        self.colormap = colormap
        self.opacity = opacity
        self.heightfactor = heightfactor
        self.pointsize = pointsize
        self.zoom = zoom

        # 3d projection
        self.p1 = pyproj.Proj(proj='longlat', a=6.1e9, b=6.1e9)
        self.p2 = pyproj.Proj(proj='geocent', a=6.1e9, b=6.1e9)

        self._traces = []

        self._renderwindow = None
        self._renderer = None

    def add(self, obj):
        """Add data trace of the same plot type.

        Arguments:
        obj -- Data trace
        """
        if isinstance(obj, MapPlot3D):
            traces = obj._traces
        elif isinstance(obj, Trace):
            traces = [obj]
        else:
            assert False

        self.setup_scene()

        for trace in traces:
            kwargs = trace.kwargs

            lut = self.lookup_table(kwargs['data'])

            # TODO just pass kwargs
            data_actor = self.data_actor(kwargs['latitude'],
                                         kwargs['longitude'], kwargs['data'],
                                         lut, kwargs.get('heightfactor'),
                                         kwargs.get('opacity'))

            colorbar_actor = self.colorbar_actor(lut)

            self._renderer.AddActor(data_actor)

            if self.showcolorbar:
                self._renderer.AddActor(colorbar_actor)

            self._traces.append(trace)

    def data_actor(self, latitude, longitude, data, lut, heightfactor, opacity):
        data_type = _data_type(latitude, longitude, data)

        # swath data: polygons for lat/lon boundaries
        if data_type == _kSwathData:
            # data points
            arrz = np.array([0] * 4 * len(data))
            fx, fy, fz = pyproj.transform(self.p1, self.p2, longitude.flat, latitude.flat, arrz)

            arr = np.column_stack([fx, fy, fz])
            data_points = numpyTovtkDataArray(arr)

            # cells
            cells = np.zeros((len(data), 5), dtype=np.int64)
            data_idx = np.arange(len(data)) * 4

            cells[:, 0] = 4
            cells[:, 1] = data_idx
            cells[:, 2] = data_idx + 1
            cells[:, 3] = data_idx + 2
            cells[:, 4] = data_idx + 3

        # grid data: polygons across grid
        elif data_type == _kGridData:
            # lat, lon mesh provided
            if latitude.shape[0] == data.shape[0] + 1 and longitude.shape[0] == data.shape[1] + 1:
                latitude_plus = latitude
                longitude_plus = longitude

            # otherwise, interpolate mesh points
            else:
                # determine midpoints between data points
                lon1 = np.append(longitude[0] - (longitude[1] - longitude[0]), longitude)
                lon2 = np.append(longitude, longitude[-1] + (longitude[-1] - longitude[-2]))
                longitude_plus = (lon1 + lon2) / 2

                lat1 = np.append(latitude[0] - (latitude[1] - latitude[0]), latitude)
                lat2 = np.append(latitude, latitude[-1] + (latitude[-1] - latitude[-2]))
                latitude_plus = (lat1 + lat2) / 2

                # crossing latitude boundaries (<90 or >90)
                if latitude_plus[0] < -90:
                    latitude_plus[0] = -180 - latitude_plus[0]
                if latitude_plus[-1] > 90:
                    latitude_plus[-1] = 180 - latitude_plus[-1]

            # create mesh grid from midpoints
            longrid, latgrid = np.meshgrid(longitude_plus, latitude_plus)

            if heightfactor is not None:
                # TODO correctness, points, swaths, match visan
                arrz = heightfactor * 2e6 * np.pad(data, ((0, 1), (0, 1)), 'edge')
            else:
                arrz = np.zeros(latgrid.shape)

            # transform to 3d vtk points
            fx, fy, fz = pyproj.transform(self.p1, self.p2, longrid.flat, latgrid.flat, arrz.flat)
            arr = np.column_stack([fx, fy, fz])
            data_points = numpyTovtkDataArray(arr)

            # create vtk cells
            cells = np.zeros((data.size, 5), dtype=np.int64)

            longrid, latgrid = np.meshgrid(range(data.shape[1]), range(data.shape[0]))
            flatlon = longrid.flatten()
            flatlat = latgrid.flatten()

            nlon = len(longitude_plus)
            f1 = flatlat * nlon + flatlon
            f2 = (flatlat + 1) * nlon + flatlon

            cells[:, 0] = 4
            cells[:, 1] = f1
            cells[:, 2] = f1 + 1
            cells[:, 3] = f2 + 1
            cells[:, 4] = f2

        # point data
        elif data_type == _kPointData:
            # data points
            arrz = np.zeros(len(longitude))
            fx, fy, fz = pyproj.transform(self.p1, self.p2, longitude.flat, latitude.flat, arrz)

            arr = np.column_stack([fx, fy, fz])
            data_points = numpyTovtkDataArray(arr)

            # cells
            cells = np.zeros((arr.shape[0], 2), dtype=np.int64)
            cells[:, 0] = 1
            cells[:, 1] = np.arange(arr.shape[0])

        else:
            raise Exception('unsupported datatype')

        # polydata
        polydata = vtk.vtkPolyData()
        points = vtk.vtkPoints()
        points.SetData(data_points)
        polydata.SetPoints(points)

        if data_type != _kPointData:
            polygons = vtk.vtkCellArray()
            polygons.SetCells(cells.shape[0], vtk.util.numpy_support.numpy_to_vtkIdTypeArray(cells, deep=True))
            polydata.SetPolys(polygons)

        else:
            vertices = vtk.vtkCellArray()
            vertices.SetCells(cells.shape[0], vtk.util.numpy_support.numpy_to_vtkIdTypeArray(cells, deep=True))
            polydata.SetVerts(vertices)

        scalars = numpyTovtkDataArray(data.flat)
        polydata.GetCellData().SetScalars(scalars)

        mappert = vtk.vtkPolyDataMapper()
        mappert.SetInputData(polydata)

        mappert.SetScalarModeToUseCellData()
        mappert.UseLookupTableScalarRangeOn()
        mappert.SetLookupTable(lut)

        actor = vtk.vtkActor()
        actor.GetProperty().SetOpacity(opacity if opacity is not None else self.opacity)
        if self.pointsize is not None:
            actor.GetProperty().SetPointSize(self.pointsize)
        actor.SetMapper(mappert)

        return actor

    def lookup_table(self, data):
        lut = vtk.vtkLookupTable()
        lut.SetNumberOfTableValues(256)  # TODO configurable

        # colormap
        colormap = self.colormap

        if isinstance(colormap, list):  # TODO interpolation mode as in visan? (sqrt, scurve).. use vtk SetRampTo*?
            # TODO check again, as vtk should have all this builtin..
            for i in range(256):
                x = i * (1. / 255)
                for idx in range(len(colormap)):
                    x1, r1, g1, b1, a1 = colormap[idx]
                    x2, r2, g2, b2, a2 = colormap[idx + 1]
                    if x1 <= x <= x2:
                        weight = (x - x1) / (x2 - x1)

                        r = (1 - weight) * r1 + weight * r2
                        g = (1 - weight) * g1 + weight * g2
                        b = (1 - weight) * b1 + weight * b2
                        a = (1 - weight) * a1 + weight * a2

                        lut.SetTableValue(i, r, g, b, a)
                        break
        else:
            colormap = colormap or 'viridis'
            cmap = matplotlib.cm.get_cmap(colormap)
            for i in range(256):
                lut.SetTableValue(i, *cmap.colors[i])

        # NaN color
        lut.SetNanColor(0.0, 0.0, 0.0, 0.0)

        # data range
        if self.colorrange is not None:
            lut.SetTableRange(self.colorrange[0], self.colorrange[1])
        else:
            lut.SetTableRange(np.nanmin(data), np.nanmax(data))

        lut.Build()
        return lut

    def colorbar_actor(self, lut):
        actor = vtk.vtkScalarBarActor()
        actor.SetOrientationToHorizontal()
        actor.SetLookupTable(lut)

        return actor

    def sphere_actor(self):
        sphere = vtk.vtkTexturedSphereSource()
        sphere.SetRadius(6e9)  # TODO do we want to normalize to 1?
        sphere.SetPhiResolution(30)
        sphere.SetThetaResolution(60)

        jpgreader = vtk.vtkJPEGReader()
        source = files(avl).joinpath('8k_earth_daymap.jpg')
        with as_file(source) as jpgpath:
            jpgreader.SetFileName(jpgpath)
            texture = vtk.vtkTexture()
            texture.SetInputConnection(jpgreader.GetOutputPort())
            texture.Update()

        transformTexture = vtk.vtkTransformTextureCoords()
        transformTexture.SetInputConnection(sphere.GetOutputPort())
        transformTexture.SetPosition(0.5, 0, 0)

        sphere_mapper = vtk.vtkPolyDataMapper()
        sphere_mapper.SetInputConnection(transformTexture.GetOutputPort())

        actor = vtk.vtkActor()
        actor.SetMapper(sphere_mapper)
        actor.SetTexture(texture)

        return actor

    def setup_scene(self):
        if self._renderwindow is not None:
            return self._renderer

        # camera
        camera = vtk.vtkCamera()
        dist = 3e10 - (self.zoom or 0) * 2e9  # TODO incorrect/different
        cx, cy, cz = pyproj.transform(self.p1, self.p2, self.centerlon, self.centerlat, dist)
        camera.SetPosition(cx, cy, cz)
        camera.SetViewUp(0, 0, 1)
        camera.SetFocalPoint(0, 0, 0)

        # renderer
        self._renderer = vtk.vtkRenderer()
        self._renderer.SetActiveCamera(camera)

        # sphere
        sphere_actor = self.sphere_actor()
        self._renderer.AddActor(sphere_actor)

        # renderwindow
        self._renderwindow = vtk.vtkRenderWindow()
        self._renderwindow.AddRenderer(self._renderer)

    def _ipython_display_(self):
        pn.extension(sizing_mode='stretch_width')
        display(pn.pane.VTK(self._renderwindow, width=self.size[0], height=self.size[1]))


# TODO add trace? specify physical north..?
def VolumePlot(data=None, size=(640, 1000), scale=(1, 1, 1),
               display_slices=True, display_volume=True, **kwargs):
    pn.extension(sizing_mode='stretch_width')
    data = np.nan_to_num(data, np.nanmin(data))  # known issue in vtk.js that it doesn't handle nans
    plot = pn.pane.VTKVolume(data, width=size[0], height=size[1],
                             display_slices=display_slices,
                             display_volume=display_volume, spacing=scale, max_data_size=1000)
    plot = pn.Row(plot.controls(jslink=False), plot)
    return plot


class Trace:
    def __init__(self, type_, **kwargs):
        self.type_ = type_
        self.kwargs = kwargs


def Geo(latitude, longitude, data=None, **kwargs):
    latitude = np.asarray(latitude)
    longitude = np.asarray(longitude)
    if data is not None:
        data = np.asarray(data)

    mapplot = MapPlot(**kwargs)
    mapplot.add(Trace(
        'geo',
        latitude=latitude,
        longitude=longitude,
        data=data,
        **kwargs
    ))
    return mapplot


def Geo3D(latitude, longitude, data=None, **kwargs):
    latitude = np.asarray(latitude)
    longitude = np.asarray(longitude)
    if data is not None:
        data = np.asarray(data)

    mapplot3d = MapPlot3D(**kwargs)
    mapplot3d.add(Trace(
        'geo3d',
        latitude=latitude,
        longitude=longitude,
        data=data,
        **kwargs
    ))
    return mapplot3d


def Scatter(xdata=None, ydata=None, title=None, xlabel=None, ylabel=None,
            pointsize=None, xnumticks=None, ynumticks=None, xerror=None,
            yerror=None, lines=False, **kwargs):
    xdata = np.asarray(xdata)
    ydata = np.asarray(ydata)

    layout = {
        'title': title,
        'xaxis': {
            'title': xlabel,
            'nticks': xnumticks,
        },
        'yaxis': {
            'title': ylabel,
            'nticks': ynumticks,
        },
    }

    fig = Plot(layout)

    if yerror is not None:
        error_y = {'array': yerror}
        mode = 'markers'
    else:
        error_y = None
        if lines:
            mode = None
        else:
            mode = 'markers'

    fig.add(Trace(
        'scatter',
        x=xdata,
        y=ydata,
        error_y=error_y,
        mode=mode,
        showlegend=False,
    ))

    if xerror is not None:  # TODO y
        fig.add(Trace(
            'scatter',
            x=xdata - xerror,
            y=ydata,
            line={'width': 0},
            showlegend=False,
        ))
        fig.add(Trace(
            'scatter',
            x=xdata + xerror,
            y=ydata,
            fill='tonexty',
            line={'width': 0},
            fillcolor='rgba(68, 68, 68, 0.3)',
            showlegend=False,
        ))

    return fig


def Line(*args, **kwargs):
    return Scatter(*args, lines=True, **kwargs)


def Histogram(data, bins=None, **kwargs):
    data = np.asarray(data)

    layout = {  # TODO dedupe
        'title_text': kwargs['title'],
        'xaxis_title_text': kwargs['ylabel'],  # TODO
        'barmode': 'overlay',
    }

    fig = Plot(layout)

    fig.add(Trace(
        'histogram',
        x=data,
        nbinsx=bins * 2 if bins else None,  # TODO why *2 needed
        name=kwargs['title'],
        opacity=0.6
    ))

    return fig


# TODO separate xcoords, ycoords
def Heatmap(data=None, coords=None, xlabel=None, ylabel=None, title=None,
            colorlabel=None, gap_threshold=None, colormap=None, **kwargs):
    xcoords, ycoords = coords
    xcoords = np.asarray(xcoords)
    ycoords = np.asarray(ycoords)
    data = np.asarray(data)

    layout = {
        'title': title,
        'xaxis': {
            'title': xlabel,
        },
        'yaxis': {
            'title': ylabel,
        },
    }

    fig = Plot(layout)

    # insert gaps
    if gap_threshold is None:
        gap_threshold = np.timedelta64(24, 'h')
    halfgap = gap_threshold / 2

    xdata_new = []
    xcoords_new = []
    for i in range(len(data)):
        xdata_new.append(data[i])

        if i == 0:
            xcoords_new.append(xcoords[i] - halfgap)
        if i < len(data) - 1:
            if xcoords[i + 1] - xcoords[i] > gap_threshold:
                xcoords_new.append(xcoords[i] + halfgap)
                xcoords_new.append(xcoords[i + 1] - halfgap)

                xdata_new.append(np.array([np.NaN] * len(ycoords)))
            else:
                xcoords_new.append(xcoords[i] + ((xcoords[i + 1] - xcoords[i]) / 2))
        else:
            xcoords_new.append(xcoords[i] + halfgap)

    data = np.array(xdata_new)
    xcoords = xcoords_new

    if isinstance(colormap, list):
        colorscale = [[x, 'rgb' + str((255 * r, 255 * g, 255 * b, a))] for (x, r, g, b, a) in colormap]
    else:
        cmap = matplotlib.cm.get_cmap(colormap or 'viridis')
        colorscale = [[i * 1. / 255, 'rgb' + str(tuple(cmap.colors[i]))] for i in range(256)]  # TODO configurable

    fig.add(Trace(
        'heatmap',
        z=np.transpose(data),
        x=xcoords,
        y=ycoords,
        colorscale=colorscale,
        colorbar={'title': colorlabel},
    ))

    return fig
