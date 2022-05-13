<a name="avl"></a>
# avl

Atmosphere Virtual Lab

A toolkit for interactive plotting of atmospheric data.

Given a Harp product and variable name, it extracts data as well as meta-data
to automatically produce a data trace.

The following types of data traces are currently supported:

- Scatter
- Histogram
- Heatmap
- Geo
- Geo3D
- Volume

There are three of plots that can also be individually instantiated, then
populated with compatible data traces:

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

<a name="avl.download"></a>
#### download

```python
download(files, target_directory=".")
```

Download file(s) from `atmospherevirtuallab.org, skipping files
that already exist.

**Arguments**:

- `files` - file name or list/tuple of file names
- `target_directory` - path where to store files (default '.')

<a name="avl.Volume"></a>
#### Volume

```python
Volume(product, value, **kwargs)
```

Return a Volume data trace for the given Harp variable.

Volume data traces cannot currently be combined in a single plot.

**Arguments**:

- `product` - Harp product
- `value` - Harp variable name
- `spherical` - Project data to a sphere (default False)

<a name="avl.Histogram"></a>
#### Histogram

```python
Histogram(product, value, **kwargs)
```

Return a Histogram data trace for the given Harp variable.

Compatible plot type: `Plot`

**Arguments**:

- `product` - Harp product
- `value` - Harp variable name
- `bins` - Number of bins

<a name="avl.Scatter"></a>
#### Scatter

```python
Scatter(product, value, **kwargs)
```

Return a Scatter data trace for the given Harp variable.

Compatible plot type: `Plot`

**Arguments**:

- `product` - Harp product
- `value` - Harp variable name

<a name="avl.Heatmap"></a>
#### Heatmap

```python
Heatmap(product, value, **kwargs)
```

Return a Heatmap data trace for the given Harp variable.

Compatible plot type: `Plot`

**Arguments**:

- `product` - Harp product
- `value` - Harp variable name
- `gap_threshold` - Add gaps when larger (np.timedelta, default 24h)

<a name="avl.Geo"></a>
#### Geo

```python
Geo(product, value, **kwargs)
```

Return a Geo data trace for the given Harp variable.

Compatible plot type: `MapPlot`

**Arguments**:

- `product` - Harp product
- `value` - Harp variable name

<a name="avl.Geo3D"></a>
#### Geo3D

```python
Geo3D(product, value, **kwargs)
```

Return a Geo3D data trace for the given Harp variable.

Compatible plot type: `MapPlot3D`

**Arguments**:

- `product` - Harp product
- `value` - Harp variable name

<a name="avl.vis"></a>
# avl.vis

<a name="avl.vis.Plot"></a>
## Plot Objects

```python
class Plot()
```

2D Plot type

<a name="avl.vis.Plot.add"></a>
#### add

```python
add(obj)
```

Add data trace of the same plot type.

**Arguments**:

- `obj` - Data trace

<a name="avl.vis.MapPlot"></a>
## MapPlot Objects

```python
class MapPlot()
```

2D Map Plot type

<a name="avl.vis.MapPlot.__init__"></a>
#### \_\_init\_\_

```python
__init__(**kwargs)
```

**Arguments**:

- `colorrange` - Color range to use (default min, max of data)

<a name="avl.vis.MapPlot.add"></a>
#### add

```python
add(obj)
```

Add data trace of the same plot type.

**Arguments**:

- `obj` - Data trace

<a name="avl.vis.MapPlot3D"></a>
## MapPlot3D Objects

```python
class MapPlot3D()
```

3D Map Plot type

<a name="avl.vis.MapPlot3D.__init__"></a>
#### \_\_init\_\_

```python
__init__(showcolorbar=True, colorrange=None, size=(640, 480), centerlon=0, centerlat=0, opacity=0.6, pointsize=None, heightfactor=None, zoom=None, **kwargs)
```

**Arguments**:

- `showcolorbar` - Show colorbar (default True)
- `colorrange` - Color range to use (default min, max of data)
- `size` - Plot size in pixels (default (640, 480))
- `centerlon` - Center longitude (default 0)
- `centerlat` - Center latitude (default 0)
- `opacity` - Opacity (default 0.6)
- `pointsize` - Point size
- `heightfactor` - Scale height
- `zoom` - Zoom factor

<a name="avl.vis.MapPlot3D.add"></a>
#### add

```python
add(obj)
```

Add data trace of the same plot type.

**Arguments**:

- `obj` - Data trace
