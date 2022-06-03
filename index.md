<a name="avl"></a>
# avl

Atmosphere Virtual Lab

A toolkit for interactive plotting of atmospheric data.

Given a Harp product and variable name, it extracts data as well as meta-data
to automatically produce an annotated data trace.

The following types of data traces are currently supported:

- Scatter
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

<a id="avl.download"></a>

#### download

```python
download(files, target_directory=".")
```

Download file(s) from `atmospherevirtuallab.org`, skipping files
that already exist.

**Arguments**:

- `files` - file name or list/tuple of file names
- `target_directory` - path where to store files (default '.')

<a id="avl.Volume"></a>

#### Volume

```python
Volume(product, value, **kwargs)
```

Return a Volume data trace for the given Harp variable.

Nan-values are converted to the lowest non-nan value.

Volume data traces cannot currently be combined in a single plot!

**Arguments**:

- `product` - Harp product
- `value` - Harp variable name
- `spherical` - Project data to a sphere (default False)

<a id="avl.Histogram"></a>

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

<a id="avl.Scatter"></a>

#### Scatter

```python
Scatter(product, value, **kwargs)
```

Return a Scatter data trace for the given Harp variable.

Compatible plot type: `Plot`

**Arguments**:

- `product` - Harp product
- `value` - Harp variable name

<a id="avl.Heatmap"></a>

#### Heatmap

```python
Heatmap(product, value, **kwargs)
```

Return a Heatmap data trace for the given Harp variable.

Compatible plot type: `Plot`

**Arguments**:

- `product` - Harp product
- `value` - Harp variable name
- `colormap` - Colormap name (matplotlib) or list of (x,r,g,b,a) values (0..1)
- `gap_threshold` - Add gaps when larger (np.timedelta, default 24h)

<a id="avl.Geo"></a>

#### Geo

```python
Geo(product, value, **kwargs)
```

Return a Geo data trace for the given Harp variable.

Compatible plot type: `MapPlot`

**Arguments**:

- `product` - Harp product
- `value` - Harp variable name

<a id="avl.Geo3D"></a>

#### Geo3D

```python
Geo3D(product, value, **kwargs)
```

Return a Geo3D data trace for the given Harp variable.

Compatible plot type: `MapPlot3D`

**Arguments**:

- `product` - Harp product
- `value` - Harp variable name

<a id="avl.vis"></a>

# avl.vis

<a id="avl.vis.Plot"></a>

## Plot Objects

```python
class Plot()
```

2D Plot type

<a id="avl.vis.Plot.add"></a>

#### add

```python
add(obj)
```

Add data trace of the same plot type.

**Arguments**:

- `obj` - Data trace

<a id="avl.vis.MapPlot"></a>

## MapPlot Objects

```python
class MapPlot()
```

2D Map Plot type

<a id="avl.vis.MapPlot.__init__"></a>

#### \_\_init\_\_

```python
__init__(data_type, centerlat=0.0, centerlon=0.0, colorrange=None, opacity=1, pointsize=2, zoom=1, size=(800, 400), colormap=None, **kwargs)
```

**Arguments**:

- `centerlon` - Center longitude (default 0)
- `centerlat` - Center latitude (default 0)
- `colormap` - Colormap name (matplotlib) or list of (x,r,g,b,a) values (0..1)
- `colorrange` - Color range to use (default min, max of data)
- `opacity` - Opacity (default 0.6)
- `pointsize` - Point size
- `size` - Plot size in pixels (default (640, 480))
- `zoom` - Zoom factor

<a id="avl.vis.MapPlot.add"></a>

#### add

```python
add(obj)
```

Add data trace of the same plot type.

**Arguments**:

- `obj` - Data trace

<a id="avl.vis.MapPlot3D"></a>

## MapPlot3D Objects

```python
class MapPlot3D()
```

3D Map Plot type

<a id="avl.vis.MapPlot3D.__init__"></a>

#### \_\_init\_\_

```python
__init__(showcolorbar=True, colorrange=None, size=(640, 480), centerlon=0, centerlat=0, opacity=0.6, pointsize=None, heightfactor=None, zoom=None, colormap=None, **kwargs)
```

**Arguments**:

- `centerlon` - Center longitude (default 0)
- `centerlat` - Center latitude (default 0)
- `colormap` - Colormap name (matplotlib) or list of (x,r,g,b,a) values (0..1)
- `colorrange` - Color range to use (default min, max of data)
- `heightfactor` - Scale height
- `opacity` - Opacity (default 0.6)
- `pointsize` - Point size
- `showcolorbar` - Show colorbar (default True)
- `size` - Plot size in pixels (default (640, 480))
- `zoom` - Zoom factor

<a id="avl.vis.MapPlot3D.add"></a>

#### add

```python
add(obj)
```

Add data trace of the same plot type.

**Arguments**:

- `obj` - Data trace

