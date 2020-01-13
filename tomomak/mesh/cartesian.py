from . import abstract_axes
import numpy as np
import matplotlib.pyplot as plt
from tomomak.plots import plot1d, plot2d, plot3d
import warnings
import tomomak.util.text


class Axis1d(abstract_axes.Abstract1dAxis):
    """1D regular or irregular coordinate system.

    Main tomomak coordinate system.
    All transforms from or to other coordinate systems, plotting, etc.
    are performed using this coordinate system as mediator.
    If regular, than it is Cartesian coordinate system.
    """
    def __init__(self, coordinates=None, edges=None, lower_limit=0, upper_limit=None, size=None, name="", units=""):
        super().__init__(name, units)
        if coordinates is not None:
            if size is not None or upper_limit is not None:
                warnings.warn("Since coordinates are given explicitly, size and upper_limit arguments are ignored.")
            if edges is not None:
                warnings.warn("Since coordinates are given explicitly, edges are ignored.")
            self._create_using_coordinates(coordinates, lower_limit)
        elif edges is not None:
            if size is not None or upper_limit is not None:
                warnings.warn("Since coordinates are given explicitly, size and upper_limit arguments are ignored.")
            self._create_using_edges(edges)
        else:
            if size is None:
                warnings.warn("Axis1d init: size was not set. Default size = 10 is used.")
                size = 10
            if upper_limit is None:
                warnings.warn("Axis1d init: upper_limit  was not set. Default upper_limit = 10 is used.")
                upper_limit = 10
            self._create_using_limits(lower_limit, upper_limit, size)

    def _create_using_edges(self, edges):
        coordinates = np.zeros(len(edges) - 1)
        for i, _ in enumerate(coordinates):
            coordinates[i] = (edges[i] + edges[i + 1]) / 2
        self._create_using_coordinates(coordinates, edges[0])
        self._cell_edges[0], self._cell_edges[-1] = edges[0], edges[-1]

    def _create_using_limits(self, lower_limit, upper_limit, size):
        self._size = size
        dv = np.abs(upper_limit - lower_limit) / size
        self._volumes = np.full(size, dv)
        self._coordinates = np.fromfunction(lambda i: lower_limit + (i * dv) + (dv / 2), (size,))
        self._calc_cell_edges(lower_limit)
        self._cell_edges[-1] = upper_limit

    def _create_using_coordinates(self, coordinates, lower_limit):
        if (any(np.diff(coordinates)) < 0 and coordinates[-1] > lower_limit
                or any(np.diff(coordinates)) > 0 and coordinates[-1] < lower_limit):
            raise Exception("Coordinates are not monotonous.")
        if (coordinates[-1] > lower_limit > coordinates[0]
                or coordinates[-1] < lower_limit < coordinates[0]):
            raise Exception("lower_limit is inside of the first segment.")
        self._size = len(coordinates)
        self._coordinates = coordinates
        dv = np.diff(coordinates)
        self._volumes = np.zeros(self.size)
        dv0 = coordinates[0] - lower_limit
        self._volumes[0] = 2 * dv0
        for i in range(self.size - 1):
            self._volumes[i + 1] = 2 * dv[i] - self._volumes[i]
        for i, v in enumerate(self._volumes):
            if v <= 0:
                raise Exception("Point № {} of the coordinates is inside of the previous segment. "
                                "Increase the distance between the points.".format(i))
        self._calc_cell_edges(lower_limit)

    def _calc_cell_edges(self, lower_limit):
        self._cell_edges = np.zeros(self.size + 1)
        self._cell_edges[0] = lower_limit
        for i in range(self.size):
            self._cell_edges[i + 1] = self._volumes[i] + self._cell_edges[i]
        return self._cell_edges

    def __str__(self):
        if self.regular:
            ax_type = 'regular'
        else:
            ax_type = 'irregular'
        return "{}D {} axis with {} cells. Name: {}. Boundaries: {} {}. " \
            .format(self.dimension, ax_type, self.size, self.name,
                    [self._cell_edges[0], self._cell_edges[-1]], self.units)

    @property
    def volumes(self):
        """See description in abstract axes.
        """
        return self._volumes

    @property
    def coordinates(self):
        """See description in abstract axes.
        """
        return self._coordinates

    @property
    def cell_edges1d(self):
        """See description in abstract axes.
        """
        return self._cell_edges

    @property
    def cell_edges(self):
        """See description in abstract axes.
        """
        return self._cell_edges

    @property
    def size(self):
        """See description in abstract axes.
        """
        return self._size

    @property
    def regular(self):
        """See description in abstract axes.
        """
        if all(self._volumes - self._volumes[0] == 0):
            return True
        else:
            return False

    def cell_edges2d(self, axis2):
        """See description in abstract axes.
        """
        if type(axis2) is not Axis1d:
            raise NotImplementedError("Cell edges with such combination of axes are not supported.")
        shape = (self.size, axis2.size)
        res = np.zeros(shape).tolist()
        edge1 = self.cell_edges1d
        edge2 = axis2.cell_edges1d
        for i, row in enumerate(res):
            for j, _ in enumerate(row):
                res[i][j] = [(edge1[i], edge2[j]), (edge1[i + 1], edge2[j]),
                             (edge1[i + 1], edge2[j + 1]), (edge1[i], edge2[j + 1])]
        return res

    def cell_edges3d(self, axis2, axis3):
        """See description in abstract axes.
        """
        if type(axis2) is not Axis1d or type(axis3) is not Axis1d:
            raise NotImplementedError("Cell edges with such combination of axes are not supported.")
        shape = (self.size, axis2.size, axis3.size)
        res = np.zeros(shape).tolist()
        edge1 = self.cell_edges1d
        edge2 = axis2.cell_edges1d
        edge3 = axis3.cell_edges1d
        for i, row in enumerate(res):
            for j, col in enumerate(row):
                for k, _ in enumerate(col):
                    res[i][j][k] = [(edge1[i], edge2[j], edge3[k]), (edge1[i + 1], edge2[j], edge3[k]),
                                     (edge1[i + 1], edge2[j + 1], edge3[k]), (edge1[i + 1], edge2[j], edge3[k + 1]),
                                     (edge1[i + 1], edge2[j + 1], edge3[k + 1]), (edge1[i], edge2[j + 1], edge3[k]),
                                     (edge1[i], edge2[j + 1], edge3[k + 1]), (edge1[i], edge2[j], edge3[k + 1])]
        return res

    def intersection(self, axis2):
        """See description in abstract axes.
        """
        if type(axis2) is not Axis1d:
            raise TypeError("Cell edges with such combination of axes are not supported.")

        intersection = np.zeros([self.size, axis2.size])

        def inters_len(a_min, a_max, b_min, b_max):
            res = min(a_max, b_max) - max(a_min, b_min)
            if res < 0:
                res = 0
            return res
        j_start = 0
        for i, row in enumerate(intersection):
            for j in range(j_start, len(row)):
                dist = inters_len(self.cell_edges1d[i], self.cell_edges1d[i + 1],
                                  axis2.cell_edges1d[j], axis2.cell_edges1d[j + 1])
                if not dist and j != j_start:
                    j_start = j-1
                    break
                intersection[i, j] = dist
        return intersection

    def plot1d(self, data, data_type='solution', filled=True,
               fill_scheme='viridis', edge_color='black', grid=False, equal_norm=False, *args, **kwargs):
        """Create 1D plot of the solution or detector geometry.

        matplotlib bar plot is used. Detector data is plotted on the interactive graph.

        Args:
            data(1D ndarray): data to plot.
            data_type(str, optional): type of the data: 'solution' or 'detector_geometry'. Default: solution.
            filled(bool_optional, optional): if true, bars are filled. Color depends on the bar height. Default: True.
            fill_scheme(str, optional): matplotlib fill scheme. Valid only if filled is True. Default: 'viridis'.
            edge_color(str, optional): color of the bar edges. See matplotlib colors for the avaiable options.
                Default: 'black'.
            grid(bool, optional): If true, grid is displayed. Default:False.
            equal_norm(bool, optional): If true, all detectors will have same norm.
                Valid only if data_type = detector_geometry. Default: False.
            *args,**kwargs: additional arguments to pass to matplotlib bar plot.

        Returns:
            matplotlib plot, matplotlib axis
        """
        if data_type == 'solution':
            y_label = r"Density, {}{}".format(self.units, '$^{-1}$')
            plot, ax = plot1d.bar1d(data, self, 'Density', y_label, filled, fill_scheme,
                                    edge_color, grid, *args, **kwargs)
        elif data_type == 'detector_geometry':
            title = "Detector 1/{}".format(data.shape[0])
            y_label = "Intersection length, {}".format(self.units)
            plot, ax, _ = plot1d.detector_bar1d(data, self, title, y_label, filled,
                                                fill_scheme, edge_color, grid, equal_norm, *args, **kwargs)
        else:
            raise AttributeError("data type {} is unknown".format(data_type))
        plt.show()
        return plot, ax

    def plot2d(self, data, axis2, data_type='solution',
               fill_scheme='viridis', grid=False, equal_norm=False, *args, **kwargs):
        """Create 2D plot of the solution or detector geometry.

        matplotlib pcolormesh is used. Detector data is plotted on the interactive graph.

        Args:
            data(1D ndarray): data to plot.
            axis2(tomomak axis): second axis. Only cartesian.Axis1d is supported.
            data_type(str, optional): type of the data: 'solution' or 'detector_geometry'. Default: solution.
            fill_scheme(str, optional): matplotlib fill scheme. Valid only if filled is True. Default: 'viridis'.
            grid(bool, optional): If true, grid is displayed. Default:False.
            equal_norm(bool, optional): If true, all detectors will have same norm.
                Valid only if data_type = detector_geometry. Default: False.
            *args,**kwargs: additional arguments to pass to matplotlib pcolormesh.

        Returns:
            matplotlib plot, matplotlib axis
        """
        if type(axis2) is not Axis1d:
            raise NotImplementedError("2D plots with such combination of axes are not supported.")
        if data_type == 'solution':
            units = tomomak.util.text.density_units([self.units, axis2.units])
            title = r"Density, {}".format(units)
            plot, ax, fig, cb = plot2d.colormesh2d(data, self, axis2, title, fill_scheme, grid, *args, **kwargs)
        elif data_type == 'detector_geometry':
            title = 'Detector 1/{}'.format(data.shape[0])
            plot, ax, _ = plot2d.detector_colormesh2d(data, self, axis2, title, fill_scheme, grid,
                                                      equal_norm, *args, **kwargs)
        else:
            raise AttributeError('data type {} is unknown'.format(data_type))
        plt.show()
        return plot, ax

    def plot3d(self, data, axis2, axis3, data_type='solution', *args, **kwargs):
        if type(axis2) is not Axis1d or type(axis3) is not Axis1d:
            raise NotImplementedError("3D plots with such combination of axes are not supported.")

        title = 'Detector 1/{}'.format(data.shape[0])

        plot3d.detector_colormesh3d(data, self, axis2, axis3, title,
                                                      *args, **kwargs)
        return 0