"""Construct and visualize the trajectory cooperative graph.

Auxiliary points are generated along and across the trajectory, triangulated,
and mapped back to trajectory-point indices after pruning.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
from pyproj import CRS, Transformer

def wgs842gk(data):
    """Project longitude/latitude columns using WGS84 three-degree zones.

    Args:
        data: Array with longitude and latitude in its first two columns,
            followed by any attributes to retain.
    Returns:
        Northing and easting coordinates followed by the remaining attributes.
        The graph-generation script expects already projected coordinates
        and does not call this helper.
    """

    lat, lon = data[:, 1], data[:, 0]
    crs_WGS84 = CRS.from_epsg(4326)
    d = int((lon[0] + 1.5) / 3)
    formatfy = '+proj=tmerc +lat_0=0 +lon_0=' + str(d * 3) + ' +k=1 +x_0=500000 +y_0=0 +ellps=WGS84 +units=m +no_defs'
    crs_GK = CRS.from_proj4(formatfy)
    # Respect the source CRS axis order when passing latitude and longitude.
    transformer = Transformer.from_crs(crs_WGS84, crs_GK)
    data = np.c_[np.fliplr(np.array(transformer.transform(lat, lon)).T), data[:, 2:]]
    return data

class Delaunaycal:
    """Build cooperative edges from projected trajectory rows.

    Args:
        data: Temporally ordered array whose first two columns are coordinates.
            Remaining columns are carried into the retained trajectory rows.
        width: Implement working width, in the same units as the coordinates.
        ratio: Along-track interpolation spacing relative to half the width.
        inter: The implemented interpolation mode is "on".

    ``calculatearea`` returns graph incidence counts, retained simplices,
    and redirected edges.
    """

    def __init__(self, data: np.ndarray, width=4, ratio=1, inter='on'):
        """Store graph-construction parameters without processing data."""
        self.split_points = None
        self.data = data
        self.width = width
        self.ratio = ratio
        self.area = ()
        self.inter = inter

    @property
    def adinter(self):
        """Return auxiliary points, their parent indices, and retained data rows.

        Each access recomputes the result, removes endpoint rows and rows
        at detected gaps, and updates ``split_points``.
        """
        if self.inter == 'on':
            dis, _, data = caldis(self.data)
            # Compute the floor-based number of samples for each segment.
            fl = np.floor(dis / (self.width / 2 * self.ratio)).astype(int)
            fl[fl <= 1] = 1
            flk = np.expand_dims(np.repeat(fl, fl, axis=0), 1)
            interk = np.zeros(len(flk))
            nfl = fl[np.nonzero(fl)]
            formers = 0
            for i in range(0, len(nfl)):
                interk[formers: formers + nfl[i]] = np.arange(1, nfl[i] + 1)
                formers = formers + nfl[i]
            interk = np.expand_dims(interk, axis=1)
            # Interpolate along each segment with the existing endpoint convention.
            aloneinter = (np.c_[np.repeat(data[:, 0], np.r_[fl, 0], axis=0), np.repeat(data[:, 1], np.r_[fl, 0],
                                                                                       axis=0)] * (
                                  flk - interk) +
                          np.c_[np.repeat(data[:, 0], np.r_[0, fl], axis=0), np.repeat(data[:, 1], np.r_[0, fl],
                                                                                       axis=0)] * interk) / flk
            interdis, interdif, aloneinter = caldis(aloneinter)
            dis, interdis = np.expand_dims(dis, 1), np.expand_dims(interdis, 1)
            # Offset the interpolated centerline by half the implement width on each side.
            widthinter = np.r_[
                aloneinter[0:-1, :] - np.c_[interdif[:, 1], -interdif[:, 0]] * (self.width / 2) / interdis,
                aloneinter[0:-1, :] + np.c_[interdif[:, 1], -interdif[:, 0]] * (self.width / 2) / interdis]
            aloneinter = np.delete(aloneinter, -1, 0)
            allpoints = np.r_[aloneinter, widthinter]
            # Assign auxiliary points to trajectory indices.
            attribution = interk
            attribution[attribution == 1] = 1e7
            # The large sentinel distinguishes the first sample of each segment.
            attribution[attribution != 1e7] -= (flk[attribution != 1e7] / 2 + 1)
            att_back_up = attribution.copy()
            data = np.delete(data, -1, 0)
            calculator = 0
            attribution[0] = 0
            for i in range(1, len(attribution)):
                if attribution[i] == 0 or (attribution[i] * att_back_up[i - 1] == 1e14) or (
                        attribution[i] > 0 and att_back_up[i - 1] < 0):
                    calculator += 1
                attribution[i] = calculator
            attribution = np.delete(attribution, -1, 0)
            attribution = np.r_[attribution, attribution, attribution]
            dis = np.delete(dis, -1, 0)
            # Locate gaps using the fixed threshold and remove boundary points.
            split_points = np.where((dis > 500) == 1)[0] + 1
            split_points = np.r_[0, split_points]
            data = np.delete(data, split_points, 0)
            for s_p_former in split_points:
                overlong_inter_index = np.squeeze(attribution == (s_p_former - 1))
                allpoints = np.delete(allpoints, overlong_inter_index, 0)
                attribution = np.delete(attribution, overlong_inter_index, 0)
            for s_p in split_points:
                overlong_inter_index = np.squeeze(attribution == s_p)
                allpoints = np.delete(allpoints, overlong_inter_index, 0)
                attribution = np.delete(attribution, overlong_inter_index, 0)
            for s_p_latter in split_points:
                overlong_inter_index = np.squeeze(attribution == (s_p_latter + 1))
                allpoints = np.delete(allpoints, overlong_inter_index, 0)
                attribution = np.delete(attribution, overlong_inter_index, 0)
            self.split_points = split_points

            return allpoints, attribution, data

    def calculatearea(self):
        """Triangulate auxiliary points and compute the retained cooperative edges.

        Returns:
            A tuple containing ``(node_ids, incidence_counts)``, retained
            triangle indices, and an integer edge array with two columns.
            The same tuple is stored in ``self.area`` for plotting.
        """

        tri = Delaunay(self.adinter[0])
        dif = tri.points[tri.simplices] - np.roll(tri.points[tri.simplices], 1, axis=1)
        dis = np.sqrt(np.sum(dif * dif, axis=2))
        # Remove entire triangles if any edge exceeds the existing width-based threshold.
        extract = np.nonzero(dis > self.width / 2 / np.cos(np.arctan(self.ratio)) * 2 * np.sqrt(2))
        extract = np.delete(tri.simplices, np.unique(extract[0]), 0)
        # Map triangle vertices to parent trajectory indices and remove self-edges.
        attr_tri = np.squeeze(self.adinter[1][extract])
        attr_tri = np.stack([np.roll(attr_tri, 1, axis=1), attr_tri], axis=0)
        attr_tri = np.moveaxis(attr_tri, -1, 0)
        attr_tri = np.moveaxis(attr_tri, -1, 1)
        attr_tri = np.reshape(attr_tri, (-1, 2))
        attr_tri = np.delete(attr_tri, attr_tri[:, 0] == attr_tri[:, 1], axis=0)

        # Add temporal connections while retaining the existing treatment of gaps.
        attr_tri_series_edge = np.arange(np.max(attr_tri) + 1)
        if self.split_points is not None:
            attr_tri_series_edge = np.delete(attr_tri_series_edge, self.split_points[self.split_points <= max(attr_tri_series_edge)], 0)
        attr_tri_series_edge = np.repeat(attr_tri_series_edge, 2)[1:-1]
        attr_tri_series_edge = np.reshape(attr_tri_series_edge, (-1, 2))
        attr_tri = np.r_[attr_tri, attr_tri_series_edge]

        # Deduplicate directed pairs, then retain one orientation of each undirected pair.
        attr_tri = np.unique(attr_tri, axis=0)
        self_loop = np.roll(attr_tri, 1, axis=1)
        attr_iter = 0
        while attr_iter < len(attr_tri):
            delet_idx = [np.where((self_loop == attr_tri[attr_iter]).all(axis=1))[0]]
            attr_tri = np.delete(attr_tri, delet_idx, 0)
            self_loop = np.delete(self_loop, delet_idx, 0)
            attr_iter += 1
        attr_tri = np.array(attr_tri, dtype=int)
        self.area = (np.unique(attr_tri, return_counts=True), extract, attr_tri)
        return self.area

    def interplot(self, ax, size=0.5, c='r', lab='predict trajectory'):
        """Scatter auxiliary points on ``ax`` using size, color ``c``, and label ``lab``."""

        ax.scatter(self.adinter[0][:, 0], self.adinter[0][:, 1], s=size, color=c, label=lab)
        plt.legend()

    def tplot(self, ax, size=0.5, c='b', lab='predict work area', z=0):
        """Plot retained triangles after ``calculatearea``; ``size`` is line width."""

        ax.triplot(self.adinter[0][:, 0], self.adinter[0][:, 1], self.area[1], linewidth=size, color=c,
                   label=lab, zorder=z)

    def uncuteplot(self, ax, size=0.5):
        """Overlay stored retained triangles and auxiliary points for inspection."""

        ax.triplot(self.adinter[0][:, 0], self.adinter[0][:, 1], self.area[1], linewidth=size, zorder=1, color='b',
                   label='triangulation')
        ax.scatter(self.adinter[0][:, 0], self.adinter[0][:, 1], s=size * 2, zorder=2, color='r',
                   label='predict trajectory')
        handle, label = plt.gca().get_legend_handles_labels()
        plt.legend(handles=[handle[0], handle[-1]], labels=[label[0], label[-1]])

def caldis(datapoints):
    """Return adjacent distances, coordinate differences, and unchanged input rows.

    Distances use the first two columns and have length ``len(datapoints) - 1``.
    The current implementation does not remove duplicate points.
    """

    difference = np.diff(datapoints[:, 0:2], axis=0)
    distance = np.sqrt(np.sum(difference * difference, axis=1))
    return distance, difference, datapoints
