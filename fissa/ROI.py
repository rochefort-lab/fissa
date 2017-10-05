"""The functions below were taken from the sima package
http://www.losonczylab.org/sima version 1.3.0.

@swkeemink: Commented out lines 52-55 to remove a warning message for FISSA.

License
-------
This file is Copyright (C) 2014 The Trustees of Columbia University in the
City of New York.

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License
as published by the Free Software Foundation; either version 2
of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
from builtins import filter

from scipy.sparse import lil_matrix
import numpy as np
from itertools import product
# from warnings import warn

from shapely.geometry import MultiPolygon, Polygon, Point


def poly2mask(polygons, im_size):
    """Converts polygons to a sparse binary mask.

    >>> from sima.ROI import poly2mask
    >>> poly1 = [[0,0], [0,1], [1,1], [1,0]]
    >>> poly2 = [[0,1], [0,2], [2,2], [2,1]]
    >>> mask = poly2mask([poly1, poly2], (3, 3))
    >>> mask[0].todense()
    matrix([[ True, False, False],
            [ True,  True, False],
            [False, False, False]], dtype=bool)

    Parameters
    ----------
    polygons : sequence of coordinates or sequence of Polygons
        A sequence of polygons where each is either a sequence of (x,y) or
        (x,y,z) coordinate pairs, an Nx2 or Nx3 numpy array, or a Polygon
        object.
    im_size : tuple
        Final size of the resulting mask

    Output
    ------
    mask
        A list of sparse binary masks of the points contained within the
        polygons, one mask per plane

    """
    if len(im_size) == 2:
        im_size = (1,) + im_size

    polygons = _reformat_polygons(polygons)

    mask = np.zeros(im_size, dtype=bool)
    for poly in polygons:
        # assuming all points in the polygon share a z-coordinate
        z = int(np.array(poly.exterior.coords)[0][2])
#        if z > im_size[0]:
#            warn('Polygon with zero-coordinate {} '.format(z) +
#                 'cropped using im_size = {}'.format(im_size))
#            continue
        x_min, y_min, x_max, y_max = poly.bounds

        # Shift all points by 0.5 to move coordinates to corner of pixel
        shifted_poly = Polygon(np.array(poly.exterior.coords)[:, :2] - 0.5)

        points = [Point(x, y) for x, y in
                  product(np.arange(int(x_min), np.ceil(x_max)),
                          np.arange(int(y_min), np.ceil(y_max)))]
        points_in_poly = list(filter(shifted_poly.contains, points))
        for point in points_in_poly:
            xx, yy = point.xy
            x = int(xx[0])
            y = int(yy[0])
            if 0 <= y < im_size[1] and 0 <= x < im_size[2]:
                mask[z, y, x] = True
    masks = []
    for z_coord in np.arange(mask.shape[0]):
        masks.append(lil_matrix(mask[z_coord, :, :]))
    return masks


def _reformat_polygons(polygons):
    """Convert polygons to a MulitPolygon.

    Accepts one more more sequence of 2- or 3-element sequences or a sequence
    of shapely Polygon objects.

    Parameters
    ----------
    polygons : sequence of 2- or 3-element coordinates or sequence of Polygons
        Polygon(s) to be converted to a MulitPolygon.  Coordinates are used to
        initialize a shapely MultiPolygon, and thus should follow a (x, y, z)
        coordinate space convention.

    Returns
    -------
    MultiPolygon

    """
    if len(polygons) == 0:
        # Just return an empty MultiPolygon
        return MultiPolygon([])
    elif isinstance(polygons, Polygon):
        polygons = [polygons]
    elif isinstance(polygons[0], Polygon):
        # polygons is already a list of polygons
        pass
    else:
        # We got some sort of sequence of sequences, ensure it has the
        # correct depth and convert to Polygon objects
        try:
            Polygon(polygons[0])
        except (TypeError, AssertionError):
            polygons = [polygons]
        new_polygons = []
        for poly in polygons:
            # Polygon.simplify with tolerance=0 will return the exact same
            # polygon with co-linear points removed
            new_polygons.append(Polygon(poly).simplify(tolerance=0))
        polygons = new_polygons

    # Polygon.exterior.coords is not settable, need to initialize new objects
    z_polygons = []
    for poly in polygons:
        if poly.has_z:
            z_polygons.append(poly)
        else:
            # warn('Polygon initialized without z-coordinate. ' +
            #    'Assigning to zeroth plane (z = 0)')
            z_polygons.append(
                Polygon([point + (0,) for point in poly.exterior.coords]))
    return MultiPolygon(z_polygons)
