'''
Tools for reading ImageJ files.

Based on code originally written by Luis Pedro Coelho <luis@luispedro.org>,
2012, available at https://gist.github.com/luispedro/3437255, distributed
under the MIT License.

Modified
    - 2014 by Jeffrey Zaremba (@jzaremba), https://github.com/losonczylab/sima
    - 2015 by Scott Lowe (@scottclowe) and Sander Keemink (@swkeemink).
'''

from __future__ import division
from __future__ import unicode_literals

from builtins import str
from builtins import range

import sys
from itertools import product

import numpy as np
from skimage.draw import ellipse
import zipfile

if sys.version_info >= (3, 0):
    import read_roi


def _parse_roi_file_py2(roi_obj):
    """Parses an individual ImageJ ROI

    This is based on the Java implementation:
    http://rsbweb.nih.gov/ij/developer/source/ij/io/RoiDecoder.java.html

    This code can is not guaranteed to work for all C compilers on Windows.

    Parameters
    ----------
    roi_obj : file object
        File object containing a single ImageJ ROI

    Returns
    -------
    dict
        Returns a parsed ROI object, a dictionary with either a `'polygons'`
        or a `'mask'` field.

    Raises
    ------
    IOError
        If there is an error reading the roi file object
    ValueError
        If unable to parse ROI

    """

    # Note:
    # _getX() calls with no assignment are present to move our pointer
    # within the imageJ roi file through bytes that we do not currently use.
    # In line comments indicate what they are; these could be extracted if
    # needed in the future.

    sub_pixel_resolution = 128

    # Other options that are not currently used
    # SPLINE_FIT = 1
    # DOUBLE_HEADED = 2
    # OUTLINE = 4
    # OVERLAY_LABELS = 8
    # OVERLAY_NAMES = 16
    # OVERLAY_BACKGROUNDS = 32
    # OVERLAY_BOLD = 64
    # DRAW_OFFSET = 256

    pos = [4]

    def _get8():
        """Read 1 byte from the roi file object"""
        pos[0] += 1
        s = roi_obj.read(1)
        if not s:
            raise IOError('read_imagej_roi: Unexpected EOF')
        return ord(s)

    def _get16():
        """Read 2 bytes from the roi file object"""
        b0 = _get8()
        b1 = _get8()
        return (b0 << 8) | b1

    def _get16signed():
        """Read a signed 16 bit integer from 2 bytes from roi file object"""
        b0 = _get8()
        b1 = _get8()
        out = (b0 << 8) | b1
        # This is a signed integer, so need to check if the value is
        # positive or negative.
        if b0 > 127:
            out = out - 65536
        return out

    def _get32():
        """Read 4 bytes from the roi file object"""
        s0 = _get16()
        s1 = _get16()
        return (s0 << 16) | s1

    def _getfloat():
        """Read a float from the roi file object"""
        v = np.int32(_get32())
        return v.view(np.float32)

    def _getcoords(z=0):
        """Get the next coordinate of an roi polygon"""
        if options & sub_pixel_resolution:
            getc = _getfloat
            points = np.empty((n_coordinates, 3), dtype=np.float32)
        else:
            getc = _get16
            points = np.empty((n_coordinates, 3), dtype=np.int16)
        points[:, 0] = [getc() for _ in range(n_coordinates)]
        points[:, 1] = [getc() for _ in range(n_coordinates)]
        points[:, 0] += left
        points[:, 1] += top
        points[:, 2] = z
        return points

    magic = roi_obj.read(4)
    if magic != b'Iout':
        raise IOError('read_imagej_roi: Magic number not found')

    _get16()  # version

    roi_type = _get8()
    # Discard extra second Byte:
    _get8()

    if not (0 <= roi_type < 11):
        raise ValueError('read_imagej_roi: \
                          ROI type {} not supported'.format(roi_type))

    top = _get16signed()
    left = _get16signed()
    bottom = _get16signed()
    right = _get16signed()
    n_coordinates = _get16()

    x1 = _getfloat()  # x1
    y1 = _getfloat()  # y1
    x2 = _getfloat()  # x2
    y2 = _getfloat()  # y2
    _get16()  # stroke width
    _get32()  # shape roi size
    _get32()  # stroke color
    _get32()  # fill color
    subtype = _get16()
    if subtype != 0 and subtype != 3:
        raise ValueError('read_imagej_roi: \
                          ROI subtype {} not supported (!= 0)'.format(subtype))
    options = _get16()
    if subtype == 3 and roi_type == 7:
        # ellipse aspect ratio
        aspect_ratio = _getfloat()
    else:
        _get8()  # arrow style
        _get8()  # arrow head size
        _get16()  # rectangle arc size
    z = _get32()  # position
    if z > 0:
        z -= 1  # Multi-plane images start indexing at 1 instead of 0
    _get32()  # header 2 offset

    if roi_type == 0:
        # Polygon
        coords = _getcoords(z)
        coords = coords.astype('float')
        return {'polygons': coords}
    elif roi_type == 1:
        # Rectangle
        coords = [[left, top, z], [right, top, z], [right, bottom, z],
                  [left, bottom, z]]
        coords = np.array(coords).astype('float')
        return {'polygons': coords}
    elif roi_type == 2:
        # Oval
        width = right - left
        height = bottom - top

        # 0.5 moves the mid point to the center of the pixel
        x_mid = (right + left) / 2.0 - 0.5
        y_mid = (top + bottom) / 2.0 - 0.5
        mask = np.zeros((z + 1, right, bottom), dtype=bool)
        for y, x in product(np.arange(top, bottom), np.arange(left, right)):
            mask[z, x, y] = ((x - x_mid) ** 2 / (width / 2.0) ** 2 +
                             (y - y_mid) ** 2 / (height / 2.0) ** 2 <= 1)
        return {'mask': mask}
    elif roi_type == 7:
        if subtype == 3:
            # ellipse
            mask = np.zeros((1, right+10, bottom+10), dtype=bool)
            r_radius = np.sqrt((x2-x1)**2+(y2-y1)**2)/2.0
            c_radius = r_radius*aspect_ratio
            r = (x1+x2)/2-0.5
            c = (y1+y2)/2-0.5
            shpe = mask.shape
            orientation = np.arctan2(y2-y1, x2-x1)
            X, Y = ellipse(r, c, r_radius, c_radius, shpe[1:], orientation)
            mask[0, X, Y] = True
            return {'mask': mask}
        else:
            # Freehand
            coords = _getcoords(z)
            coords = coords.astype('float')
            return {'polygons': coords}

    else:
        try:
            coords = _getcoords(z)
            coords = coords.astype('float')
            return {'polygons': coords}
        except BaseException:
            raise ValueError(
                'read_imagej_roi: ROI type {} not supported'.format(roi_type))


def _parse_roi_file_py3(roi_source):
    """Parses an individual ImageJ ROI

    This implementation utilises the read_roi package, which is more robust
    but does only supports Python 3+ and not Python 2.7.

    Parameters
    ----------
    roi_source : str or file object
        Path to file, or file object containing a single ImageJ ROI

    Returns
    -------
    dict
        Returns a parsed ROI object, a dictionary with either a `'polygons'`
        or a `'mask'` field.

    Raises
    ------
    IOError
        If there is an error reading the roi file object.
    ValueError
        If unable to parse ROI.
    """

    # Use read_roi package to load up the roi as a dictionary
    roi = read_roi.read_roi_file(roi_source)
    # This is a dictionary with a single entry, whose key is the label
    # of the roi. We need to get out its contents, which is another dictionary.
    keys = list(roi.keys())
    if len(keys) == 1:
        roi = roi[keys[0]]

    # Convert the roi dictionary into either polygon or a mask
    if roi['type'] in ('polygon', 'freehand'):
        # Polygon
        coords = np.empty((roi['n'], 3), dtype=np.float)
        coords[:, 0] = roi['x']
        coords[:, 1] = roi['y']
        coords[:, 2] = 0
        return {'polygons': coords}

    width = roi['width']
    height = roi['height']
    left = roi['left']
    top = roi['top']
    right = left + width
    bottom = top - height
    z = 0

    if roi['type'] == 'rectangle':
        # Rectangle
        coords = [[left, top, z], [right, top, z], [right, bottom, z],
                  [left, bottom, z]]
        coords = np.array(coords).astype('float')
        return {'polygons': coords}

    elif roi['type'] == 'oval':
        # Oval
        # 0.5 moves the mid point to the center of the pixel
        x_mid = (right + left) / 2.0 - 0.5
        y_mid = (top + bottom) / 2.0 - 0.5
        mask = np.zeros((z + 1, right, bottom), dtype=bool)
        for y, x in product(np.arange(top, bottom), np.arange(left, right)):
            mask[z, x, y] = ((x - x_mid) ** 2 / (width / 2.0) ** 2 +
                             (y - y_mid) ** 2 / (height / 2.0) ** 2 <= 1)
        return {'mask': mask}

    elif roi['type'] == 'ellipse':
        # ellipse
        mask = np.zeros((1, right+10, bottom+10), dtype=bool)
        r_radius = np.sqrt((x2-x1)**2+(y2-y1)**2)/2.0
        c_radius = r_radius*aspect_ratio
        r = (x1+x2)/2-0.5
        c = (y1+y2)/2-0.5
        shpe = mask.shape
        orientation = np.arctan2(y2-y1, x2-x1)
        X, Y = ellipse(r, c, r_radius, c_radius, shpe[1:], orientation)
        mask[0, X, Y] = True
        return {'mask': mask}

    else:
        raise ValueError(
            'ROI type {} not supported'.format(roi['type'])
        )


# Handle different functions on Python 2/3
parse_roi_file = _parse_roi_file_py3 if sys.version_info >= (3, 0) else _parse_roi_file_py2


def read_imagej_roi_zip(filename):
    """Reads an ImageJ ROI zip set and parses each ROI individually

    Parameters
    ----------
    filename : str
        Path to the ImageJ ROis zip file

    Returns
    -------
    roi_list : list
        List of the parsed ImageJ ROIs
    """
    roi_list = []
    with zipfile.ZipFile(filename) as zf:
        for name in zf.namelist():
            roi = parse_roi_file(zf.open(name))
            if roi is None:
                continue
            roi['label'] = str(name).rstrip('.roi')
            roi_list.append(roi)
        return roi_list
