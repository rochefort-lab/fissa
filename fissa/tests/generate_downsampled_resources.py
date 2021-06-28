#!/usr/bin/env python

from __future__ import division, unicode_literals

import glob
import os
import shutil
import sys
import zipfile
from itertools import product

import numpy as np
import scipy.ndimage
import tifffile
from skimage.draw import ellipse

from fissa import extraction, readimagejrois


def maybe_make_dir(dirname):
    """
    If it doesn't exist, make a directory. Compatible with Python 2 and 3.
    """
    if sys.version_info[0] >= 3:
        os.makedirs(dirname, exist_ok=True)
    elif os.path.isdir(dirname):
        return
    try:
        os.makedirs(dirname)
    except OSError as err:
        if err.errno != 17:
            raise


def downscale_roi(source_file, dest_file, downsamp=None, offsets=None):
    """
    Downscale a ROI appearing in an ImageJ ROI file.

    The co-ordinates of the ROI are adjusted, and all metadata remains the same.

    Parameters
    ----------
    source_file : str
        Path to file containing the original ROI, in ImageJ ROI format.
    dest_file : str
        Path where the output file will be written. If a file with this
        filename already exists, it will be overwritten.
    downsamp : list or array_like, optional
        Downsampling factor for [x, y]. This must be a length two list,
        tuple, or :term:`array-like`. Default is `[1, 1]`, which corresponds to
        an output roi the same size as the original roi. If `downsamp=[2, 2]`,
        the output will be half the size.
    offsets : list or array_like, optional
        Amount by which to offset the [x, y] co-ordinates of the roi.
        Default is `[0, 0]`.

    Notes
    -----
    Based on `fissa.readimagejrois.read_roi`. The original version of
    `read_roi` was written by Luis Pedro Coelho, released under the MIT
    License.
    """
    if downsamp is None:
        downsamp = [1, 1]
    if offsets is None:
        offsets = [0, 0]
    with open(source_file, "rb") as inhand, open(dest_file, "wb") as outhand:

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
            """Read 1 byte from the roi file object."""
            pos[0] += 1
            s = inhand.read(1)
            if not s:
                raise IOError("read_imagej_roi: Unexpected EOF")
            return ord(s)

        def _write8(s):
            """Write 1 byte to a roi file object."""
            # outhand.write(chr(s))  # When opened with text mode, Py2.7
            # outhand.write(bytes([s]))  # When opened with b mode, Py3
            outhand.write(bytes(bytearray([s])))  # b mode Py2/3

        def _get16():
            """Read 2 bytes from the roi file object."""
            b0 = _get8()
            b1 = _get8()
            return (b0 << 8) | b1

        def _write16(s):
            """Write 2 byte to a roi file object."""
            b0 = (s >> 8) & 0xFF
            _write8(b0)
            b1 = s & 0xFF
            _write8(b1)

        def _get16signed():
            """Read a signed 16 bit integer from 2 bytes from roi file object."""
            b0 = _get8()
            b1 = _get8()
            out = (b0 << 8) | b1
            # This is a signed integer, so need to check if the value is
            # positive or negative.
            if b0 > 127:
                out = out - 65536
            return out

        def _write16signed(s):
            """Write a signed 16 bit integer to 2 bytes in a roi file object."""
            if s < 0:
                s += 65536
            _write16(s)

        def _get32():
            """Read 4 bytes from the roi file object."""
            s0 = _get16()
            s1 = _get16()
            return (s0 << 16) | s1

        def _write32(s):
            """Write 4 bytes to the roi file object."""
            s0 = (s >> 16) & 0xFFFF
            _write16(s0)
            s1 = s & 0xFFFF
            _write16(s1)

        def _getfloat():
            """Read a float from the roi file object."""
            v = np.int32(_get32())
            return v.view(np.float32)

        def _writefloat(f):
            """Write a float from the roi file object."""
            f = f.astype(np.float32)
            s = int(f.view(np.int32))
            _write32(s)

        def _getcoords(z=0):
            """Get the next coordinate of an roi polygon."""
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

        def _writecoords(points):
            """Write the coordinate of an roi polygon."""
            if options & sub_pixel_resolution:
                writec = _writefloat
                convert = lambda x: x.astype(np.float32)  # noqa: E731
                # points = np.empty((n_coordinates, 3), dtype=np.float32)
            else:
                writec = _write16signed
                convert = lambda x: np.round(x).astype(np.int16)  # noqa: E731
                # points = np.empty((n_coordinates, 3), dtype=np.int16)
            x = (points[:, 0] - left) / downsamp[0]
            y = (points[:, 1] - top) / downsamp[1]
            for xi in x:
                writec(convert(xi))
            for yi in y:
                writec(convert(yi))

        magic = inhand.read(4)
        if magic != b"Iout":
            raise IOError(
                "read_imagej_roi: Magic number not found."
                " Expected: {}. Detected: {}."
                "".format(b"Iout", magic)
            )
        outhand.write(magic)

        b = _get16()  # version
        _write16(b)

        roi_type = _get8()
        _write8(roi_type)
        # Discard extra second Byte:
        b = _get8()
        _write8(b)

        if not (0 <= roi_type < 11):
            raise ValueError(
                "read_imagej_roi: \
                              ROI type {} not supported".format(
                    roi_type
                )
            )

        top = _get16signed()
        _write16signed(int(np.round((top + offsets[1]) / downsamp[1])))
        left = _get16signed()
        _write16signed(int(np.round((left + offsets[0]) / downsamp[0])))
        bottom = _get16signed()
        _write16signed(int(np.round((bottom + offsets[1]) / downsamp[1])))
        right = _get16signed()
        _write16signed(int(np.round((right + offsets[0]) / downsamp[0])))
        n_coordinates = _get16()
        _write16(n_coordinates)

        x1 = _getfloat()  # x1
        _writefloat((x1 + offsets[0]) / downsamp[0])
        y1 = _getfloat()  # y1
        _writefloat((y1 + offsets[1]) / downsamp[1])
        x2 = _getfloat()  # x2
        _writefloat((x2 + offsets[0]) / downsamp[0])
        y2 = _getfloat()  # y2
        _writefloat((y2 + offsets[1]) / downsamp[1])
        b = _get16()  # stroke width
        _write16(b)
        b = _get32()  # shape roi size
        _write32(b)
        b = _get32()  # stroke color
        _write32(b)
        b = _get32()  # fill color
        _write32(b)
        subtype = _get16()
        _write16(subtype)
        if subtype != 0 and subtype != 3:
            raise ValueError(
                "read_imagej_roi: \
                              ROI subtype {} not supported (!= 0)".format(
                    subtype
                )
            )
        options = _get16()
        _write16(options)
        if subtype == 3 and roi_type == 7:
            # ellipse aspect ratio
            aspect_ratio = _getfloat()
            _writefloat(aspect_ratio)
        else:
            b = _get8()  # arrow style
            _write8(b)
            b = _get8()  # arrow head size
            _write8(b)
            b = _get16()  # rectangle arc size
            _write16(b)
        z = _get32()  # position
        _write32(z)
        if z > 0:
            z -= 1  # Multi-plane images start indexing at 1 instead of 0
        b = _get32()  # header 2 offset
        _write32(b)

        if roi_type == 0:
            # Polygon
            coords = _getcoords(z)
            _writecoords(coords)
            coords = coords.astype("float")
            # return {'polygons': coords}
        elif roi_type == 1:
            # Rectangle
            coords = [
                [left, top, z],
                [right, top, z],
                [right, bottom, z],
                [left, bottom, z],
            ]
            coords = np.array(coords).astype("float")
            # return {'polygons': coords}
        elif roi_type == 2:
            # Oval
            width = right - left
            height = bottom - top

            # 0.5 moves the mid point to the center of the pixel
            x_mid = (right + left) / 2.0 - 0.5
            y_mid = (top + bottom) / 2.0 - 0.5
            mask = np.zeros((z + 1, right, bottom), dtype=bool)
            for y, x in product(np.arange(top, bottom), np.arange(left, right)):
                mask[z, x, y] = (x - x_mid) ** 2 / (width / 2.0) ** 2 + (
                    y - y_mid
                ) ** 2 / (height / 2.0) ** 2 <= 1
            # return {'mask': mask}
        elif roi_type == 7:
            if subtype == 3:
                # ellipse
                mask = np.zeros((1, right + 10, bottom + 10), dtype=bool)
                r_radius = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2) / 2.0
                c_radius = r_radius * aspect_ratio
                r = (x1 + x2) / 2 - 0.5
                c = (y1 + y2) / 2 - 0.5
                shpe = mask.shape
                orientation = np.arctan2(y2 - y1, x2 - x1)
                X, Y = ellipse(r, c, r_radius, c_radius, shpe[1:], orientation)
                mask[0, X, Y] = True
                # return {'mask': mask}
            else:
                # Freehand
                coords = _getcoords(z)
                _writecoords(coords)
                coords = coords.astype("float")
                # return {'polygons': coords}

        else:
            try:
                coords = _getcoords(z)
                _writecoords(coords)
                coords = coords.astype("float")
                # return {'polygons': coords}
            except BaseException:
                raise ValueError(
                    "read_imagej_roi: ROI type {} not supported".format(roi_type)
                )

        # Copy the rest of the file, line by line
        for line in inhand:
            outhand.write(line)


def main(name=None, roi_ids=None, x_down=4, y_down=None, t_down=10):
    """
    Convert example data into downsampled test data.
    """
    # Default arguments
    if y_down is None:
        y_down = x_down

    if name is None:
        import datetime

        name = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    # Input resolution
    this_dir = os.path.dirname(os.path.abspath(__file__))
    repo_dir = os.path.dirname(os.path.dirname(this_dir))
    rois_location = os.path.join(
        repo_dir,
        "examples",
        "exampleData",
        "20150429.zip",
    )
    images_location = os.path.join(
        repo_dir,
        "examples",
        "exampleData",
        "20150529",
    )

    # Output configuration
    output_folder_base = os.path.join(this_dir, "resources")
    roi_extract_dir = os.path.join(output_folder_base, "_build_rois")

    output_folder = os.path.join(output_folder_base, name)

    datahandler = extraction.DataHandlerTifffile

    # Extract the rois from the zip file
    print("Extracting rois from {} into {}".format(rois_location, roi_extract_dir))
    with zipfile.ZipFile(rois_location, "r") as zr:
        zr.extractall(roi_extract_dir)

    # Read in rois
    roi_list = readimagejrois.read_imagej_roi_zip(rois_location)
    if roi_ids is None or len(roi_ids) == 0:
        roi_ids = range(len(roi_list))

    image_paths = sorted(glob.glob(os.path.join(images_location, "*.tif*")))
    img = datahandler.image2array(image_paths[0])

    # Workout where to crop images when downscaling to include ROIs
    buffer = 19
    off_x = off_y = 1e9
    end_x = end_y = 0
    for roi_id in roi_ids:
        mn = roi_list[roi_id]["polygons"].min(0)
        mx = roi_list[roi_id]["polygons"].max(0)
        off_x = min(off_x, max(0, int(mn[0] - buffer)))
        end_x = max(end_x, min(img.shape[2], int(mx[0] + buffer)))
        off_y = min(off_y, max(0, int(mn[1] - buffer)))
        end_y = max(end_y, min(img.shape[1], int(mx[1] + buffer)))

    # Downsample images
    for img_pth in image_paths:
        img = datahandler.image2array(img_pth)
        print("Loaded image {} shaped {}".format(img_pth, img.shape))
        img_dwn = scipy.ndimage.uniform_filter(img, size=[t_down, y_down, x_down])
        img_dwn = img_dwn[::t_down, off_y:end_y:y_down, off_x:end_x:x_down]
        img_dwn_pth = os.path.join(output_folder, "images", os.path.basename(img_pth))
        print(
            "Saving downsampled image shaped {} as {}".format(
                img_dwn.shape, img_dwn_pth
            )
        )
        maybe_make_dir(os.path.dirname(img_dwn_pth))
        tifffile.imsave(img_dwn_pth, img_dwn)

    # Downscale roi(s)
    for roi_id in roi_ids:
        roi_raw_pth = os.path.join(
            roi_extract_dir,
            "{:02d}.roi".format(roi_id + 1),
        )
        roi_dwn_pth = os.path.join(
            output_folder, "rois", "{:02d}.roi".format(roi_id + 1)
        )
        print(
            "Downsampling ROI {} with factor ({}, {}); saving as {}"
            "".format(roi_raw_pth, x_down, y_down, roi_dwn_pth)
        )
        maybe_make_dir(os.path.dirname(roi_dwn_pth))
        downscale_roi(roi_raw_pth, roi_dwn_pth, [x_down, y_down], [-off_x, -off_y])

    # Turn rois into a zip file
    roi_zip_pth = os.path.join(output_folder, "rois.zip")
    print("Zipping rois {} as {}".format(os.path.dirname(roi_dwn_pth), roi_zip_pth))
    maybe_make_dir(os.path.dirname(roi_dwn_pth))
    with zipfile.ZipFile(roi_zip_pth, "w") as zr:
        for roi_id in roi_ids:
            roi_dwn_pth = os.path.join(
                output_folder, "rois", "{:02d}.roi".format(roi_id + 1)
            )
            zr.write(roi_dwn_pth, os.path.basename(roi_dwn_pth))

    # Remove _build_rois directory
    shutil.rmtree(roi_extract_dir)


def get_parser():
    """
    Build parser for generate_downsampled_resources command line interface.

    Returns
    -------
    parser : argparse.ArgumentParser
        CLI argument parser.
    """
    import argparse

    prog = os.path.split(sys.argv[0])[1]
    if prog == "__main__.py" or prog == "__main__":
        prog = os.path.split(__file__)[1]
    parser = argparse.ArgumentParser(
        prog=prog,
        description="Downsample example dataset to create test data",
        add_help=False,
    )

    parser.add_argument(
        "-h",
        "--help",
        action="help",
        help="Show this help message and exit.",
    )

    parser.add_argument(
        "--name",
        type=str,
        help="Name of output dataset (default: current datetime)",
    )
    parser.add_argument(
        "--roi-id",
        nargs="+",
        type=int,
        help="""
            ROI indices to include. The image will be downsampled to include
            the ROIs and their surroundings. Multiple ROIs can be specified.
            If omitted, all ROIs are used.
        """,
    )
    parser.add_argument(
        "--x-down",
        type=int,
        default=4,
        help="Downsampling factor for x dimension (default: 4)",
    )
    parser.add_argument(
        "--y-down",
        type=int,
        default=4,
        help="Downsampling factor for y dimension (default: 4)",
    )
    parser.add_argument(
        "--t-down",
        type=int,
        default=10,
        help="Downsampling factor for time dimension (default: 10)",
    )

    return parser


if __name__ == "__main__":
    __package__ = "fissa.tests.generate_downsampled_resources"
    parser = get_parser()
    kwargs = vars(parser.parse_args())
    roi_ids = kwargs.pop("roi_id")
    main(roi_ids=roi_ids, **kwargs)
