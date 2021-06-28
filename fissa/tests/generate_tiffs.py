#!/usr/bin/env python
import os
from inspect import getsourcefile

import imageio
import numpy as np
import tifffile

TEST_DIRECTORY = os.path.dirname(os.path.abspath(getsourcefile(lambda: 0)))


def main():
    """
    Generate TIFF test resources, using a variety save methods, formats, dtypes.
    """
    expected = np.array(
        [
            [[-11, 12], [14, 15], [17, 18]],
            [[21, 22], [24, 25], [27, 28]],
            [[31, 32], [34, 35], [37, 38]],
            [[41, 42], [44, 45], [47, 48]],
            [[51, 52], [54, 55], [57, 58]],
            [[61, 62], [64, 55], [67, 68]],
        ]
    )
    output_dir = os.path.join(TEST_DIRECTORY, "resources", "tiffs")
    os.makedirs(output_dir, exist_ok=True)

    for data, dtype, shp in (
        (expected, "uint8", None),
        (expected, "uint8", (3, 2, 3, 2)),
        (expected, "uint8", (2, 1, 3, 3, 2)),
        (expected, "uint8", (2, 3, 1, 1, 3, 2)),
        (expected, "uint16", None),
        (expected, "uint64", None),
        (expected, "int16", None),
        (expected, "int64", None),
        (expected / 10, "float16", None),
        (expected / 10, "float32", None),
        (expected / 10, "float64", None),
    ):
        if dtype.startswith("uint"):
            data = np.abs(data)
        if dtype:
            data = data.astype(dtype)
        if shp:
            data = data.reshape(shp)

        qualifier = ""
        if shp:
            qualifier += "_" + ",".join(str(x) for x in shp)
        qualifier += "_{}".format(dtype)

        if not shp:
            imageio.imwrite(
                os.path.join(output_dir, "imageio.imwrite{}.tif".format(qualifier)),
                data[0],
            )

        tifffile.imsave(
            os.path.join(output_dir, "tifffile.imsave{}.tif".format(qualifier)),
            data,
        )
        tifffile.imsave(
            os.path.join(output_dir, "tifffile.imsave.bigtiff{}.tif".format(qualifier)),
            data,
            bigtiff=True,
        )
        if dtype in ("uint8", "uint16", "float32") and (shp is None or len(shp) <= 5):
            tifffile.imsave(
                os.path.join(
                    output_dir, "tifffile.imsave.imagej{}.tif".format(qualifier)
                ),
                data,
                imagej=True,
            )

        with tifffile.TiffWriter(
            os.path.join(
                output_dir, "TiffWriter.write.discontiguous{}.tif".format(qualifier)
            )
        ) as tif:
            for frame in data:
                tif.write(frame, contiguous=False)

        with tifffile.TiffWriter(
            os.path.join(
                output_dir, "TiffWriter.write.contiguous{}.tif".format(qualifier)
            )
        ) as tif:
            for frame in data:
                tif.write(frame, contiguous=True)

        with tifffile.TiffWriter(
            os.path.join(output_dir, "TiffWriter.save{}.tif".format(qualifier))
        ) as tif:
            for frame in data:
                tif.save(frame)

        if not shp:
            with tifffile.TiffWriter(
                os.path.join(output_dir, "TiffWriter.mixedA{}.tif".format(qualifier))
            ) as tif:
                tif.write(data[:2], contiguous=False)
                tif.write(data[2:4], contiguous=True)
                tif.write(data[4:], contiguous=False)

        if not shp:
            with tifffile.TiffWriter(
                os.path.join(output_dir, "TiffWriter.mixedB{}.tif".format(qualifier))
            ) as tif:
                tif.write(data[:4], contiguous=False)
                tif.write(data[4:], contiguous=False)

        if not shp:
            with tifffile.TiffWriter(
                os.path.join(output_dir, "TiffWriter.mixedC{}.tif".format(qualifier))
            ) as tif:
                tif.write(data[:2], contiguous=True)
                tif.write(data[2:4], contiguous=False)
                tif.save(data[4:])


if __name__ == "__main__":
    print("Using imageio {}".format(imageio.__version__))
    print("Using tifffile {}".format(tifffile.__version__))
    main()
