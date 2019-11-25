import os
from glob import glob

class HSCPatch:
    """
    Holds information about the location of image
    files for each path.
    Each patch has a g, i, r, y and z band image.
    Each patch needs to be set by accssing the attribute.

    Parameters
    ----------
    name : string
        Name of patch

    """

    def __init__(self, name):
        self.name = name
        self.tract, self.patch = name.split("_")

        self.g = None  # Path to g band image
        self.i = None  # Path to i band image
        self.r = None  # Path to r band image
        self.y = None  # Path to y band image
        self.z = None  # Path to z band image

    def info(self):
        print("name:\t{}".format(self.name))
        print("g:\t{}".format(self.g))
        print("i:\t{}".format(self.i))
        print("r:\t{}".format(self.r))
        print("y:\t{}".format(self.y))
        print("z:\t{}".format(self.z))

    @property
    def bands(self):
        """"
        Make sure to match the following order:
        'g','r','i','z','y'
        """
        return {
            "g": self.g,
            "r": self.r,
            "i": self.i,
            "z": self.z,
            "y": self.y,
        }


def find_hsc_files(hsc_top_path, verbose=True):
    """
    This function takes the top level path and searches for
    HSC images. It will try to find and organize associated
    images into a HSCPatch patch object. If a band is missing
    that patch (set of images) will not be included in the result.

    Parameters
    ----------
    hsc_top_path : string
        path to top level dir containing the patch images

    verbose : bool
        Pint out info

    Returns
    -------
    patch_dict : dict
        A dict of patch objects with the patch names as keys
    """

    patch_dict = {}

    fb = glob(hsc_top_path + "/**/calexp-HSC*.gz", recursive=True)  # Filebase

    for f in fb:

        fbase = os.path.basename(f)
        band, *name = fbase.split(".")[0].split("-")[2:]
        name = "_".join(name)
        if name not in patch_dict:
            patch_dict[name] = HSCPatch(name)
        setattr(patch_dict[name], band.lower(), f)

    for name in patch_dict:
        for band in ['g', 'i', 'r', 'y', 'z']:
            assert getattr(patch_dict[name], band) is not None, "error, patch {} is missing {} band".format(name, band)

        if verbose:
            print("✓ {}".format(name))

    if verbose:
        print("\nNumber of Patches: ", len(patch_dict))

    return patch_dict