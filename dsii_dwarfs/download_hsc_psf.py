import os
import requests

"""
For security reasons, the psf downloader is not a class 
since we would have to store password information
"""

__all__ = ['get_s16a_deep_psf', 'get_s16a_udeep_psf', 'get_s16a_wide_psf',
           'get_pdr2_deep_psf', 'get_pdr2_udeep_psf', 'get_pdr2_wide_psf',]


def _get_psf(url, ra, dec, band, patch, output_path, username, password):

    r = requests.get(url, auth=(username, password))

    filename = "psf_{}_{}.fits".format(patch.name, band)
    filename = os.path.join(output_path, filename)
    if r.status_code == 200:
        with open(filename, 'wb') as out:
            for bits in r.iter_content():
                out.write(bits)
    else:
        raise Exception(
            "PSF NOT FOUND: Check link in get_psf(...)n\Check if correct data version pdr2 or s16a.\nURL: {}".format(
                url))

    return filename


"""
########
# s16a #
########
"""

def get_s16a_deep_psf(ra, dec, band, patch, output_path, username, password):
    url = 'https://hsc-release.mtk.nao.ac.jp/psf/s16a/cgi/getpsf?ra={:0.2f}&dec={:0.2f}&filter={}&rerun=s16a_deep&tract={}&patch={}&type=coadd'.format(
        ra, dec, band, patch.tract, patch.patch)

    return _get_psf(url, ra, dec, band, patch, output_path, username, password)


def get_s16a_udeep_psf(ra, dec, band, patch, output_path, username, password):
    url = 'https://hsc-release.mtk.nao.ac.jp/psf/s16a/cgi/getpsf?ra={:0.2f}&dec={:0.2f}&filter={}&rerun=s16a_udeep&tract={}&patch={}&type=coadd'.format(
        ra, dec, band, patch.tract, patch.patch)

    return _get_psf(url, ra, dec, band, patch, output_path, username, password)

def get_s16a_wide_psf(ra, dec, band, patch, output_path, username, password):
    url = 'https://hsc-release.mtk.nao.ac.jp/psf/s16a/cgi/getpsf?ra={:0.2f}&dec={:0.2f}&filter={}&rerun=s16a_wide&tract={}&patch={}&type=coadd'.format(
        ra, dec, band, patch.tract, patch.patch)

    return _get_psf(url, ra, dec, band, patch, output_path, username, password)

"""
########
# pdr2 #
########
"""

def get_pdr2_deep_psf(ra, dec, band, patch, output_path, username, password):
    url = 'https://hsc-release.mtk.nao.ac.jp/psf/pdr2/cgi/getpsf?ra={:0.2f}&dec={:0.2f}&filter={}&rerun=pdr2_deep&tract={}&patch={}&type=coadd'.format(
        ra, dec, band, patch.tract, patch.patch)

    return _get_psf(url, ra, dec, band, patch, output_path, username, password)


def get_pdr2_udeep_psf(ra, dec, band, patch, output_path, username, password):
    url = 'https://hsc-release.mtk.nao.ac.jp/psf/pdr2/cgi/getpsf?ra={:0.2f}&dec={:0.2f}&filter={}&rerun=pdr2_udeep&tract={}&patch={}&type=coadd'.format(
        ra, dec, band, patch.tract, patch.patch)

    return _get_psf(url, ra, dec, band, patch, output_path, username, password)


def get_pdr2_wide_psf(ra, dec, band, patch, output_path, username, password):
    url = 'https://hsc-release.mtk.nao.ac.jp/psf/pdr2/cgi/getpsf?ra={:0.2f}&dec={:0.2f}&filter={}&rerun=pdr2_wide&tract={}&patch={}&type=coadd'.format(
        ra, dec, band, patch.tract, patch.patch)

    return _get_psf(url, ra, dec, band, patch, output_path, username, password)



master_docstring = """
Parameters
----------
ra, dec : float
    position of psf in deg
band : str
    ['g', 'i', 'r', 'y', 'z']
patch : HSCPatch
    HSCPatch object
output_path : str
    path to save downloaded files
username : str
    User name for hsc-release.mtk.nao.ac.jp
password : str
    Hash for hsc-release.mtk.nao.ac.jp

Returns
-------
filename : str
    Path to downloaded file
"""

for func in __all__:
    locals()[func].__doc__ = master_docstring


HSC_PSF_DOWNLOADERS = {
    's16a_deep' : get_s16a_deep_psf,
    's16a_udeep': get_s16a_udeep_psf,
    's16a_wide' : get_s16a_wide_psf,

    'pdr2_deep' : get_pdr2_deep_psf,
    'pdr2_udeep': get_pdr2_udeep_psf,
    'pdr2_wide' : get_pdr2_wide_psf
}

__all__.append("HSC_PSF_DOWNLOADERS")

