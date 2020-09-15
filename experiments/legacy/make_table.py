from astropy.io import fits
from astropy.table import Table
import numpy as np
import glob
import sys

def create_table():
    colnames = [
        'filename',
        'logmass',
        'distance',
        'rekpc',
        'rearcsec',
        'FeH',
        'age',
        'ellip',
        'pa',
        'scale',
        'oversamp',
        'isofile',
        'gamma_g',
        'gamma_r',
        'gamma_i',
        'gamma_z',
        'gamma_y',
        'noise_g',
        'noise_r',
        'noise_i',
        'noise_z',
        'noise_y',
    ]
    coltypes = [
        'S60',
        'f8',
        'f8',
        'f8',
        'f8',
        'f8',
        'f8',
        'f8',
        'f8',
        'f8',
        'i4',
        'S60',
        'f8',
        'f8',
        'f8',
        'f8',
        'f8',
        'f8',
        'f8',
        'f8',
        'f8',
        'f8',
    ]
    t = Table(names=colnames,dtype=coltypes)
    return t

def read_fits(fitsfile,kwnames):
    bands = ['g','r','i','z','y']
    hdulist = fits.open(fitsfile)
    h = hdulist[0].header
    row = [fitsfile]
    for c in kwnames:
        row += [h[c]]
    hdulist.close()
    return row

def fits_to_table(directory='stamps1',suffix='_HSC'):
    files = glob.glob(directory+'/'+'*'+suffix+'.fits')
    t = create_table()
    for f in files:
        print(f)
        row = read_fits(f,t.colnames[1:])
        t.add_row(row)
    for c in t.colnames:
       if t[c].dtype == 'float64':
           t[c].format = '.4f'
    return t

if __name__ == "__main__":
    if len(sys.argv) > 1:
        t = fits_to_table(directory=sys.argv[1])
        t.write(sys.argv[1]+'.ecsv',format='ascii.ecsv')
    else:
        t=fits_to_table()
        t.write('stamps1.ecsv',format='ascii.ecsv')
