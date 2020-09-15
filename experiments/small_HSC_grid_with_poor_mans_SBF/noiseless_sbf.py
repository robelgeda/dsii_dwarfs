from astropy.table import Table
from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
from astropy.nddata import block_reduce
from astropy.modeling import models, fitting
from os.path import splitext
import sys

def noiseless_sbf(infile):

    t = Table.read(infile)
    outfile = splitext(infile)[0]+'_sbf.ecsv'
    bands = ['g','r','i','z','y']
    
    sbf = {}
    for b in bands:
        sbf[b] = []
    
    x,y = np.meshgrid(np.arange(256), np.arange(256))
    fit_p = fitting.SimplexLSQFitter()
    
    for filename in t['filename']:
        root = splitext(filename)[0]
        noiseless_filename = root[:-4]+'_noiseless.fits'
        hdu = fits.open(noiseless_filename)
        header = hdu[0].header
        repix = header['REARCSEC']//0.168 # HSC pixel scale
        ellip = header['ELLIP']
        pa = header['PA']
        status = noiseless_filename + " "
        for i,b in enumerate(bands):    
            img = hdu[0].data[i,:,:]
            mod = models.Sersic2D(amplitude = 0.001, r_eff = repix , n=1, x_0=128, y_0=128, ellip=ellip, theta=pa)
            mod.r_eff.fixed = True
            mod.n.fixed = True
            mod.x_0.fixed = True
            mod.y_0.fixed = True
            mod.ellip.fixed = True
            sersic = fit_p(mod,x,y,img,maxiter=400)
            img_fit = sersic(x,y)
            mask = img_fit > 0.0001 # Take the variance only of brightish pixels
            normalized_residual = (img-img_fit)/np.sqrt(img_fit)
            var = normalized_residual[mask].var()
            sbf[b] += [var]
            status += f"{var:.5f} "
        print(status)
        hdu.close()
    
    for b in bands:
        sbf[b] = np.array(sbf[b])
        t['sbf_'+b] = sbf[b]
    
    t.write(outfile)

if __name__ == "__main__":
    noiseless_sbf(sys.argv[1])

