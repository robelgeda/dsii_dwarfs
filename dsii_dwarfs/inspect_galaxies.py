import numpy as np
from make_galaxies import SimulateRandomDwarfElliptical,simulate_HSC
import make_galaxies
from astropy.convolution import convolve_fft
import matplotlib.pyplot as plt

def inspect(img):
    vmin=np.percentile(img,1.)
    vmax=np.percentile(img,99.5)
#    plt.figure(figsize=(10,10))
    plt.imshow(img,vmin=vmin,vmax=vmax)
    
def inspectall(sim,img):
    plt.figure(figsize=(20,10))
    for i,b in enumerate(sim.bands):
        vmin=np.percentile(img[b],1.)
        vmax=np.percentile(img[b],99.5)
        plt.subplot(1,5,i+1)
        plt.imshow(img[b],vmin=vmin,vmax=vmax)
        
def inspectsum(sim,convolved=True):
    summed_image = sim.HSCimage['g']
    if convolved:
        for b in sim.bands:
            pshape = (np.array(sim.psf[b].shape)/sim.oversampling).astype('int32')
            psf = make_galaxies.rebin(sim.psf[b],pshape[0],pshape[1])
            summed_image += convolve_fft(sim.HSCimage[b],psf)
        else:
            summed_image += sim.HSCimage[b]
    return(summed_image)
