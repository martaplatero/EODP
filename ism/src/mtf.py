from math import pi
from config.ismConfig import ismConfig
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.special import j1
from numpy.matlib import repmat
from common.io.readMat import writeMat
from common.plot.plotMat2D import plotMat2D
from scipy.interpolate import interp2d
from numpy.fft import fftshift, ifft2
import os

class mtf:
    """
    Class MTF. Collects the analytical modelling of the different contributions
    for the system MTF
    """
    def __init__(self, logger, outdir):
        self.ismConfig = ismConfig()
        self.logger = logger
        self.outdir = outdir

    def system_mtf(self, nlines, ncolumns, D, lambd, focal, pix_size,
                   kLF, wLF, kHF, wHF, defocus, ksmear, kmotion, directory, band):
        """
        System MTF
        :param nlines: Lines of the TOA
        :param ncolumns: Columns of the TOA
        :param D: Telescope diameter [m]
        :param lambd: central wavelength of the band [m]
        :param focal: focal length [m]
        :param pix_size: pixel size in meters [m]
        :param kLF: Empirical coefficient for the aberrations MTF for low-frequency wavefront errors [-]
        :param wLF: RMS of low-frequency wavefront errors [m]
        :param kHF: Empirical coefficient for the aberrations MTF for high-frequency wavefront errors [-]
        :param wHF: RMS of high-frequency wavefront errors [m]
        :param defocus: Defocus coefficient (defocus/(f/N)). 0-2 low defocusing
        :param ksmear: Amplitude of low-frequency component for the motion smear MTF in ALT [pixels]
        :param kmotion: Amplitude of high-frequency component for the motion smear MTF in ALT and ACT
        :param directory: output directory
        :return: mtf
        """

        self.logger.info("Calculation of the System MTF")

        # Calculate the 2D relative frequencies
        self.logger.debug("Calculation of 2D relative frequencies")
        fn2D, fr2D, fnAct, fnAlt = self.freq2d(nlines, ncolumns, D, lambd, focal, pix_size)

        # Diffraction MTF
        self.logger.debug("Calculation of the diffraction MTF")
        Hdiff = self.mtfDiffract(fr2D)

        # Defocus
        Hdefoc = self.mtfDefocus(fr2D, defocus, focal, D)

        # WFE Aberrations
        Hwfe = self.mtfWfeAberrations(fr2D, lambd, kLF, wLF, kHF, wHF)

        # Detector
        Hdet  = self. mtfDetector(fn2D)

        # Smearing MTF
        Hsmear = self.mtfSmearing(fnAlt, ncolumns, ksmear)

        # Motion blur MTF
        Hmotion = self.mtfMotion(fn2D, kmotion)

        # Calculate the System MTF
        self.logger.debug("Calculation of the Sysmtem MTF by multiplying the different contributors")
        Hsys = Hdiff*Hdefoc*Hwfe*Hdet*Hsmear*Hmotion
        print(Hsys)

        if self.ismConfig.save_mtfs:
            writeMat(directory, f"Hdiff_{band}", Hdiff)
            writeMat(directory, f"Hdefoc_{band}", Hdefoc)
            writeMat(directory, f"Hwfe_{band}", Hwfe)
            writeMat(directory, f"Hdet_{band}", Hdet)
            writeMat(directory, f"Hsmear_{band}", Hsmear)
            writeMat(directory, f"Hmotion_{band}", Hmotion)
            writeMat(directory, f"Hsys_{band}", Hsys)

        # Plot cuts ACT/ALT of the MTF
        self.plotMtf(Hdiff, Hdefoc, Hwfe, Hdet, Hsmear, Hmotion, Hsys, nlines, ncolumns, fnAct, fnAlt, directory, band)


        return Hsys

    def freq2d(self,nlines, ncolumns, D, lambd, focal, w):
        """
        Calculate the relative frequencies 2D (for the diffraction MTF)
        :param nlines: Lines of the TOA
        :param ncolumns: Columns of the TOA
        :param D: Telescope diameter [m]
        :param lambd: central wavelength of the band [m]
        :param focal: focal length [m]
        :param w: pixel size in meters [m]
        :return fn2D: normalised frequencies 2D (f/(1/w))
        :return fr2D: relative frequencies 2D (f/(1/fc))
        :return fnAct: 1D normalised frequencies 2D ACT (f/(1/w))
        :return fnAlt: 1D normalised frequencies 2D ALT (f/(1/w))
        """

        fstepAlt = 1 / nlines / w
        fstepAct = 1 / ncolumns / w

        eps = 1e-6
        fAlt = np.arange(-1 / (2 * w), 1 / (2 * w) - eps, fstepAlt)
        fAct = np.arange(-1 / (2 * w), 1 / (2 * w) - eps, fstepAct)

        [fAltxx,fActxx] = np.meshgrid(fAlt,fAct,indexing='ij')
        f2D=np.sqrt(fAltxx*fAltxx + fActxx*fActxx)

        fc = D/(focal*lambd)

        fn2D = f2D/(1/w)

        fr2D = f2D/fc

        fnAct = fAct/(1/w)
        fnAlt = fAlt/(1/w)
        
        return fn2D, fr2D, fnAct, fnAlt

    def mtfDiffract(self,fr2D):
        """
        Optics Diffraction MTF
        :param fr2D: 2D relative frequencies (f/fc), where fc is the optics cut-off frequency
        :return: diffraction MTF
        """

        Hdiff = 2/pi * (np.arccos(fr2D) - fr2D * np.sqrt(1 - fr2D ** 2))

        return Hdiff


    def mtfDefocus(self, fr2D, defocus, focal, D):
        """
        Defocus MTF
        :param fr2D: 2D relative frequencies (f/fc), where fc is the optics cut-off frequency
        :param defocus: Defocus coefficient (defocus/(f/N)). 0-2 low defocusing
        :param focal: focal length [m]
        :param D: Telescope diameter [m]
        :return: Defocus MTF
        """

        x = pi * defocus * fr2D * (1 - fr2D)

        Hdefoc = 2 * j1(x) / x

        return Hdefoc

    def mtfWfeAberrations(self, fr2D, lambd, kLF, wLF, kHF, wHF):
        """
        Wavefront Error Aberrations MTF
        :param fr2D: 2D relative frequencies (f/fc), where fc is the optics cut-off frequency
        :param lambd: central wavelength of the band [m]
        :param kLF: Empirical coefficient for the aberrations MTF for low-frequency wavefront errors [-]
        :param wLF: RMS of low-frequency wavefront errors [m]
        :param kHF: Empirical coefficient for the aberrations MTF for high-frequency wavefront errors [-]
        :param wHF: RMS of high-frequency wavefront errors [m]
        :return: WFE Aberrations MTF
        """

        wfe_term = kLF * (wLF / lambd) ** 2 + kHF * (wHF / lambd) ** 2

        Hwfe = np.exp(-fr2D * (1 - fr2D) * wfe_term)
        
        return Hwfe

    def mtfDetector(self,fn2D):
        """
        Detector MTF
        :param fnD: 2D normalised frequencies (f/(1/w))), where w is the pixel width
        :return: detector MTF
        """

        Hdet = np.abs(np.sinc(fn2D))

        return Hdet

    def mtfSmearing(self, fnAlt, ncolumns, ksmear):
        """
        Smearing MTF
        :param ncolumns: Size of the image ACT
        :param fnAlt: 1D normalised frequencies 2D ALT (f/(1/w))
        :param ksmear: Amplitude of low-frequency component for the motion smear MTF in ALT [pixels]
        :return: Smearing MTF
        """

        nlines = len(fnAlt)
        Hsmear = np.zeros((nlines, ncolumns))

        for col in range(ncolumns):
            Hsmear[:, col] = np.sinc(ksmear * fnAlt)

        return Hsmear

    def mtfMotion(self, fn2D, kmotion):
        """
        Motion blur MTF
        :param fnD: 2D normalised frequencies (f/(1/w))), where w is the pixel width
        :param kmotion: Amplitude of high-frequency component for the motion smear MTF in ALT and ACT
        :return: detector MTF
        """

        Hmotion = np.sinc(kmotion * fn2D)

        return Hmotion

    def plotMtf(self,Hdiff, Hdefoc, Hwfe, Hdet, Hsmear, Hmotion, Hsys, nlines, ncolumns, fnAct, fnAlt, directory, band):
        """
        Plotting the system MTF and all of its contributors
        :param Hdiff: Diffraction MTF
        :param Hdefoc: Defocusing MTF
        :param Hwfe: Wavefront electronics MTF
        :param Hdet: Detector MTF
        :param Hsmear: Smearing MTF
        :param Hmotion: Motion blur MTF
        :param Hsys: System MTF
        :param nlines: Number of lines in the TOA
        :param ncolumns: Number of columns in the TOA
        :param fnAct: normalised frequencies in the ACT direction (f/(1/w))
        :param fnAlt: normalised frequencies in the ALT direction (f/(1/w))
        :param directory: output directory
        :param band: band
        :return: N/A
        """
        #TODO
        # tenemos que implementarlo luego para replicar una imagen que es simetrica. una imagen que esta en el
        # documento grande. Figure 7-52 en pg 65. use the fn2D to cut the 100x150 through the 150.
        # para el report cross validation de sus outputs con los nuestros. validar los mtf. irsf toa and irsf optical.
        # all out outputs and hers. at the plot of the system mtf and explain what are the dominating effects. the two dimensining
        # mtf of out system.


        os.makedirs(directory, exist_ok=True)
        mid_alt = nlines // 2
        mid_act = ncolumns // 2

        # ----------------------------------------------------------------------
        # 1. Slice ACT (Figure 7-51)
        # ----------------------------------------------------------------------
        # Only positive frequencies up to Nyquist (fn = 0.5)
        act_pos = fnAct >= 0.0
        x_act = fnAct[act_pos]

        plt.figure(figsize=(9, 5.5))
        plt.plot(x_act, Hdiff[mid_alt, act_pos], label='Diffraction MTF', linewidth=1.2)
        plt.plot(x_act, Hdefoc[mid_alt, act_pos], label='Defocus MTF', linewidth=1.2)
        plt.plot(x_act, Hwfe[mid_alt, act_pos], label='WFE Aberrations MTF', linewidth=1.2)
        plt.plot(x_act, Hdet[mid_alt, act_pos], label='Detector MTF', linewidth=1.2)
        plt.plot(x_act, Hsmear[mid_alt, act_pos], label='Smearing MTF', linewidth=1.2)
        plt.plot(x_act, Hmotion[mid_alt, act_pos], label='Motion blur MTF', linewidth=1.2)
        plt.plot(x_act, Hsys[mid_alt, act_pos], label='System MTF', color='black', linewidth=2.0)
        plt.axvline(x=0.5, color='black', linestyle='--', linewidth=1.5, label='f Nyquist')

        plt.title(f"System MTF - slice ACT ({band})")
        plt.xlabel("Spatial frequencies f/(1/w) [-]")
        plt.ylabel("MTF")
        plt.xlim(0.0, 0.51)
        plt.ylim(0.0, 1.05)
        plt.grid(True)
        plt.legend(loc='lower left', fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(directory, f"system_mtf_act_{band}.png"), dpi=300)
        plt.close()

        # ----------------------------------------------------------------------
        # 2. Slice ALT (Figure 7-52)
        # ----------------------------------------------------------------------
        alt_pos = fnAlt >= 0.0
        x_alt = fnAlt[alt_pos]

        plt.figure(figsize=(9, 5.5))
        plt.plot(x_alt, Hdiff[alt_pos, mid_act], label='Diffraction MTF', linewidth=1.2)
        plt.plot(x_alt, Hdefoc[alt_pos, mid_act], label='Defocus MTF', linewidth=1.2)
        plt.plot(x_alt, Hwfe[alt_pos, mid_act], label='WFE Aberrations MTF', linewidth=1.2)
        plt.plot(x_alt, Hdet[alt_pos, mid_act], label='Detector MTF', linewidth=1.2)
        plt.plot(x_alt, Hsmear[alt_pos, mid_act], label='Smearing MTF', linewidth=1.2)
        plt.plot(x_alt, Hmotion[alt_pos, mid_act], label='Motion blur MTF', linewidth=1.2)
        plt.plot(x_alt, Hsys[alt_pos, mid_act], label='System MTF', color='black', linewidth=2.0)
        plt.axvline(x=0.5, color='black', linestyle='--', linewidth=1.5, label='f Nyquist')

        plt.title(f"System MTF - slice ALT ({band})")
        plt.xlabel("Spatial frequencies f/(1/w) [-]")
        plt.ylabel("MTF")
        plt.xlim(0.0, 0.51)
        plt.ylim(0.0, 1.05)
        plt.grid(True)
        plt.legend(loc='lower left', fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(directory, f"system_mtf_alt_{band}.png"), dpi=300)
        plt.close()

        # ----------------------------------------------------------------------
        # 3. 2D Map of System MTF (Figure 7-53)
        # ----------------------------------------------------------------------
        plt.figure(figsize=(8, 6))

        # Hsys ya tiene el (0,0) en el centro, NO apliques fftshift aqui:
        plt.imshow(Hsys, cmap='jet', aspect='auto', origin='lower')

        plt.colorbar()
        plt.title(f"System MTF for {band}")
        plt.xlabel("ACT")
        plt.ylabel("ALT")
        plt.tight_layout()
        plt.savefig(os.path.join(directory, f"system_mtf_2d_{band}.png"), dpi=300)
        plt.close()

        self.logger.info(f"MTF plots successfully saved to: {directory}")




