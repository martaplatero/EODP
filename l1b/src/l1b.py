
# LEVEL-1B MODULE

from l1b.src.initL1b import initL1b
from common.io.writeToa import writeToa, readToa
from common.src.auxFunc import getIndexBand
from common.io.readFactor import readFactor, EQ_MULT, EQ_ADD, NC_EXT
import numpy as np
import os
import matplotlib.pyplot as plt

class l1b(initL1b):

    def __init__(self, auxdir, indir, outdir):
        super().__init__(auxdir, indir, outdir)

    def processModule(self):

        self.logger.info("Start of the L1B Processing Module")

        for band in self.globalConfig.bands:

            self.logger.info("Start of BAND " + band)

            # Read TOA - output of the ISM in Digital Numbers
            # -------------------------------------------------------------------------------
            toa = readToa(self.indir, self.globalConfig.ism_toa + band + '.nc')

            # Equalization (radiometric correction)
            # -------------------------------------------------------------------------------
            if self.l1bConfig.do_equalization:
                self.logger.info("EODP-ALG-L1B-1010: Radiometric Correction (equalization)")

                # Read the multiplicative and additive factors from auxiliary/equalization/
                eq_mult = readFactor(os.path.join(self.auxdir,self.l1bConfig.eq_mult+band+NC_EXT),EQ_MULT)
                eq_add = readFactor(os.path.join(self.auxdir,self.l1bConfig.eq_add+band+NC_EXT),EQ_ADD)

                # Do the equalization and save to file
                toa = self.equalization(toa, eq_add, eq_mult)
                writeToa(self.outdir, self.globalConfig.l1b_toa_eq + band, toa)

            # Restitution (absolute radiometric gain)
            # -------------------------------------------------------------------------------
            self.logger.info("EODP-ALG-L1B-1020: Absolute radiometric gain application (restoration)")
            toa = self.restoration(toa, self.l1bConfig.gain[getIndexBand(band)])

            # Write output TOA
            # -------------------------------------------------------------------------------
            writeToa(self.outdir, self.globalConfig.l1b_toa + band, toa)
            self.plotL1bToa(toa, self.outdir, band)

            self.logger.info("End of BAND " + band)

        self.logger.info("End of the L1B Module!")


    def equalization(self, toa, eq_add, eq_mult):
        """
        Equlization. Apply an offset and a gain.
        :param toa: TOA in DN
        :param eq_add: Offset in DN
        :param eq_mult: Gain factor, adimensional
        :return: TOA in DN, equalized
        """
        toa = (toa - eq_add) / eq_mult

        return toa

    def restoration(self,toa,gain):
        """
        Absolute Radiometric Gain - restore back to radiances
        :param toa: TOA in DN
        :param gain: gain in [rad/DN]
        :return: TOA in radiances [mW/sr/m2]
        """
        toa = toa * gain

        self.logger.debug('Sanity check. TOA in radiances after gain application ' + str(toa[1,-1]) + ' [mW/m2/sr]')

        return toa

    def plotL1bToa(self, toa_l1b, outputdir, band):
        #TODO
        a=1 # dummy
        # Central ALT line cut to avoid edge effects
        mid_alt = toa_l1b.shape[0] // 2
        profile = toa_l1b[mid_alt, :]
        act_pixels = range(len(profile))

        fig, ax = plt.subplots(figsize=(10, 5.5))

        # Red curve with 'TOA L1B' label
        ax.plot(act_pixels, profile, color='#d62728', linewidth=1.5, label='TOA L1B')

        # Titles and axis formatting
        ax.set_title(f"Equalization and restoration for {band}", fontsize=13)
        ax.set_xlabel("ACT pixel [-]", fontsize=11)
        ax.set_ylabel("TOA [mW/m2/sr]", fontsize=11)

        # Grid and legend in upper right
        ax.grid(True, which='both', linestyle='-', linewidth=0.5, color='gray', alpha=0.6)
        ax.legend(loc='upper right', fontsize=8, frameon=True)

        fig_path = os.path.join(outputdir, f"l1b_toa_{band}.png")
        plt.tight_layout()
        plt.savefig(fig_path, dpi=300)
        plt.close(fig)

        self.logger.info(f"Figure saved: {fig_path}")
