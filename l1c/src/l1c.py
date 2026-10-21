
# LEVEL-1C

from l1c.src.initL1c import initL1c
from common.io.writeToa import writeToa, readToa
from common.io.readGeodetic import readGeodetic, getCorners
import mgrs
import sys
import numpy as np
from scipy.interpolate import bisplrep, bisplev
import matplotlib.pyplot as plt
from common.io.l1cProduct import writeL1c
from matplotlib import cm

class l1c(initL1c):

    def __init__(self, auxdir, indir, outdir):
        super().__init__(auxdir, indir, outdir)

    def processModule(self):

        self.logger.info("Start of the L1C Processing Module")

        for band in self.globalConfig.bands:

            self.logger.info("Start of BAND " + band)

            # Read TOA - output of the L1B in Radiances
            # -------------------------------------------------------------------------------
            toa = readToa(self.l1bdir, self.globalConfig.l1b_toa + band + '.nc')
            lat,lon = readGeodetic(self.gmdir, self.globalConfig.gm_geoloc)
            self.checkSize(lat,toa)

            # L1C reprojection onto the MGRS grid
            # -------------------------------------------------------------------------------
            lat_l1c, lon_l1c, toa_l1c = self.l1cProjtoa(lat, lon, toa, band)

            # Write output TOA
            # -------------------------------------------------------------------------------
            writeL1c(self.outdir, self.globalConfig.l1c_toa + band, lat_l1c, lon_l1c, toa_l1c)
            if self.l1cConfig.plotL1cGrid:
                self.plotL1cToa(lat, lon, toa, lat_l1c, lon_l1c, toa_l1c, band)

            self.logger.info("End of BAND " + band)

        self.logger.info("End of the L1C Module!")


    def l1cProjtoa(self, lat, lon, toa, band):
        '''
        This function reprojects the L1B radiances into the MGRS grid.

        The MGRS reference system
        https://www.bluemarblegeo.com/knowledgebase/calculator-2020/Military_Grid_Reference_System_(MGRS).htm
        MGRS: '31REQ4367374067'
        31 is the UTM zone, R is the UTM latitude band; EQ are the MGRS column and row band letters
        43673 is the MGRS Easting (5 dig); 74067 is the MGRS Northing (5dig)

        Python mgrs library documentation
        https://pypi.org/project/mgrs/

        :param lat: L1B latitudes [deg]
        :param lon: L1B longitudes [deg]
        :param toa: L1B radiances
        :param band: band
        :return: L1C radiances, L1C latitude and longitude in degrees
        '''
        # Interpolant relating the L1B geodetic coordinates to the TOA radiances
        tck = bisplrep(lat, lon, toa)

        # MGRS tiles covered by the L1B image (a set keeps only unique tiles)
        m = mgrs.MGRS()
        mgrs_tiles = set([])

        for ii in range(toa.shape[0]):
            for jj in range(toa.shape[1]):
                mgrstile = str(m.toMGRS(lat[ii, jj], lon[ii, jj], inDegrees=True, MGRSPrecision=self.l1cConfig.mgrs_tile_precision))
                mgrs_tiles.add(mgrstile)

        # Sorted list so that the output order is reproducible between executions
        mgrs_tiles = sorted(mgrs_tiles)

        # Initialize vairiables
        toa_l1c = np.zeros(len(mgrs_tiles))
        lat_l1c = np.zeros(len(mgrs_tiles))
        lon_l1c = np.zeros(len(mgrs_tiles))

        # Geodetic coordinates of each MGRS tile and interpolated radiance
        for i in range(len(mgrs_tiles)):
            lat_l1c[i], lon_l1c[i] = m.toLatLon(mgrs_tiles[i], inDegrees=True)
            toa_l1c[i] = bisplev(lat_l1c[i], lon_l1c[i], tck)

        return lat_l1c, lon_l1c, toa_l1c

    def checkSize(self, lat,toa):
        '''
        Check the sizes of the input radiances and geodetic coordinates.
        If they don't match, exit.
        :param lat: Latitude 2D matrix
        :param toa: Radiance 2D matrix
        :return: NA
        '''
        if lat.shape != toa.shape:
            self.logger.error("Size mismatch between the geodetic coordinates " + str(lat.shape) +
                              " and the L1B TOA " + str(toa.shape) + ". Exiting.")
            sys.exit(1)

        self.logger.debug("Sizes of the geodetic coordinates and L1B TOA match: " + str(toa.shape))

    def plotL1cToa(self, lat, lon, toa, lat_l1c, lon_l1c, toa_l1c, band):
        '''
        Plot the L1B footprint (irregular grid) and the L1C MGRS points (regular grid)
        :param lat, lon, toa: L1B latitude, longitude and radiances (2D)
        :param lat_l1c, lon_l1c, toa_l1c: L1C latitude, longitude and radiances (1D)
        :param band: band
        :return: NA
        '''
        fig = plt.figure(figsize=(20, 10))

        # L1B footprint: the four corners of the image, closed
        plt.plot(getCorners(lon), getCorners(lat), 'k-', linewidth=2, label='L1B')

        # L1C MGRS grid points
        plt.plot(lon_l1c, lat_l1c, 'r.', markersize=4, label='L1C MGRS')

        plt.title('Projection on ground', fontsize=20)
        plt.xlabel('Longitude [deg]', fontsize=16)
        plt.ylabel('Latitude [deg]', fontsize=16)
        plt.grid(True)
        plt.legend(loc='upper right')
        plt.axis('equal')

        savestr = self.outdir + 'footprint_' + band + '.png'
        plt.savefig(savestr, dpi=100)
        plt.close(fig)
        self.logger.info('Saved figure ' + savestr)