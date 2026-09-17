
# MAIN FUNCTION TO CALL THE ISM MODULE

from ism.src.ism import ism

# Directory - this is the common directory for the execution of the E2E, all modules
auxdir = r'C:/Users/marta/Documents/GitHub/EODP/auxiliary'
indir = r"C:/Users/marta/Desktop/MISE/2 YEAR/Earth Observation Data Processing/EODP_TER_2021/EODP-TS-ISM/input/gradient_alt100_act150" # small scene
outdir = r"C:/Users/marta/Desktop/MISE/2 YEAR/Earth Observation Data Processing/EODP_TER_2021/EODP-TS-ISM/my_output"

# Initialise the ISM
myIsm = ism(auxdir, indir, outdir)
myIsm.processModule()
