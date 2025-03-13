import numpy as np
from BiG import bispectrumExtractor as BiG
from BiG import fileLoader as fl
import argparse
import os
from pathlib import Path

os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform" # This is needed so that GPU variables are freed if no longer needed

#print("Warning: Powerspectrum is calculated unnormalized!")

# COMMANDLINE PARSING
parser = argparse.ArgumentParser(description='Measures 3D powerspectrum.')

parser.add_argument("--L", help='box side length [Mpc/h]. If you are using a folded box, this needs to be L/(2^k) where k is the number of folds!', type=float)
parser.add_argument("--Nmesh", help='Number of grid cells along one dimension', type=int)
parser.add_argument("--kbinFile", help='File containing k-bins')
parser.add_argument("--outfn", help='Prefix for output files')
parser.add_argument("--infiles", help='File with names of density files')
parser.add_argument("--verbose", help='Verbosity', type=bool, default=True)
parser.add_argument("--filetype", help="Type of density file. Must be numpy.", default='numpy')


args = parser.parse_args()

# if not all(vars(args).values()):
#     parser.error("Not the right number of command line parameters! All are required!")

L=args.L
Nmesh=args.Nmesh

# Nkbins=args.Nkbins
# kmin=args.kmin
# kmax=args.kmax

outfn=args.outfn
infiles=args.infiles

if args.verbose:
    print("Finished reading CMD line arguments")

# # K BINS SETTING
# if args.kbinmode=='lin':
#     kbins=np.linspace(kmin, kmax, Nkbins+1)
# elif args.kbinmode=='log':
#     kbins=np.geomspace(kmin, kmax, Nkbins+1)
# else:
#     raise ValueError(f"kbinmode cannot be {args.kbinmode}, has to be either 'lin' or 'log'")

# K BINS SETTING

kbins=np.loadtxt(args.kbinFile)
kbinedges_low=kbins[:,0]
kbinedges_cen=kbins[:,1]
kbinedges_hig=kbins[:,2]

kbinedges=[kbinedges_low, kbinedges_hig, kbinedges_cen]

if args.verbose:
    print("Settings:")
    print(f"Boxsize: {L} Mpc/h")
    print(f"Grid Cells (1D): {Nmesh}")
    print(f"Using ks from {args.kbinFile}")
    print(f"Reading density files from {infiles}")
    print(f"Writing output to {outfn}")

# READ IN DENSITY FILES
file=open(infiles, 'r')
filenames=file.readlines()

# INITIALIZATION EXTRACTOR

Xtract=BiG.bispectrumExtractor(L, Nmesh, kbinedges, args.verbose)

prefactor=L**3/Nmesh**6

if args.verbose:
    print("Finished initialization BispectrumExtractor")

# NORM CALCULATION

norm=np.array(Xtract.calculatePowerspectrumNormalization(precision=np.float32))
norm/=prefactor

loader=fl.fileloader(filetype=args.filetype)

# POWERSPEC CALCULATION AND OUTPUT
for f in filenames:
    if args.verbose:
        print(f"Calculating powerspectrum for {f}")
    
    field_real=loader.load(f)
    powerspec=Xtract.calculatePowerspectrum(field_real)
    if args.verbose:
        print(f"Finished powerspectrum calculation")

    outfn_now=outfn+Path(f.strip()).stem+".dat"

    with open(outfn_now, "w") as o:
        print("# k [h/Mpc] unnorm.Powerspec norm norm.Powerspectrum", file=o)
        for i in range(len(kbinedges_cen)):

            print(kbinedges_cen[i], powerspec[i], norm[i], powerspec[i]/norm[i],  file=o)
    if args.verbose:
        print(f"Written output to {outfn_now}")