import numpy as np
from BiG import bispectrumExtractor as BiG
import argparse
import os
from pathlib import Path

os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform" # This is needed so that GPU variables are freed if no longer needed

print("Warning: Powerspectrum is calculated unnormalized!")

# COMMANDLINE PARSING
parser = argparse.ArgumentParser(description='Measures 3D powerspectrum.')

parser.add_argument("--L", help='box side length [Mpc/h]', type=float)
parser.add_argument("--Nmesh", help='Number of grid cells along one dimension', type=int)
parser.add_argument("--Nkbins", help='Number of k bins, will be binned linearily', type=int)
parser.add_argument("--kmin", help='Minimal k [Mpc/h]', type=float)
parser.add_argument("--kmax", help='Maximal k [Mpc/h]', type=float)
parser.add_argument("--outfn", help='Prefix for output files')
parser.add_argument("--infiles", help='File with names of density files')
parser.add_argument("--verbose", help='Verbosity', type=bool, default=True)
parser.add_argument("--filetype", help="Type of density file. Must be numpy.", default='numpy')


args = parser.parse_args()

if not all(vars(args).values()):
    parser.error("Not the right number of command line parameters! All are required!")

L=args.L
Nmesh=args.Nmesh

Nkbins=args.Nkbins
kmin=args.kmin
kmax=args.kmax

outfn=args.outfn
infiles=args.infiles

if args.verbose:
    print("Finished reading CMD line arguments")

# K BINS SETTING

kbins=np.linspace(kmin, kmax, Nkbins+1)
kbins_lower=kbins[:-1]
kbins_upper=kbins[1:]
kbins_mid=0.5*(kbins_lower+kbins_upper)

kbinedges=[kbins_lower, kbins_upper, kbins_mid]

if args.verbose:
    print("Settings:")
    print(f"Boxsize: {L} Mpc/h")
    print(f"Grid Cells (1D): {Nmesh}")
    print(f"ks: {kbins_lower}")
    print(f"Reading density files from {infiles}")
    print(f"Writing output to {outfn}")

# READ IN DENSITY FILES
file=open(infiles, 'r')
filenames=file.readlines()

# INITIALIZATION EXTRACTOR

Xtract=BiG.bispectrumExtractor(L, Nmesh, kbinedges, args.verbose)

if args.verbose:
    print("Finished initialization BispectrumExtractor")


# POWERSPEC CALCULATION AND OUTPUT
for f in filenames:
    if args.verbose:
        print(f"Calculating powerspectrum for {f}")
    
    powerspec=Xtract.calculatePowerspectrum(f.strip(), filetype=args.filetype)
    if args.verbose:
        print(f"Finished powerspectrum calculation")

    outfn_now=outfn+Path(f.strip()).stem+".dat"

    with open(outfn_now, "w") as o:
        print("# k [h/Mpc] unnorm.Powerspec", file=o)
        for i in range(Nkbins):
            print(kbins_mid[i], powerspec[i],  file=o)
    if args.verbose:
        print(f"Written output to {outfn_now}")