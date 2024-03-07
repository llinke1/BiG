import numpy as np
from BiG import bispectrumExtractor as BiG
import argparse
import os
from pathlib import Path
import time

os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform" # This is needed so that GPU variables are freed if no longer needed


# COMMANDLINE PARSING
parser = argparse.ArgumentParser(description='Measures 3D bispectrum.')

parser.add_argument("--L", help='box side length [Mpc/h]. If you are using a folded box, this needs to be L/(2^k) where k is the number of folds!', type=float)
parser.add_argument("--Nmesh", help='Number of grid cells along one dimension', type=int)
parser.add_argument("--kbinFile", help='File containing k-bins')
parser.add_argument("--outfn", help='Prefix for output files')
parser.add_argument("--infiles", help='File with names of density files')
parser.add_argument("--verbose", help='Verbosity', action='store_true')
parser.add_argument("--doTiming", help='Whether to time the measurements', action='store_true')
parser.add_argument("--filetype", help="Type of density file. Must be numpy.", default='numpy')
parser.add_argument("--effectiveTriangles", help="Whether to calculate and output the effective k-triangles", action='store_true')

args = parser.parse_args()

# if not all(vars(args).values()):
#     parser.error("Not the right number of command line parameters! All are required!")

L=args.L
Nmesh=args.Nmesh



outfn=args.outfn
infiles=args.infiles

if args.verbose:
    print("Finished reading CMD line arguments")

# K BINS SETTING

kbins=np.loadtxt(args.kbinFile)
kbinedges_low=kbins[:,0:3]
kbinedges_cen=kbins[:,3:6]
kbinedges_hig=kbins[:,6:9]


if args.verbose:
    print("Settings:")
    print(f"Boxsize: {L} Mpc/h")
    print(f"Grid Cells (1D): {Nmesh}")
    print(f"Using ks from {args.kbinFile}")
    print(f"Reading density files from {infiles}")
    print(f"Writing output to {outfn}")
    if args.effectiveTriangles:
        print("Calculating and outputting effective triangles")

# READ IN DENSITY FILES
file=open(infiles, 'r')
filenames=file.readlines()

# INITIALIZATION EXTRACTOR

Xtract=BiG.bispectrumExtractor(L, Nmesh, [], args.verbose)



if args.verbose:
    print("Finished initialization BispectrumExtractor")

if args.doTiming:
    time1=time.time()

# NORM CALCULATION

norm=Xtract.calculateBispectrumNormalization_slow(mode='custom', custom_kbinedges_low=kbinedges_low, custom_kbinedges_high=kbinedges_hig)

if args.doTiming:
    time2=time.time()

if args.verbose:
    print("Finished calculating bispectrum norm")
    if args.doTiming:
        print(f"Needed {time2-time1} seconds to run")
        time1=time2

# EFFECTIVE TRIANGLE CALCULATION
if args.effectiveTriangles:
    effTriangles=Xtract.calculateEffectiveTriangle_slow(mode='custom', custom_kbinedges_low=kbinedges_low, custom_kbinedges_high=kbinedges_hig)

    if args.doTiming:
        time2=time.time()
    
    if args.verbose:
        print("Finished calculating effective triangles")
        if args.doTiming:
            print(f"Needed {time2-time1} seconds to run")
            time1=time2

# BISPEC CALCULATION AND OUTPUT
for f in filenames:
    if args.verbose:
        print(f"Calculating bispectrum for {f}")
    
    bispec=Xtract.calculateBispectrum_slow(f.strip(), mode='custom', custom_kbinedges_low=kbinedges_low, custom_kbinedges_high=kbinedges_hig, filetype=args.filetype)


    if args.doTiming:
        time2=time.time()


    if args.verbose:
        print(f"Finished bispectrum calculation")
        if args.doTiming:
            print(f"Needed {time2-time1} seconds to run")
            time1=time2

    outfn_now=outfn+Path(f.strip()).stem+".dat"

    with open(outfn_now, "w") as o:
        if args.effectiveTriangles:
            print("# k1 [h/Mpc] k2 [h/Mpc] k3 [h/Mpc] k1_eff [h/Mpc] k2_eff [h/Mpc] k3_eff [h/Mpc] unnorm.Bispec norm norm.Bispec", file=o)
        else:
            print("# k1 [h/Mpc] k2 [h/Mpc] k3 [h/Mpc] unnorm.Bispec norm norm.Bispec", file=o)

        for i in range(len(kbinedges_cen)):
            if args.effectiveTriangles:
                print(kbinedges_cen[i][0], kbinedges_cen[i][1], kbinedges_cen[i][2], effTriangles[i][0]/norm[i], effTriangles[i][1]/norm[i], effTriangles[i][2]/norm[i], bispec[i], norm[i], bispec[i]/norm[i]*Xtract.prefactor, file=o)
            else:
                print(kbinedges_cen[i][0], kbinedges_cen[i][1], kbinedges_cen[i][2], bispec[i], norm[i], bispec[i]/norm[i]*Xtract.prefactor, file=o)

    if args.verbose:
        print(f"Written output to {outfn_now}")