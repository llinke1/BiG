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
parser.add_argument("--Nkbins", help='Number of k bins, will be binned linearly or log depending on kbinmode', type=int)
parser.add_argument("--kmin", help='Minimal k [Mpc/h]', type=float)
parser.add_argument("--kmax", help='Maximal k [Mpc/h]', type=float)
parser.add_argument("--kbinmode", help='How to bin the ks, can be lin or log', default='lin')
parser.add_argument("--mode", help='Which k-triangles to calculate. Can be all or equilateral')
parser.add_argument("--outfn", help='Prefix for output files')
parser.add_argument("--infiles", help='File with names of density files')
parser.add_argument("--verbose", help='Verbosity', action='store_true')
parser.add_argument("--doTiming", help='Whether to time the measurements', action='store_true')
parser.add_argument("--filetype", help="Type of density file. Must be numpy.", default='numpy')
parser.add_argument("--effectiveTriangles", help="Whether to calculate and output the effective k-triangles", action='store_true')

args = parser.parse_args()

if not all(vars(args).values()):
    parser.error("Not the right number of command line parameters! All are required!")

L=args.L
Nmesh=args.Nmesh

Nkbins=args.Nkbins
kmin=args.kmin
kmax=args.kmax

mode=args.mode
if (mode != "equilateral") & (mode != "all"):
    parser.error("Mode needs to be 'equilateral' or 'all'")

outfn=args.outfn
infiles=args.infiles

if args.verbose:
    print("Finished reading CMD line arguments")

# K BINS SETTING
if args.kbinmode=='lin':
    kbins=np.linspace(kmin, kmax, Nkbins+1)
elif args.kbinmode=='log':
    kbins=np.geomspace(kmin, kmax, Nkbins+1)
else:
    raise ValueError(f"kbinmode cannot be {args.kbinmode}, has to be either 'lin' or 'log'")


kbins_lower=kbins[:-1]
kbins_upper=kbins[1:]
kbins_mid=0.5*(kbins_lower+kbins_upper)

kbinedges=[kbins_lower, kbins_upper, kbins_mid]

if args.verbose:
    print("Settings:")
    print(f"Boxsize: {L} Mpc/h")
    print(f"Grid Cells (1D): {Nmesh}")
    print(f"ks: {kbins_lower}")
    print(f"Using {args.kbinmode} binning")
    print(f"Triangles: {mode}")
    print(f"Reading density files from {infiles}")
    print(f"Writing output to {outfn}")
    if args.effectiveTriangles:
        print("Calculating and outputting effective triangles")

# READ IN DENSITY FILES
file=open(infiles, 'r')
filenames=file.readlines()

# INITIALIZATION EXTRACTOR

Xtract=BiG.bispectrumExtractor(L, Nmesh, kbinedges, args.verbose)



if args.verbose:
    print("Finished initialization BispectrumExtractor")

if args.doTiming:
    time1=time.time()

# NORM CALCULATION

norm=Xtract.calculateBispectrumNormalization_slow(mode=mode)

if args.doTiming:
    time2=time.time()

if args.verbose:
    print("Finished calculating bispectrum norm")
    if args.doTiming:
        print(f"Needed {time2-time1} seconds to run")
        time1=time2

# EFFECTIVE TRIANGLE CALCULATION
if args.effectiveTriangles:
    effTriangles=Xtract.calculateEffectiveTriangle_slow(mode=mode)

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
    
    bispec=Xtract.calculateBispectrum_slow(f.strip(), mode=mode, filetype=args.filetype)


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
        if mode=='equilateral':
            for i in range(Nkbins):
                if args.effectiveTriangles:
                    print(kbins_mid[i], kbins_mid[i], kbins_mid[i], effTriangles[i][0]/norm[i], effTriangles[i][0]/norm[i], effTriangles[i][0]/norm[i], bispec[i], norm[i], bispec[i]/norm[i]*Xtract.prefactor, file=o)
                else:
                    print(kbins_mid[i], kbins_mid[i], kbins_mid[i], bispec[i], norm[i], bispec[i]/norm[i]*Xtract.prefactor, file=o)

        elif mode=='all':
            ix=0
            for i in range(Nkbins):
                for j in range(i, Nkbins):
                    for k in range(j, Nkbins):
                        if kbins_mid[k]<=kbins_mid[i]+kbins_mid[j]:
                            if args.effectiveTriangles:
                                print(kbins_mid[i], kbins_mid[j], kbins_mid[k], effTriangles[ix][0]/norm[ix], effTriangles[ix][1]/norm[ix], effTriangles[ix][2]/norm[ix],  bispec[ix], norm[ix], bispec[ix]/norm[ix]*Xtract.prefactor, file=o)
                            else:
                                print(kbins_mid[i], kbins_mid[j], kbins_mid[k], bispec[ix], norm[ix], bispec[ix]/norm[ix]*Xtract.prefactor, file=o)
                            ix+=1
        else:
            raise ValueError(f"Mode cannot be {mode}, has to be either 'all' or 'equilateral'")
    if args.verbose:
        print(f"Written output to {outfn_now}")