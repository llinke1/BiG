# Script for converting HDF Catalog to numpy array
# This script requires nbodykit!

from nbodykit.source.catalog.file import HDFCatalog
import argparse
import numpy as np

# COMMANDLINE PARSING
parser = argparse.ArgumentParser(description='Measures 3D bispectrum.')

parser.add_argument("--i", help="Input filename [hdf file]")
parser.add_argument("--o", help="Output filename [numpy binary]")

parser.add_argument("--L", help='box side length [Mpc/h]', type=float)
parser.add_argument("--Nmesh", help='Number of grid cells along one dimension', type=int)

parser.add_argument("--field", help="Name of column in hdf with right particles, e.g. GrO")
parser.add_argument("--ex", help="Fields of hdf file that need to be excluded", type=str, nargs='*')


args = parser.parse_args()

if not all(vars(args).values()):
    parser.error("Not the right number of command line parameters! All are required!")

print(f"Excluding these fields from HDF file: {args.ex}")


f=HDFCatalog(args.i, exclude=args.ex)

mesh=f.to_mesh(Nmesh=args.Nmesh, BoxSize=args.L, position=args.field)

a=mesh.preview()

np.save(args.o, a)
