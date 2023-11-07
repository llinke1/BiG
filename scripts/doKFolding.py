# Script for k-folding an HDF Catalog and converting to a numpy array
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

parser.add_argument("--k", help="Number of foldings. k=1 leads to L/2, k=2 leads to L/4 and so on.", type=int)

parser.add_argument("--ex", help="Fields of hdf file that need to be excluded", type=str, nargs='*')
args = parser.parse_args()

if not all(vars(args).values()):
    parser.error("Not the right number of command line parameters! All are required!")

print(f"Excluding these fields from HDF file: {args.ex}")

f=HDFCatalog(args.i, exclude=args.ex)

array=np.array(f[args.field])

L=args.L
for i in range(args.k):
    x=array[:,0]
    y=array[:,1]
    z=array[:,2]

    x=x%(L/2)
    y=y%(L/2)
    z=z%(L/2)

    array=np.column_stack([x,y,z])
    L/=2

f[args.field+"_folded"]=array

mesh=f.to_mesh(Nmesh=args.Nmesh, BoxSize=L, position=args.field+"_folded")
a=mesh.preview()
np.save(args.o, a)
