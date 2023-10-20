from nbodykit.source.catalog.file import HDFCatalog
import argparse

# COMMANDLINE PARSING
parser = argparse.ArgumentParser(description='Measures 3D bispectrum.')

parser.add_argument("--i", help="Input filename [hdf file]")
parser.add_argument("--o", help="Output filename [bigfile]")

parser.add_argument("--L", help='box side length [Mpc/h]', type=float)
parser.add_argument("--Nmesh", help='Number of grid cells along one dimension', type=int)

parser.add_argument("--field", help="Name of column in hdf with right particles, e.g. GrO")

args = parser.parse_args()

if not all(vars(args).values()):
    parser.error("Not the right number of command line parameters! All are required!")





f=HDFCatalog(args.i)
mesh=f.to_mesh(Nmesh=args.Nmesh, BoxSize=args.L, position=args.field)

mesh.save(args.o, dataset='Field', mode='real')
