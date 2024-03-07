
<h3 align="center">Bispectrum estimator on GPUs</h3>

<p align="center">
    Code for measuring 3-D bispectra from density maps using GPUs.
</p>

<!-- ABOUT THE PROJECT -->
## About The Project

This code can measure the 3D-bispectra from density maps using the FFT-based measurement algorithm presented for example in <a href="https://arxiv.org/abs/1904.11055"> Tomlinson+ (2019)</a>.  Our implementation is similar to <a href="https://github.com/sjforeman/bskit/blob/master/README.md"> bskit </a>, but our code uses <a href="https://jax.readthedocs.io/en/latest/notebooks/quickstart.html">JAX </a> for GPU acceleration of the FFTs.

<!-- GETTING STARTED -->
## Getting Started

### Prerequisites
* **python3**: This code will not work for python2! 
* **jax**: Check <a href="https://github.com/google/jax#installation"> here </a> for how to install it

### Installation

To install this code, clone the gitrepo, go to the root folder and execute
```
pip install .
```

The files for the examples in the folder `tests` are accessible <a href=https://fileshare.uibk.ac.at/d/6e64b19298ac4e9290f3/> here </a>.

<!-- USAGE EXAMPLES -->
## Usage

### Input
#### Density Maps
The density should be provided as regularly gridded in a three-dimensional numpy binary file. Examples are given here:  <a href=https://fileshare.uibk.ac.at/d/6e64b19298ac4e9290f3/> here </a>.

#### List of density maps
The bispectrum extractor reads the file names of density maps from an input file. An example is given in `tests/testRun_input.dat`

### Measure bispectrum
The bispectrum is measured with `python scripts/runBispectrumExtractor.py`. The command line arguments are
* **L**: box side length [Mpc/h]. If you are using a folded box, this needs to be L/2^k where k is the number of folds!
* **Nmesh**: Number of grid cells along one dimension
* **Nkbins**: Number of $k$ bins
* **kmin**: Minimal $k$ [h/Mpc]
* **kmax**: Maximal $k$ [h/Mpc]
* **kbinmode**: How to bin the ks, can be lin or log
* **mode**: Either `equilateral` (calculates only equilateral $k$-triangles) or `all` (calculates all triangle configurations)
* **outfn**: Outputfileprefix
* **infiles**: ASCII file with filenames of density files
* **verbose**: Verbosity
* **doTiming**: Whether to time the measurements
* **filetype**: Type of density file. Currently only supports `numpy`
* **effectiveTriangles**: Whether to calculate and output the effective $k$-triangles per bin.

An example run would be
```
python ../scripts/runBispectrumExtractor.py --L 1000 --Nmesh 512 --Nkbins 19 --kmin 0.01 --kmax 1.91 --kbinmode lin --mode equilateral --outfn ../tests/testRun_output --infiles ../tests/testRun_input.dat --verbose True --doTiming True --filetype numpy --effectiveTriangles False
```

### Output
For each input file an outputfile is generated. These files are ASCII and the columns are
1. $k_1$ [$h$/Mpc]
2. $k_2$ [$h$/Mpc]
3. $k_3$ [$h$/Mpc]
4. Unnormalized Bispectrum
5. Normalization of Bispectrum
6. Bispectrum $B(k_1, k_2, k_3)$

If the `effectiveTriangles` switch is set to `True`, the output will include the effective triangles, so the columns are
1. $k_1$ [$h$/Mpc]
2. $k_2$ [$h$/Mpc]
3. $k_3$ [$h$/Mpc]
4. $k_{eff, 1}$ [$h$/Mpc]
5. $k_{eff, 2}$ [$h$/Mpc]
6. $k_{eff, 3}$ [$h$/Mpc]
7. Unnormalized Bispectrum
8. Normalization of Bispectrum
9. Bispectrum $B(k_1, k_2, k_3)$