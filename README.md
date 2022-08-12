# docking-plp-openCL

OpenCL implementation of a docking algorithm based on PLP fitness function [1].

## Compile and run

### Arnes HPC (V100)

hpc-login.arnes.si

@hpc-login1 (not login2):

salloc -G1 --partition=gpu -n1

ssh wnXYZ

@wnXYZ:

module load CUDA

module load CMake

./build_and_run.sh make 1 1000

### Windows

PATH: ninja, cmake, git

windows_build_w_ninja_run.bat

## References

**Empirical Scoring Functions for Advanced Protein−Ligand Docking with PLANTS**, Oliver Korb, Thomas Stützle, and Thomas E. Exner. Journal of Chemical Information and Modeling 2009 49 (1), 84-96, DOI: 10.1021/ci800298z.
