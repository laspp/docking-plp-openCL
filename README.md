# docking-plp-openCL
OpenCL implementation of a docking algorithm based on plp fitness function.

## Compile and run:

### Arnes HPC (V100):

hpc-login.arnes.si

@hpc-login1 (not login2):

salloc -G1 --partition=gpu -n1

ssh wnXYZ

@wnXYZ:

module load CUDA

module load CMake

./build_and_run.sh make 1


### Windows:

PATH: ninja, cmake, git

windows_build_w_ninja_run.bat
