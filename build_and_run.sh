#!/bin/bash
# Usage: build_and_run.sh [ninja | make] [num_runs] [num_cycles]
# Example: 
# build_and_run.sh ninja 2
# will build the solution with ninja and run test 2 times

# Input arguments check
builder="ninja"  # ninja or make
num_runs=1
num_cycles=0

if [ "$#" -ge 1 ] 
then
    builder=$1
fi

if [ "$#" -ge 2 ] 
then
    num_runs=$2
fi

if [ "$#" -eq 3 ] 
then
    num_cycles=$3
fi

if [ "$#" -gt 3 ] 
then
    echo "Usage: $0 [ninja | make] [num_runs] [num_cycles]"
    echo "Defaults: builder=ninja, num_runs=1 and num_cycles=0"
    echo "Example: "
    echo "  build_and_run.sh ninja 2 1000"
    echo "will build the solution with ninja and run test 2 times with 1000 iterations of genetic algorithm"
    exit 1
fi

req_file="./src/kernels/tyche_i.cl"
req_folder="./output"

echo ======================================
echo Checking files and folders ...

if [[ -f $req_file ]];
then
    echo "  File $req_file is found."
else
    echo "  File $req_file is NOT FOUND, downloading file ..."
    wget --directory-prefix=./src/kernels/ https://raw.githubusercontent.com/bstatcomp/RandomCL/master/generators/tyche_i.cl
fi

if [[ -d $req_folder ]];
then
    echo "  Folder $req_folder exists."
else
	echo "  Folder $req_folder DOES NOT exist, creating ..."
    mkdir $req_folder
fi

echo ======================================
echo Building solution ...

if [ "$builder" = "ninja" ]; then
    cmake -B ./build -G Ninja
    ninja -C build
fi

if [ "$builder" = "make" ]; then
    cmake -B ./build
    make -C build
fi

if [ "$?" -eq "0" ]
then
    echo ======================================
    echo "Running tests (num_runs: $num_runs)"
    echo ======================================
    for FILE in batches/*.json
    do 
        for (( i=1; i<=$num_runs; i++ )) 
        do 
            ./build/CmDockOpenCL $FILE $num_cycles
        done
        echo
    done
fi
