#!/bin/bash

# testing using convergence checking
# (for fixed num. of steps, add num. of steps as cmd. arg. at L41 (example: ./build/CmDockOpenCL ./test/test.json 1000))
commit="806432e"

req_file="./src/kernels/tyche_i.cl"
req_folder="./test/temp"

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
    echo "  Folder $req_folder exists, deleting contents and creating new folder ..."
    rm -r -v $req_folder
    mkdir $req_folder
else
	echo "  Folder $req_folder DOES NOT exist, creating ..."
    mkdir $req_folder
fi

echo ======================================
echo Building solution ...

cmake -B ./build
make -C build

echo ======================================

if [ "$?" -eq "0" ];
then
    ./build/CmDockOpenCL ./test/test.json

    if [ "$?" -eq "0" ];
    then
        echo ======================================
        diff -ur ./test/$commit/ ./test/temp/
        if [ "$?" -eq "0" ];
        then
            echo -e "\033[32mPASSED\033[0m, no regression compared to commit:$commit"
        else
            echo -e "\033[31mFAILED\033[0m, there are regressions compared to commit:$commit"
        fi
    fi
fi
