@echo off

rem Version of MS Visual Studio
set VS_VER=2019

echo ======================================
echo Checking files and folders ...

if exist .\src\kernels\tyche_i.cl (
  echo   File ./src/kernels/tyche_i.cl found.
) else (
    echo   File ./src/kernels/tyche_i.cl NOT found, downloading file ...
    rem curl is pre-installed on Windows 10 and later
    curl -o ./src/kernels/tyche_i.cl https://raw.githubusercontent.com/bstatcomp/RandomCL/master/generators/tyche_i.cl
)

if exist output\ (
  echo   Folder "output" found.
) else (
  echo   Folder "output" NOT found, creating ...
  mkdir output
)
echo ======================================
echo Building solution ...

rem Set environment
echo --------------------------------------------------------
echo Setting MS VS environment
echo --------------------------------------------------------
call "C:\Program Files (x86)\Microsoft Visual Studio\%VS_VER%\Community\VC\Auxiliary\Build\vcvarsall.bat" x86_amd64

cmake -B .\build -G Ninja
ninja -C .\build

rem Run program num_runs times
rem Number of command line arguments
set argC=0
for %%x in (%*) do Set /A argC+=1
if %argC% == 0 (
  set /A num_runs=1
) else (
  set /A num_runs=%1
)

echo ======================================
echo Running tests (num_runs: %num_runs%)

for /L %%y in (1, 1, %num_runs%) do (
  echo --------------------------------------
  echo Run %%y of %num_runs%
  echo --------------------------------------
  .\build\CmDockOpenCL batches\run.json
  echo.
  echo.
)

