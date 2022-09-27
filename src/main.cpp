#include <string>
#include <iostream>
#include <sstream>
#include <omp.h>
#include "host/Batch.hpp"
#include "host/dbgl.hpp"
#include "host/WorkerCL.hpp"
#include "host/Data.hpp"

int main(int argc, char* argv[]) {


    int cycle_limit=0;
    if (argc >= 3) {
        //Read cycle limit (number of iterations from command line arguments)
        std::istringstream iss( argv[2] );
        int val;
        if (iss >> val)
        {
            cycle_limit=val;
        }

    }
    double programStartTime = omp_get_wtime();
    double loopStartTime, loopEndTime;
    Batch batch;
    parseBatch(argc, argv, batch);
    // WorkerCL scope
    for(auto  & job : batch.jobs)
    {
        std::string fileName=job;
        Data data(batch.inputPath + "/" + fileName, batch);
        if (cycle_limit>0){
            data.parameters.ncycles = cycle_limit;
        }
        else{
            //default cycle limit
            data.parameters.ncycles=10000;
        }
        //Testing nconvergence parameter, config is read from binary
        //data.parameters.nconvergence=30;
        WorkerCL workerCL(data, batch);
        workerCL.kernelCreation(data, batch);
        loopStartTime = omp_get_wtime(); 
        workerCL.initMemory(data, batch);
        workerCL.kernelSetArgs(data, batch);
        workerCL.initialStep(data, batch);
        //dbg("0/" + std::to_string(data.parameters.ncycles));
        int cyclesDone=0;
        for(int i=0; i < data.parameters.ncycles; i++) {
            
            workerCL.runStep(data, batch);

            cyclesDone++;
            /*if(i % 50 == 0) {
                dbg("\r" + std::to_string(i) + "/" + std::to_string(data.parameters.ncycles));
                std::cout << std::flush;
            }*/
            if(cycle_limit==0 && data.convergenceFlag) break;
        }

        workerCL.finalize(data, batch);
        loopEndTime = omp_get_wtime();
        std::cout << fileName + "\t" + std::to_string(cyclesDone) + "\t" + std::to_string(data.parameters.nruns) + "\t" + std::to_string(loopEndTime - loopStartTime) << std::endl;    
	//dbg("\r" + std::to_string(cyclesDone) + "/" + std::to_string(data.parameters.ncycles));
	//std::cout << std::endl;
    //dbgl("Per step per run time: " + std::to_string((loopEndTime - loopStartTime)/(cyclesDone*data.parameters.nruns)) + "s");
    }
    
    /*
    for receptor in receptors
    {

        initReceptor(receptor);

        for ligand in ligands

            initLigand(ligand);

            doWork();
    }
    */
    //dbgl("Loop run time: " + std::to_string((loopEndTime - loopStartTime)) + "s");
    //dbgl("Program run time: " + std::to_string((omp_get_wtime() - programStartTime)) + "s");
    
   
   return 0;
}
