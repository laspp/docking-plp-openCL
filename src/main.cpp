#include <string>
#include <iostream>
#include <omp.h>
#include "host/Batch.hpp"
#include "host/dbgl.hpp"
#include "host/WorkerCL.hpp"
#include "host/Data.hpp"

int main(int argc, char* argv[]) {

    double programStartTime = omp_get_wtime();

    Batch batch;
    parseBatch(argc, argv, batch);

    // WorkerCL scope
    {
        Data data(batch.inputPath + "/" + batch.jobs.at(0), batch);
        data.parameters.ncycles *= 10;// REMOVE THIS
        WorkerCL workerCL(data, batch);

        workerCL.initMemory(data, batch);
        workerCL.kernelCreation(data, batch);
        workerCL.kernelSetArgs(data, batch);

        workerCL.initialStep(data, batch);

        dbg("0/" + std::to_string(data.parameters.ncycles));
        for(int i=0; i < data.parameters.ncycles; i++) {
            
            workerCL.runStep(data, batch);

            if(i % 50 == 0) {
                dbg("\r" + std::to_string(i) + "/" + std::to_string(data.parameters.ncycles));
                std::cout << std::flush;
            }
        }

        workerCL.finalize(data, batch);
        
	dbg("\r" + std::to_string(data.parameters.ncycles) + "/" + std::to_string(data.parameters.ncycles));
	std::cout << std::endl;
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
    dbgl("Program run time: " + std::to_string(omp_get_wtime() - programStartTime) + "s");
   
   return 0;
}
