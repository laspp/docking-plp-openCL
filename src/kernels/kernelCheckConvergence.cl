#include <clStructs.h>
#include <constants.cl>

__kernel void kernelCheckConvergence(constant parametersForGPU* parameters, global uint* convergenceAchieved, global uint* iConvergence, global float* oldBestScore, global float* bestScore, local uint* localTmp) {

    int localID=get_local_id(ID_1D);
    int localSize=get_local_size(ID_1D);

    localTmp[localID]=0;
    /*if(localID==0){
    //printf("CHECK CONVERGENCE\n");
    for (int i=0;i<parameters->nruns;i++)
            printf("%3.2f %3.2f (%d)\t", bestScore[i], oldBestScore[i], iConvergence[i]);
            //printf("%d\t", iConvergence[i]);
        printf("\n"); 
    }*/
    barrier(CLK_LOCAL_MEM_FENCE);
    for(int run=localID; run < parameters->nruns; run+=localSize){
        if (bestScore[run] < oldBestScore[run]){            
            iConvergence[run]=0;
            oldBestScore[run]=bestScore[run];      
        }else{
            iConvergence[run]=min((unsigned int)iConvergence[run]+1,(unsigned int)parameters->nconvergence);
        }
        
    }
    barrier(CLK_GLOBAL_MEM_FENCE);
    unsigned int sum = 0;
    int run = localID;
    while( run < parameters->nruns )
	{
		sum += iConvergence[run];
		run += localSize;
	}
	localTmp[localID] = sum;
    int floorPow2 = exp2(log2((float)localSize));
    if (localSize != floorPow2)										
	{
		if ( localID >= floorPow2 )
            localTmp[localID - floorPow2] += localTmp[localID];
		barrier(CLK_LOCAL_MEM_FENCE);
    }

	for(int i = (floorPow2>>1); i>0; i >>= 1) 
	{
		if(localID < i) 
			localTmp[localID] += localTmp[localID + i];
		barrier(CLK_LOCAL_MEM_FENCE);
	}
    barrier(CLK_LOCAL_MEM_FENCE);
	if(localID == 0){
        convergenceAchieved[0]=(localTmp[0]==parameters->nruns*parameters->nconvergence)?1:0;
        /*printf("Converged: %d/%d | %d %d\n",localTmp[0],parameters->nruns*parameters->nconvergence,parameters->nruns,parameters->nconvergence);
        printf("END\n");
        for (int i=0;i<parameters->nruns;i++)
            printf("%d", iConvergence[i]);
        printf("\n"); */
    } 
}

