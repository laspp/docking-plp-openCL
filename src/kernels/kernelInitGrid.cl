#include <clStructs.h>
#include <constants.cl>

#include <grid.cl>

__kernel void kernelInitGrid(constant parametersForGPU* parameters,
                    global GridPointGPU* grid,
                    global AtomGPU* receptorAtoms,
                    global CoordGPU cavityGPU,
                    global int* numGoodReceptors,// one int
                    global int* receptorIndex) {

    uint classID=get_global_id(CLASS_ID_2D);
    uint xyzID=get_global_id(XYZ_ID_2D);

    if(xyzID < parameters->grid.N) {

        GridGPU ownGrid;

        ownGrid.gridStep[0] = parameters->grid.gridStep[0];
        ownGrid.gridStep[1] = parameters->grid.gridStep[1];
        ownGrid.gridStep[2] = parameters->grid.gridStep[2];

        ownGrid.gridMin[0] = parameters->grid.gridMin[0];
        ownGrid.gridMin[1] = parameters->grid.gridMin[1];
        ownGrid.gridMin[2] = parameters->grid.gridMin[2];

        ownGrid.SXYZ[0] = parameters->grid.SXYZ[0];
        ownGrid.SXYZ[1] = parameters->grid.SXYZ[1];
        ownGrid.SXYZ[2] = parameters->grid.SXYZ[2];

        int xyz[3];
        index1Dto3D(xyzID, &ownGrid, (int*)xyz);

        float coord[3];
        index3DtoCoords((int*)xyz, &ownGrid, (float*)coord);

        gridStepRadius=sqrt(ownGrid.gridStep[0]*ownGrid.gridStep[0]+ownGrid.gridStep[1]*ownGrid.gridStep[1]+ownGrid.gridStep[2]*ownGrid.gridStep[2]);

        for(int i=0; i<parameters.cavityInfo.numCoords;i++){
            
            float cavityPoint[3];
            cavityPoint[0]=cavityGPU[i].x;
            cavityPoint[1]=cavityGPU[i].y;
            cavityPoint[2]=cavityGPU[i].z;
            d=distance2Points3(coord, cavityPoint);
            if(d <= gridStepRadius/2){
                grid[xyzID].csiteFlag=1;
            }
                
        }

        float score = 0.0f;
        for(int i = 0; i < (*numGoodReceptors); i++) {

            score += FplpActual((float*)coord, classID, &(receptorAtoms[receptorIndex[i]]));
        }

        grid[xyzID].point[classID] = score;

        // Set at index=0 c_site penalty, for coords outside docking site.
        if(xyzID == 0) {
            grid[xyzID].point[classID] = 50.0f;
        }
    }
}
