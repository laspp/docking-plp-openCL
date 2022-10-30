#ifndef SORT_POPULATIONS_CL_H
#define SORT_POPULATIONS_CL_H

// Based on: https://www.youtube.com/watch?v=w544Rn4KC8I
// sortLength length must be 2^n
// Bitonic sort is not a stable sorting algorithm. Previously used sorting algorithm was stable, so possible differences.
void bitonicMergeSort(const uint localSize, const uint localID, const int sortLength, local float* array, local ushort* indexes) {

    // Stages: (arrows in same stage group have same sortDirection
    uint distance = 1; // initial arrow distance of a stage aka. compare distance, 1 is the minimum
    while(distance < sortLength) { // last distance is sortLength / 2.

        // Passes: (distance reduces by half every pass)
        uint tempDistance = distance;
        while(tempDistance > 0) { // last distance is 1

            // Arrows:
            uint tempLocalID = localID;
            while(tempLocalID < sortLength / 2) { // one arrow compares 2 numbers -> half as many arrows as numbers needed
                
                int groupID = tempLocalID / tempDistance; // ID of a group
                int arrowID = tempLocalID % tempDistance; // ID of an arrow inside it's group
                int sortDirection = (tempLocalID / distance) & 1; // & 1 == % 2, changing sort direction of stage groups ^v^v^v^

                int aIndex = groupID * (tempDistance << 1) + arrowID; // Index of fist element
                int bIndex = aIndex + tempDistance; // Index of second element

                barrier(CLK_LOCAL_MEM_FENCE);

                float a = array[aIndex];
                float b = array[bIndex];

                if(a < b == sortDirection) {
                    // Swap:
                    array[bIndex] = a;
                    array[aIndex] = b;
                    ushort ai = indexes[aIndex];
                    ushort bi = indexes[bIndex];
                    indexes[bIndex] = ai;
                    indexes[aIndex] = bi;
                }
                tempLocalID += localSize;
            }
            tempDistance >>= 1;
        }
        distance <<= 1;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
}

#endif
