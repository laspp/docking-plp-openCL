#ifndef SORT_POPULATIONS_CL_H
#define SORT_POPULATIONS_CL_H

// Adapter from: https://www.geeksforgeeks.org/bubble-sort/
void bubblePopulationSort(int startIndex, int endIndex, local float* array, local ushort* indexes) {
    
    local float* arraySetToZero = &(array[startIndex]);
    local ushort* indexesSetToZero = &(indexes[startIndex]);
    int n = endIndex - startIndex;

    float tempRW;
    ushort tempI;
    for (int i = 0; i < n-1; i++) {
        for (int j = 0; j < n-i-1; j++) {
            if (arraySetToZero[j] < arraySetToZero[j+1]) {
                // Swap Values
                tempRW = arraySetToZero[j];
                arraySetToZero[j] = arraySetToZero[j+1];
                arraySetToZero[j+1] = tempRW;
                // Swap Indexes
                tempI = indexesSetToZero[j];
                indexesSetToZero[j] = indexesSetToZero[j+1];
                indexesSetToZero[j+1] = tempI;
            }
        }
    }
}

// Adapter from: https://www.geeksforgeeks.org/merge-sort/
void mergePopulationSort(int startIndex, int length, int step01, int popMaxSize, int size, local float* array, local ushort* indexes) {
    // step01 == 1, put in second part od array
    int source = (step01 == 0) ? 1 : 0;
    int destination = step01;

    int currentLeft = startIndex;
    int currentRight = startIndex + length;

    int endLeft = currentRight;
    int endRight = currentRight + length;

    int k = startIndex;

    while (currentLeft < endLeft && currentLeft < size && currentRight < endRight && currentRight < size) {
        if (array[source*popMaxSize+currentLeft] >= array[source*popMaxSize+currentRight]) {
            array[destination*popMaxSize+k] = array[source*popMaxSize+currentLeft];
            indexes[destination*popMaxSize+k] = indexes[source*popMaxSize+currentLeft];
            currentLeft++;
        }
        else {
            array[destination*popMaxSize+k] = array[source*popMaxSize+currentRight];
            indexes[destination*popMaxSize+k] = indexes[source*popMaxSize+currentRight];
            currentRight++;
        }
        k++;
    }
    
    while (currentLeft < endLeft && currentLeft < size) {
        array[destination*popMaxSize+k] = array[source*popMaxSize+currentLeft];
        indexes[destination*popMaxSize+k] = indexes[source*popMaxSize+currentLeft];
        currentLeft++;
        k++;
    }

    while (currentRight < endRight && currentRight < size) {
        array[destination*popMaxSize+k] = array[source*popMaxSize+currentRight];
        indexes[destination*popMaxSize+k] = indexes[source*popMaxSize+currentRight];
        currentRight++;
        k++;
    }
}

#endif
