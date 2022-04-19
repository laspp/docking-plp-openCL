#include "io.hpp"

#include "dbgl.hpp"

void openFile(std::string file, std::ifstream& fileIfs) {

    fileIfs = std::ifstream(file.c_str());
    if(fileIfs.fail()){
        dbglWE(__FILE__, __FUNCTION__, __LINE__, "Failed to open file: " + file);
    }
}

void openFileBinary(std::string file, FILE*& File) {

    File = fopen(file.c_str(), "rb");
    if(File == NULL){
        dbglWE(__FILE__, __FUNCTION__, __LINE__, "Failed to open file: " + file);
    }
}

void openFileC(std::string file, FILE*& File) {

    File = fopen(file.c_str(), "w");
    if(File == NULL){
        dbglWE(__FILE__, __FUNCTION__, __LINE__, "Failed to open file: " + file);
    }
}
