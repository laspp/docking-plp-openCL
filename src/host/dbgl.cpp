#include "dbgl.hpp"

#include <iostream>

void dbglWE(std::string file, std::string function, uint32_t line, std::string comment) {
    std::cout << "Error in file: " << file << " in function: " << function << " at line: "<< line << ". " << comment << std::endl;
    exit(EXIT_FAILURE);
}

void dbgl(std::string file, std::string function, uint32_t line, std::string comment) {
    std::cout << "Error in file: " << file << " in function: " << function << " at line: "<< line << ". " << comment << std::endl;
}

void dbgl(std::string comment) {
    std::cout << "dbgl: " << comment << std::endl;
}

void dbg(std::string comment) {
    std::cout << comment;
}
