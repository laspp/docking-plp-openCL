#pragma once

#include <fstream>
#include <string>
#include <cstdio>

void openFile(std::string file, std::ifstream& fileIfs);
void openFileBinary(std::string file, FILE*& File);
void openFileC(std::string file, FILE*& File);