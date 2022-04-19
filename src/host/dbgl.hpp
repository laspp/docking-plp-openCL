#pragma once

#include <string>

void dbglWE(std::string file, std::string function, uint32_t line, std::string comment);

void dbgl(std::string file, std::string function, uint32_t line, std::string comment);

void dbgl(std::string comment);

void dbg(std::string comment);
