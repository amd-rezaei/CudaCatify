// util.hpp
#ifndef UTIL_HPP
#define UTIL_HPP

#include <filesystem>
#include <algorithm>
#include <string>

// Get the base name of a file without the extension
std::string getBaseName(const std::string &filepath);

// Clamp a value between a minimum and maximum
float clamp(float value, float minValue, float maxValue);

#endif // UTIL_HPP
