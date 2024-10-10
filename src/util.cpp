// util.cpp
#include "util.hpp"

// Get the base name of a file without the extension
std::string getBaseName(const std::string &filepath)
{
    std::filesystem::path p(filepath);
    return p.stem().string();
}

// Clamp a value between a minimum and maximum
float clamp(float value, float minValue, float maxValue)
{
    return std::max(minValue, std::min(value, maxValue));
}
