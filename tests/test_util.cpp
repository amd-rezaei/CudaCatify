// test_util.cpp
#include "util.hpp"
#include <gtest/gtest.h>

TEST(Util, ClampFunction)
{
    EXPECT_EQ(clamp(10.0f, 0.0f, 5.0f), 5.0f); // Clamped to max
    EXPECT_EQ(clamp(-1.0f, 0.0f, 5.0f), 0.0f); // Clamped to min
    EXPECT_EQ(clamp(2.5f, 0.0f, 5.0f), 2.5f);  // No clamping required
}
