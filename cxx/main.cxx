#include <iostream>
#include <string>
#include <cstddef>
#include "vector.h"

int main()
{
    std::cout << "Hello World\n";
    std::cout << "Max alignment: " << alignof(std::max_align_t) << std::endl;

    auto vec = dyvp::Vector<float, 0>(10, 3.14159);
    auto vec_sum = dyvp::sum(vec + vec);
    return 0;
}