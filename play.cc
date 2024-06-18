#include <fstream>
#include <ios>
#include <iostream>

int main() {
    std::ofstream result_file("result", std::ios::app);
    int batch_size = 8;
    int ratio = 0.001;
    result_file<<batch_size<<" "<<ratio<<std::endl;
    return 0;
}