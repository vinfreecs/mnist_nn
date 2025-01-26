#include <iostream>
#include <Eigen/Dense>
#include <fstream>
#include <string>
#include <vector>
//#include <print>
#include "helper_functions.hpp"

int main(int agrs, char* argv[]){

    //label part part one b

    std::vector<std::vector<double>>labels;
    read_label(labels,argv[1]);
    int label_idx = std::stoi(argv[3]);
    write_label(labels[label_idx],argv[2]);


    // for(auto i : labels[label_idx]){
    //     std::println("{}",i);
    // }

}