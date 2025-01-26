#include <iostream>
#include <Eigen/Dense>
#include <fstream>
#include <string>
#include <vector>
//#include <print>
//#include<chrono>
#include "helper_functions.hpp"

int main(int agrs, char* argv[]){

    //part one a image part

    //path for the images = "../mnist-datasets/single-image.idx3-ubyte"
    
    //auto start = std::chrono::high_resolution_clock::now();
    

    std::string path = argv[1];
    std::vector<Eigen::MatrixXd> images;
    int no_of_images = 0;
    read_image(path,images,no_of_images);
    std::cout<<no_of_images<<std::endl;
    Eigen::IOFormat Persion(2);

    // std::cout<<images[std::stoi(argv[3])].format(Persion)<<std::endl;

    std::string output_image_path = argv[2];
    int img_idx = std::stoi(argv[3]);
    write_image(images[img_idx],output_image_path);

    //auto end = std::chrono::high_resolution_clock::now();

    //std::chrono::duration<double> time_value=end-start;
    //std::println("the time taken: {}",time_value.count());

}