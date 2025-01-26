#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include <random>
#include <fstream>
#include <cmath>
#include "helper_functions.hpp"

void random_xavier_weight(Eigen::MatrixXd &weights_two, double input_size,double output_size, unsigned int seed_two){
    std::mt19937 gen(seed_two);

    std::cout << "Random Seed two XAVIER: " << seed_two << std::endl;

    // std::normal_distribution<double> distri(-std::sqrt(6 / (input_size+output_size)), std::sqrt(6 / (input_size+output_size)));
    std::normal_distribution<double> distri(0, std::sqrt(2 / (input_size+output_size)));


    for(int i=0 ; i<output_size; i++){
        for (int j = 0; j < input_size; j++)
        {
            weights_two(i,j) = distri(gen);
        }
    }
    
}

void random_he_weight(Eigen::MatrixXd& weights_one, double input_size, unsigned int seed_one)
{

    std::mt19937 gen(seed_one);

    std::cout << "Random Seed one HE: " << seed_one << std::endl;

    std::normal_distribution<double> distri(0, std::sqrt(2.0 / input_size));

    for (int i = 0; i < 500; i++)
    {
        for (int j = 0; j < 784; j++)
        {
            weights_one(i,j)= distri(gen);
        }
    }
}

int main()
{
    double input_size = 784;
    double hidden_size = 500;
    double num_classes = 10;
    double num_epochs = 5;
    double batch_size = 100;
    double learning_rate = 0.001;

    // train dataset
    std::vector<Eigen::VectorXd> flat_train_dataset;
    std::string train_image_path = "../mnist-datasets/train-images.idx3-ubyte";
    int train_image_size = 0;
    flatten_image(train_image_path, flat_train_dataset, train_image_size);
    Eigen::VectorXd image = flat_train_dataset[0];
    std::cout<<train_image_size<<std::endl;


    //label dataset
    std::vector<std::vector<double>>train_labels;
    std::string train_label_path = "../mnist-datasets/single-label.idx1-ubyte";
    read_label(train_labels,train_label_path);
    std::vector<double> label = train_labels[0];
    Eigen::VectorXd label_eigen = Eigen::Map<Eigen::VectorXd>(label.data(), label.size());

    // for FC1 baises are xero initially weights are randomly generated between 0,sqrt(2/784)
    Eigen::MatrixXd weights_one(500,784);
    std::random_device rd;
    unsigned int seed_one = 10;
    random_he_weight(weights_one, input_size, seed_one);
    Eigen::VectorXd baises_one = Eigen::VectorXd::Zero(hidden_size);

    //FC1
    Eigen::VectorXd fc1 = weights_one * image + baises_one;

    //relu
    Eigen::VectorXd relu = fc1.array().max(0.0);

    //xavier initialisation for softmax 
    Eigen::MatrixXd weights_two(10,500);
    std::random_device rd2;
    unsigned int seed_two = 11;
    random_xavier_weight(weights_two, hidden_size,num_classes,seed_two);
    Eigen::VectorXd baises_two = Eigen::VectorXd::Zero(num_classes);

    //fc2
    Eigen::VectorXd fc2 = weights_two * relu + baises_two;
    
    //softmax
    Eigen::VectorXd softmax = fc2.array().exp();
    softmax = softmax / softmax.sum();

    //cross entropy loss
    double cross_entropy_loss = 0;
    for(int i=0;i<num_classes;i++){
        cross_entropy_loss += label[i]*std::log(softmax[i]);
    }
    std::cout<<"the cross entropy loss : "<<cross_entropy_loss<<std::endl;

    //cross entropy loss error tensor en= -yi/yi^
    std::vector<double> first_error_tensor(10) ;
    for(int i=0; i<10;i++){
        first_error_tensor[i]=(-1)*(label[i]/softmax[i]);
        std::cout<<i<<" th error loss = "<<first_error_tensor[i]<<std::endl;
    }

    //error tensor of softmax layer en-1 = y^-y 
    Eigen::VectorXd softmax_error_tensor = softmax - label_eigen;
    for(int i=0 ;i <10 ;i++){
        std::cout<<i<<" th softmax error loss = "<<softmax_error_tensor[i]<<std::endl;
    }

    //error of softmax is 10,1 and relu is 500,1 so en-1*relu.Transpose = 10,1 * 1,500 = 10, 500
    Eigen::MatrixXd gradient_fc2_weights = softmax_error_tensor * relu.transpose();
    Eigen::VectorXd gradient_fc2_baises = softmax_error_tensor;

    //error tensor of relu layer weights2 is 10,500 and softmaxError is 10 so to get a 500 relu error
    Eigen::VectorXd relu_error_tensor = weights_two.transpose()*softmax_error_tensor;

    for(int i=0;i<relu_error_tensor.size();i++){
        if(fc1[i] <= 0){
            relu_error_tensor[i] = 0;
        }
    }

    Eigen::MatrixXd gradient_fc1_weights = relu_error_tensor * image.transpose();
    Eigen::VectorXd gradient_fc1_baises = relu_error_tensor;

    //weights and baises update

    //fc1
    weights_one -= learning_rate*gradient_fc1_weights;
    baises_one -= learning_rate*gradient_fc1_baises;

    //fc2
    weights_two -= learning_rate*gradient_fc2_weights;
    baises_two -= learning_rate*gradient_fc2_baises;

    
    
}