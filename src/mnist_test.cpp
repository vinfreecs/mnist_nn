#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include <random>
#include <fstream>
#include <cmath>
#include <chrono>
#include "helper_functions.hpp"

void random_xavier_weight(Eigen::MatrixXd &weights_two, double input_size, double output_size, unsigned int seed_two)
{
    std::mt19937 gen(seed_two);

    std::cout << "Random Seed two XAVIER: " << seed_two << std::endl;

    // std::normal_distribution<double> distri(-std::sqrt(6 / (input_size+output_size)), std::sqrt(6 / (input_size+output_size)));
    std::normal_distribution<double> distri(0, std::sqrt(2 / (input_size + output_size)));

    for (int i = 0; i < output_size; i++)
    {
        for (int j = 0; j < input_size; j++)
        {
            weights_two(i, j) = distri(gen);
        }
    }
}

void random_he_weight(Eigen::MatrixXd &weights_one, double input_size, unsigned int seed_one)
{

    std::mt19937 gen(seed_one);

    std::cout << "Random Seed one HE: " << seed_one << std::endl;

    std::normal_distribution<double> distri(0, std::sqrt(2.0 / input_size));

    for (int i = 0; i < 500; i++)
    {
        for (int j = 0; j < 784; j++)
        {
            weights_one(i, j) = distri(gen);
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
    std::vector<Eigen::VectorXd> images;
    std::string train_image_path = "../mnist-datasets/train-images.idx3-ubyte";
    int train_image_size = 0;
    flatten_image(train_image_path, images, train_image_size);
    std::cout << train_image_size << std::endl;

    // label dataset
    std::vector<Eigen::VectorXd> train_labels;
    std::string train_label_path = "../mnist-datasets/train-labels.idx1-ubyte";
    read_eigen_label(train_labels, train_label_path);

    // for FC1 baises are xero initially weights are randomly generated between 0,sqrt(2/784)
    Eigen::MatrixXd weights_one(500, 784);
    std::random_device rd;
    unsigned int seed_one = 10;
    random_he_weight(weights_one, input_size, seed_one);
    Eigen::VectorXd baises_one = Eigen::VectorXd::Zero(hidden_size);

    // xavier initialisation for softmax
    Eigen::MatrixXd weights_two(10, 500);
    std::random_device rd2;
    unsigned int seed_two = 11;
    random_xavier_weight(weights_two, hidden_size, num_classes, seed_two);
    Eigen::VectorXd baises_two = Eigen::VectorXd::Zero(num_classes);

    Eigen::VectorXd fc1;
    Eigen::VectorXd relu;
    Eigen::VectorXd fc2;
    Eigen::VectorXd softmax;

    for (int epoch = 0; epoch < 5; epoch++)
    {
        auto start = std::chrono::high_resolution_clock::now();
        std::cout << "Epoch : " << epoch + 1 << std::endl;

        for (int i = 0; i < images.size(); i++)
        {
            // FC1
            fc1 = weights_one * images[i] + baises_one;

            // relu
            relu = fc1.array().max(0.0);

            // fc2
            fc2 = weights_two * relu + baises_two;

            // softmax
            softmax = fc2.array().exp();
            softmax = softmax / softmax.sum();

            // cross entropy loss
            double cross_entropy_loss = 0;
            for (int j = 0; j < num_classes; j++)
            {
                cross_entropy_loss += train_labels[i][j] * std::log(softmax[j]);
            }

            // cross entropy loss error tensor en= -yi/yi^
            std::vector<double> first_error_tensor(10);
            for (int j = 0; j < 10; j++)
            {
                first_error_tensor[j] = (-1) * (train_labels[i][j] / softmax[j]);
            }

            // error tensor of softmax layer en-1 = y^-y
            Eigen::VectorXd softmax_error_tensor = softmax - train_labels[i];

            // error of softmax is 10,1 and relu is 500,1 so en-1*relu.Transpose = 10,1 * 1,500 = 10, 500
            Eigen::MatrixXd gradient_fc2_weights = softmax_error_tensor * relu.transpose();
            Eigen::VectorXd gradient_fc2_baises = softmax_error_tensor;

            // error tensor of relu layer weights2 is 10,500 and softmaxError is 10 so to get a 500 relu error
            Eigen::VectorXd relu_error_tensor = weights_two.transpose() * softmax_error_tensor;

            for (int i = 0; i < relu_error_tensor.size(); i++)
            {
                if (fc1[i] <= 0)
                {
                    relu_error_tensor[i] = 0;
                }
            }

            Eigen::MatrixXd gradient_fc1_weights = relu_error_tensor * images[i].transpose();
            Eigen::VectorXd gradient_fc1_baises = relu_error_tensor;

            // weights and baises update

            // fc1
            weights_one -= learning_rate * gradient_fc1_weights;
            baises_one -= learning_rate * gradient_fc1_baises;

            // fc2
            weights_two -= learning_rate * gradient_fc2_weights;
            baises_two -= learning_rate * gradient_fc2_baises;
        }
        auto end = std::chrono::high_resolution_clock::now();
        auto total_duration = std::chrono::duration_cast<std::chrono::seconds>(end - start);
        std::cout<<"Time taen By Single Epoch : "<< total_duration.count()<<" seconds "<<std::endl;
    }

    //testing
    //test image dataset
    std::vector<Eigen::VectorXd> test_images;
    std::string test_image_path = "../mnist-datasets/t10k-images.idx3-ubyte";
    int test_image_size = 0;
    flatten_image(test_image_path, test_images, test_image_size);
    std::cout<<test_image_size<<std::endl;

    //test label dataset
    std::vector<Eigen::VectorXd>test_labels;
    std::string test_label_path = "../mnist-datasets/t10k-labels.idx1-ubyte";
    read_eigen_label(test_labels,test_label_path);

    for (int i = 0; i < 100; i++)
    {
        // FC1
        fc1 = weights_one * test_images[i] + baises_one;

        // relu
        relu = fc1.array().max(0.0);

        // fc2
        fc2 = weights_two * relu + baises_two;

        // softmax
        softmax = fc2.array().exp();
        softmax = softmax / softmax.sum();
        std::cout<<"the index of test image : "<<i<<std::endl;
        for(int j =0 ;j<softmax.size();j++){
            std::cout<<"softmax values : "<<softmax[j]<<"   the true vale : "<<test_labels[i][j]<<std::endl;
        }

        // cross entropy loss
        double cross_entropy_loss = 0;
        for (int j = 0; j < num_classes; j++)
        {
            cross_entropy_loss += test_labels[i][j] * std::log(softmax[j]);
        }
        std::cout<<"cross entrphy loss : "<<cross_entropy_loss<<std::endl;
    }
}