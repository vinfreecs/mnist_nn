#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include <random>
#include <fstream>
#include <cmath>
#include <chrono>
#include <sstream>
#include <unordered_map>
#include "helper_functions.hpp"

void random_xavier_weight(Eigen::MatrixXd &weights_two, double input_size, double output_size, unsigned int seed_two)
{
    std::mt19937 gen(seed_two);

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

    std::normal_distribution<double> distri(0, std::sqrt(2.0 / input_size));

    for (int i = 0; i < 500; i++)
    {
        for (int j = 0; j < 784; j++)
        {
            weights_one(i, j) = distri(gen);
        }
    }
}

int main(int agrs, char* argv[])
{

    std::unordered_map<std::string, std::string> configMap;
    std::cout<<"the path in terminal: the argv path 1"<<argv[1]<<std::endl;
    std::string config_file_path =argv[1];
    std::cout<<"the path in terminal: the cofig path before modification 1"<<argv[1]<<std::endl;
    //config_file_path = "../"+config_file_path;
    std::cout<<"the path in terminal: the modified path "<<config_file_path<<std::endl;

    read_config_file(config_file_path, configMap);
    int input_size = 784;
    int hidden_size = std::stoi(configMap["hidden_size"]);
    int num_classes = 10;
    int num_epochs = std::stoi(configMap["num_epochs"]);;
    int batch_size = std::stoi(configMap["batch_size"]);
    double learning_rate = std::stod(configMap["learning_rate"]);
    // train dataset with batch size 100 100,784

    std::string train_image_path = configMap["rel_path_train_images"];
    int train_image_size = 0;
    std::vector<Eigen::MatrixXd> images_batched;
    read_image_as_batch(train_image_path, images_batched, train_image_size, batch_size);
    

    // label dataset
    std::string train_label_path = configMap["rel_path_train_labels"];
    std::vector<Eigen::MatrixXd> labels_batched;
    read_label_as_batch(labels_batched, train_label_path, batch_size);
    
    // for FC1 baises are xero initially weights are randomly generated between 0,sqrt(2/784)
    Eigen::MatrixXd weights_one(hidden_size, input_size);
    std::random_device rd;
    unsigned int seed_one = 10;
    random_he_weight(weights_one, input_size, seed_one);
    Eigen::MatrixXd baises_one = Eigen::VectorXd::Zero(hidden_size);

    // xavier initialisation for softmax
    Eigen::MatrixXd weights_two(num_classes, hidden_size);
    std::random_device rd2;
    unsigned int seed_two = 11;
    random_xavier_weight(weights_two, hidden_size, num_classes, seed_two);
    Eigen::MatrixXd baises_two = Eigen::VectorXd::Zero(num_classes);

    Eigen::MatrixXd fc1;
    Eigen::MatrixXd relu;
    Eigen::MatrixXd fc2;
    Eigen::MatrixXd softmax;
    std::cout<<"Training started"<<std::endl;
    auto train_start = std::chrono::high_resolution_clock::now();

    for (int epoch = 0; epoch < num_epochs; epoch++)
    {
        auto start = std::chrono::high_resolution_clock::now();
        std::cout << "Epoch : " << epoch + 1 << std::endl;

        for (int i = 0; i < images_batched.size(); i++)
        {
            // FC1
            // w1 500*784 i_b 100*784 +500*100 =100,500 =fc1
            fc1 = images_batched[i] * weights_one.transpose();

            for (int bais = 0; bais < batch_size; bais++)
            {
                fc1.row(bais) += baises_one.transpose();
            }

            // relu
            relu = fc1.array().max(0.0);

            // fc2
            // relu is 100,500 weight2 is10,500
            fc2 = relu * weights_two.transpose();

            for (int bais = 0; bais < batch_size; bais++)
            {
                fc2.row(bais) += baises_two.transpose();
            }

            // softmax
            softmax = fc2;
            for (int idx = 0; idx < batch_size; idx++)
            {
                Eigen::VectorXd row = softmax.row(idx);
                row = row.array().exp();
                double deno = row.sum();
                softmax.row(idx) = row / deno;
                
            }
            // cross entropy loss
            // write later for batching
            // error tensor of softmax layer en-1 = y^-y
            Eigen::MatrixXd softmax_error_tensor = softmax - labels_batched[i];

            // error of softmax is 100,10 and relu is 100,500 so en-1.transpose*relu = 10,100 * 100,500 = 10, 500
            Eigen::MatrixXd gradient_fc2_weights = softmax_error_tensor.transpose() * relu;
            Eigen::MatrixXd gradient_fc2_baises = softmax_error_tensor.colwise().sum();
            gradient_fc2_baises = gradient_fc2_baises.transpose();

            // error tensor of relu layer in batch w2 is 10,500 softmaxerror is 100,10 to get
            Eigen::MatrixXd relu_error_tensor = softmax_error_tensor * weights_two;
            for (int i = 0; i < batch_size; i++)
            {
                for (int j = 0; j < relu_error_tensor.cols(); j++)
                {
                    if (fc1(i, j) <= 0)
                        relu_error_tensor(i, j) = 0;
                }
            }
            // relu_error_tensor = relu_error_tensor.array() * (fc1.array() > 0).cast<double>().array();

            // reluerror is 100,500 imagesbatched is 100,784
            Eigen::MatrixXd gradient_fc1_weights = relu_error_tensor.transpose() * images_batched[i];
            Eigen::MatrixXd gradient_fc1_baises = relu_error_tensor.colwise().sum();
            gradient_fc1_baises = gradient_fc1_baises.transpose();

            // weights and baises update
            weights_one -= learning_rate * gradient_fc1_weights;
            baises_one -= learning_rate * gradient_fc1_baises;

            weights_two -= learning_rate * gradient_fc2_weights;
            baises_two -= learning_rate * gradient_fc2_baises;
        }
        auto end = std::chrono::high_resolution_clock::now();
        auto total_duration = std::chrono::duration_cast<std::chrono::seconds>(end - start);
        std::cout << "Time taen By Single Epoch : " << total_duration.count() << " seconds " << std::endl;
    }
    auto train_end = std::chrono::high_resolution_clock::now();
    auto total_duration_train = std::chrono::duration_cast<std::chrono::seconds>(train_end - train_start);
    std::cout << "Time taen for traing : " << total_duration_train.count() << " seconds " << std::endl;

    std::cout<<"Training completed"<<std::endl;

    std::cout<<"testing started"<<std::endl;


    // testing
    // test image dataset
    std::vector<Eigen::MatrixXd> test_images_batched;
    std::string test_image_path = configMap["rel_path_test_images"];
    int test_image_size = 0;
    read_image_as_batch(test_image_path, test_images_batched, test_image_size, batch_size);

    // test label dataset
    std::vector<Eigen::MatrixXd> test_labels_batched;
    std::string test_label_path = configMap["rel_path_test_labels"];
    read_label_as_batch(test_labels_batched, test_label_path, batch_size);

    Eigen::VectorXd fc1_test;
    Eigen::VectorXd relu_test;
    Eigen::VectorXd fc2_test;
    Eigen::VectorXd softmax_test;
    std::string log_path = configMap["rel_path_log_file"];
    std::ofstream out(log_path);

    for (int i = 0; i < test_image_size/batch_size; i++)
    {
        // FC1
        // w1 500*784 i_b 100*784 +500*100 =100,500 =fc1
        fc1 = test_images_batched[i] * weights_one.transpose();

        for (int bais = 0; bais < batch_size; bais++)
        {
            fc1.row(bais) += baises_one.transpose();
        }

        // relu
        relu = fc1.array().max(0.0);

        // fc2
        // relu is 100,500 weight2 is10,500
        fc2 = relu * weights_two.transpose();

        for (int bais = 0; bais < batch_size; bais++)
        {
            fc2.row(bais) += baises_two.transpose();
        }

        // softmax
        softmax = fc2;
        for (int idx = 0; idx < batch_size; idx++)
        {
            Eigen::VectorXd row = softmax.row(idx);
            row = row.array().exp();
            double deno = row.sum();
            softmax.row(idx) = row / deno;
        }
        

        out << "Current batch: " << i << std::endl;
        std::cout << "Current batch: " << i << std::endl;
        for (int j = 0; j < batch_size; j++)
        {
            double pred_valu = 0;
            int predicted = 0;
            int real = 0;
            int image_idx = batch_size * i + j;
            for (int k = 0; k < num_classes; k++)
            {
                if (test_labels_batched[i](j, k) == 1)
                    real = k;
                if (softmax(j, k) > pred_valu)
                {
                    pred_valu = softmax(j, k);
                    predicted = k;
                }
            }

            out << " - image " << image_idx << ": " << "Prediction=" << predicted << "." << " Label=" << real << std::endl;
            std::cout << " - image " << image_idx << ": " << "Prediction=" << predicted << "." << " Label=" << real << std::endl;
        }
    }
    std::cout<<"testing completed"<<std::endl;

}