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

    // train dataset with batch size 100 100,784
    std::string train_image_path = "../mnist-datasets/train-images.idx3-ubyte";
    int train_image_size = 0;
    std::vector<Eigen::MatrixXd> images_batched;
    read_image_as_batch(train_image_path, images_batched, train_image_size, batch_size);
    std::cout << train_image_size << std::endl;
    std::cout << images_batched.size() << std::endl;
    std::cout << images_batched[1].rows() << std::endl;
    std::cout << images_batched[1].cols() << std::endl;



    // label dataset
    std::string train_label_path = "../mnist-datasets/train-labels.idx1-ubyte";
    std::vector<Eigen::MatrixXd> labels_batched;
    read_label_as_batch(labels_batched,train_label_path,batch_size);
    std::cout<<labels_batched.size()<<std::endl;
    std::cout<<labels_batched[1].rows()<<std::endl;
    std::cout<<labels_batched[1].cols()<<std::endl;



    // for FC1 baises are xero initially weights are randomly generated between 0,sqrt(2/784)
    Eigen::MatrixXd weights_one(500, 784);
    std::random_device rd;
    unsigned int seed_one = 10;
    random_he_weight(weights_one, input_size, seed_one);
    Eigen::MatrixXd baises_one = Eigen::VectorXd::Zero(hidden_size);

    // xavier initialisation for softmax
    Eigen::MatrixXd weights_two(10, 500);
    std::random_device rd2;
    unsigned int seed_two = 11;
    random_xavier_weight(weights_two, hidden_size, num_classes, seed_two);
    Eigen::MatrixXd baises_two = Eigen::VectorXd::Zero(num_classes);

    Eigen::MatrixXd fc1;
    Eigen::MatrixXd relu;
    Eigen::MatrixXd fc2;
    Eigen::MatrixXd softmax;

    for (int epoch = 0; epoch < 5; epoch++)
    {
        auto start = std::chrono::high_resolution_clock::now();
        std::cout << "Epoch : " << epoch + 1 << std::endl;

        for (int i = 0; i < images_batched.size(); i++)
        {
            // FC1
            //w1 500*784 i_b 100*784 +500*100 =100,500 =fc1
            fc1 = images_batched[i] * weights_one.transpose();

            for(int bais=0 ; bais <batch_size; bais++){
                fc1.row(bais) += baises_one.transpose();
            }
            

            // relu 
            relu = fc1.array().max(0.0);

            //fc2
            //relu is 100,500 weight2 is10,500
            fc2 = relu * weights_two.transpose();

            for(int bais=0 ; bais <batch_size; bais++){
                fc2.row(bais) += baises_two.transpose();
            }

            // softmax
            softmax = fc2;
            for(int idx =0 ;idx<batch_size;idx++){
                Eigen::VectorXd row = softmax.row(idx);
                row = row.array().exp();
                double deno = row.sum();
                softmax.row(idx) = row/deno;
            }
            // cross entropy loss
            //write later for batching
            // error tensor of softmax layer en-1 = y^-y
            Eigen::MatrixXd softmax_error_tensor = softmax - labels_batched[i];

            // error of softmax is 100,10 and relu is 100,500 so en-1.transpose*relu = 10,100 * 100,500 = 10, 500
            Eigen::MatrixXd gradient_fc2_weights = softmax_error_tensor.transpose() * relu;
            Eigen::MatrixXd gradient_fc2_baises = softmax_error_tensor.colwise().sum();
            gradient_fc2_baises=gradient_fc2_baises.transpose();

            // error tensor of relu layer in batch w2 is 10,500 softmaxerror is 100,10 to get 
            Eigen::MatrixXd relu_error_tensor =softmax_error_tensor * weights_two;
            for (int i = 0; i < batch_size; i++)
            {
                for (int j = 0; j < relu_error_tensor.cols(); j++)
                {
                    if (fc1(i,j) <= 0)
                        relu_error_tensor(i,j) = 0;
                }
            }
            //relu_error_tensor = relu_error_tensor.array() * (fc1.array() > 0).cast<double>().array();

            //reluerror is 100,500 imagesbatched is 100,784 
            Eigen::MatrixXd gradient_fc1_weights = relu_error_tensor.transpose() * images_batched[i];
            Eigen::MatrixXd gradient_fc1_baises = relu_error_tensor.colwise().sum();
            gradient_fc1_baises=gradient_fc1_baises.transpose();


            // weights and baises update

            // fc1
            // std::cout<<"enter update w and b 1 error"<<std::endl;
            // std::cout<<"size of grad fc1 w "<<gradient_fc1_weights.rows()<<"<-rows cols->"<<gradient_fc1_weights.cols()<<std::endl;
            // std::cout<<"size of w1 "<<weights_one.rows()<<"<-rows cols->"<<weights_one.cols()<<std::endl;
            // std::cout<<"size of grad fc1 b"<<gradient_fc1_baises.rows()<<"<-rows cols->"<<gradient_fc1_baises.cols()<<std::endl;
            // std::cout<<"shape of b1 "<<baises_one.rows()<<"<-rows cols->"<<baises_one.cols()<<std::endl;


            weights_one -= learning_rate * gradient_fc1_weights;
            baises_one -= learning_rate * gradient_fc1_baises;

            // std::cout<<"exit update w and b 1 error"<<std::endl;


            // fc2
            // std::cout<<"size of grad fc w2"<<gradient_fc2_weights.rows()<<"<-rows cols->"<<gradient_fc2_weights.cols()<<std::endl;
            // std::cout<<"size of grad fc b2"<<gradient_fc2_baises.rows()<<"<-rows cols->"<<gradient_fc2_weights.cols()<<std::endl;
            weights_two -= learning_rate * gradient_fc2_weights;
            baises_two -= learning_rate * gradient_fc2_baises;

        }
        auto end = std::chrono::high_resolution_clock::now();
        auto total_duration = std::chrono::duration_cast<std::chrono::seconds>(end - start);
        std::cout << "Time taen By Single Epoch : " << total_duration.count() << " seconds " << std::endl;
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

    Eigen::VectorXd fc1_test;
    Eigen::VectorXd relu_test;
    Eigen::VectorXd fc2_test;
    Eigen::VectorXd softmax_test;

    for (int i = 0; i < 10; i++)
    {
        // FC1
        fc1_test = weights_one * test_images[i] + baises_one;

        // relu
        relu_test = fc1_test.array().max(0.0);

        // fc2
        fc2_test = weights_two * relu_test + baises_two;

        // softmax
        softmax_test = fc2_test.array().exp();
        softmax_test = softmax_test / softmax_test.sum();
        std::cout<<"the index of test image : "<<i<<std::endl;
        for(int j =0 ;j<softmax_test.size();j++){
            std::cout<<"softmax values : "<<softmax_test[j]<<"   the true vale : "<<test_labels[i][j]<<std::endl;
        }

        // cross entropy loss
        double cross_entropy_loss = 0;
        for (int j = 0; j < num_classes; j++)
        {
            cross_entropy_loss += test_labels[i][j] * std::log(softmax_test[j]);
        }
        std::cout<<"cross entrphy loss : "<<cross_entropy_loss<<std::endl;
    }
}