#include<iostream>
#include <vector>
#include <Eigen/Dense>
#include <random>
#include <fstream>
#include <cmath>
#include <chrono>
#include <sstream>
#include <unordered_map>
#include "helper_functions.hpp"
#include "weights_generator.hpp"

class neural_network
{
public:
    
    Eigen::MatrixXd fc1;
    Eigen::MatrixXd relu;
    Eigen::MatrixXd fc2;
    Eigen::MatrixXd softmax;
    Eigen::MatrixXd weights_two;
    Eigen::MatrixXd weights_one;
    Eigen::MatrixXd baises_one;
    Eigen::MatrixXd baises_two;
    neural_network(int hidden_size, int input_size, int num_classes)
    {
        // for FC1 baises are xero initially weights are randomly generated between 0,sqrt(2/784)
        weights_one.resize(hidden_size, input_size);
        std::random_device rd;
        unsigned int seed_one = 10;
        random_he_weight(weights_one, input_size, seed_one);
        // Eigen::MatrixXd w_b_1 (hidden_size, input_size+1);
        // w_b_1 << weights_one, baises_one;
        baises_one = Eigen::VectorXd::Zero(hidden_size);
        baises_two = Eigen::VectorXd::Zero(num_classes);
        // xavier initialisation for softmax
        weights_two.resize(num_classes, hidden_size);
        std::random_device rd2;
        unsigned int seed_two = 11;
        random_xavier_weight(weights_two, hidden_size, num_classes, seed_two);
    }
    

};

// class layer{
//     public:
//     virtual Eigen::MatrixXd forward();
//     virtual void backward(){};
//     virtual void update(){};

// };

class fc{
    public:
    Eigen::MatrixXd fc_mat;
    Eigen::MatrixXd gradient_fc_weights;
    Eigen::MatrixXd gradient_fc_baises;
    Eigen::MatrixXd forward(Eigen::MatrixXd prev_layer_mat, Eigen::MatrixXd weights, Eigen::MatrixXd baises){

        fc_mat= prev_layer_mat * weights.transpose();
        for (int bais = 0; bais < prev_layer_mat.rows(); bais++)
        {
            fc_mat.row(bais) += baises.transpose();
        }
        return fc_mat;
    }
    void backward(Eigen::MatrixXd error_mat, Eigen::MatrixXd input_mat){
        gradient_fc_weights = error_mat.transpose() * input_mat;
        gradient_fc_baises = error_mat.colwise().sum();
        gradient_fc_baises = gradient_fc_baises.transpose();
    }
    void update(Eigen::MatrixXd& weights, Eigen::MatrixXd& baises , double lr){
        
        weights -= lr * gradient_fc_weights;

        baises -= lr * gradient_fc_baises;

    }
};

class relu{
    public:
    Eigen::MatrixXd relu_mat;
    Eigen::MatrixXd forward(Eigen::MatrixXd fc1_mat){
        relu_mat = fc1_mat.array().max(0.0);
        return relu_mat;
    }
    Eigen::MatrixXd backward(Eigen::MatrixXd softmax_error_tensor, Eigen::MatrixXd weights_two, Eigen::MatrixXd fc_mat){
        Eigen::MatrixXd relu_error_tensor = softmax_error_tensor * weights_two;
            for (int i = 0; i < softmax_error_tensor.rows(); i++)
            {
                for (int j = 0; j < relu_error_tensor.cols(); j++)
                {
                    if (fc_mat(i, j) <= 0)
                        relu_error_tensor(i, j) = 0;
                }
            }
            return relu_error_tensor;
    }
};

class softmax{
    public:
    Eigen::MatrixXd softmax_mat;
    Eigen::MatrixXd softmax_error_tensor;
    Eigen::MatrixXd forward(Eigen::MatrixXd fc1_mat){
        softmax_mat = fc1_mat;
        for (int idx = 0; idx < fc1_mat.rows(); idx++)
            {
                Eigen::VectorXd row = softmax_mat.row(idx);
                row = row.array().exp();
                double deno = row.sum();
                softmax_mat.row(idx) = row / deno;
            }
            return softmax_mat;
    }
    Eigen::MatrixXd backward(Eigen::MatrixXd label_batched){
        softmax_error_tensor = softmax_mat - label_batched;
        return softmax_error_tensor;
    }
};

void train(){
    
}

int main(int agrs, char* argv[]){
    std::unordered_map<std::string, std::string> configMap;
    std::string config_file_path =argv[1];

    read_config_file(config_file_path, configMap);

    int input_size = 784;
    int hidden_size = std::stoi(configMap["hidden_size"]);
    int num_classes = 10;
    int num_epochs = std::stoi(configMap["num_epochs"]);;
    int batch_size = std::stoi(configMap["batch_size"]);
    double learning_rate = std::stod(configMap["learning_rate"]);

    // train dataset with batch size 100 100,784
    std::string train_image_path = "../"+configMap["rel_path_train_images"];
    int train_image_size = 0;
    std::vector<Eigen::MatrixXd> images_batched;
    read_image_as_batch(train_image_path, images_batched, train_image_size, batch_size);
    

    // label dataset
    std::string train_label_path = "../"+configMap["rel_path_train_labels"];
    std::vector<Eigen::MatrixXd> labels_batched;
    read_label_as_batch(labels_batched, train_label_path, batch_size);
        
    // test image dataset
    std::vector<Eigen::MatrixXd> test_images_batched;
    std::string test_image_path = "../"+configMap["rel_path_test_images"];
    int test_image_size = 0;
    read_image_as_batch(test_image_path, test_images_batched, test_image_size, batch_size);

    // test label dataset
    std::vector<Eigen::MatrixXd> test_labels_batched;
    std::string test_label_path = "../"+configMap["rel_path_test_labels"];
    read_label_as_batch(test_labels_batched, test_label_path, batch_size);

    std::string log_path = configMap["rel_path_log_file"];
    std::ofstream out(log_path);
    neural_network nn(hidden_size, input_size, num_classes);
    fc fc1;
    fc fc2;
    relu relu_ac;
    softmax softmax_ac;

    Eigen::MatrixXd carrier;

    for (int epoch = 0; epoch < num_epochs; epoch++)
    {
        auto start = std::chrono::high_resolution_clock::now();
        std::cout << "Epoch : " << epoch + 1 << std::endl;

        for (int i = 0; i < images_batched.size(); i++)
        {
            // FC1

            carrier = fc1.forward(images_batched[i], nn.weights_one, nn.baises_one);


            // relu
            carrier = relu_ac.forward(carrier);

            
            // fc2
            carrier = fc2.forward(carrier, nn.weights_two, nn.baises_two);

            // softmax
            carrier = softmax_ac.forward(carrier);

            // cross entropy loss

            // error tensor of softmax layer en-1 = y^-y
            carrier= softmax_ac.backward(labels_batched[i]);

            // error of softmax is 100,10 and relu is 100,500 so en-1.transpose*relu = 10,100 * 100,500 = 10, 500
            fc2.backward(carrier, relu_ac.relu_mat);

            // error tensor of relu layer in batch w2 is 10,500 softmaxerror is 100,10 to get
            carrier = relu_ac.backward(carrier, nn.weights_two, fc1.fc_mat);

            // relu_error_tensor = relu_error_tensor.array() * (fc1.array() > 0).cast<double>().array();
            // reluerror is 100,500 imagesbatched is 100,784
            fc1.backward(carrier, images_batched[i]);
            // weights and baises update
            fc1.update(nn.weights_one, nn.baises_one, learning_rate);
            fc2.update(nn.weights_two, nn.baises_two, learning_rate);
        }
        auto end = std::chrono::high_resolution_clock::now();
        auto total_duration = std::chrono::duration_cast<std::chrono::seconds>(end - start);
        std::cout << "Time taen By Single Epoch : " << total_duration.count() << " seconds " << std::endl;
    }
    std::cout<<"Training completed"<<std::endl;

    std::cout<<"testing started"<<std::endl;


    for (int i = 0; i < test_image_size/batch_size; i++)
    {
        // FC1
            carrier = fc1.forward(test_images_batched[i], nn.weights_one, nn.baises_one);

            // relu
            carrier = relu_ac.forward(carrier);
            
            // fc2
            carrier = fc2.forward(carrier, nn.weights_two, nn.baises_two);

            // softmax
            carrier = softmax_ac.forward(carrier);

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
                if (softmax_ac.softmax_mat(j, k) > pred_valu)
                {
                    pred_valu = softmax_ac.softmax_mat(j, k);
                    predicted = k;
                }
            }

            out << " - image " << image_idx << ": " << "Prediction=" << predicted << "." << " Label=" << real << std::endl;
            std::cout << " - image " << image_idx << ": " << "Prediction=" << predicted << "." << " Label=" << real << std::endl;
        }
    }
    std::cout<<"testing completed"<<std::endl;


}