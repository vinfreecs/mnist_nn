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