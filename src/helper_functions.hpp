int reverseInt(int num)
{
    return ((num >> 24) & 0x000000FF) |
           ((num >> 8) & 0x0000FF00) |
           ((num << 8) & 0x00FF0000) |
           ((num << 24) & 0xFF000000);
}

double percision(double num){
    return std::round(num*100)/100;
}
void remove_spaces(std::string& str){
    std::string::iterator end_pos = std::remove(str.begin(), str.end(), ' ');
    str.erase(end_pos, str.end());
}

void read_image(std::string& path, std::vector<Eigen::MatrixXd>& images,int& no_of_images ){
    std::ifstream file(path);
    if(!file) std::cout<<"File read failed"<<std::endl;
    unsigned char ele =1;
    int magicNumbers = 0;
    file.read((char*)&magicNumbers,sizeof(magicNumbers));
    magicNumbers = reverseInt(magicNumbers);
    file.read((char*)&no_of_images,sizeof(no_of_images));
    no_of_images = reverseInt(no_of_images);
    int rows = 0 ;
    file.read((char*)&rows,sizeof(rows));
    rows = reverseInt(rows);
    int cols = 0;
    file.read((char*)&cols,sizeof(cols));
    cols = reverseInt(cols);
    Eigen::MatrixXd img(rows,cols);
    for(size_t k=0 ; k<no_of_images;k++){
        for(size_t i =0 ; i<28; i++){
            for(size_t j = 0 ; j <28 ; j++){
                file.read((char*)(&ele),sizeof(ele));
                double normal = (((double)ele)/255);
                img(i,j) = normal;
            }
        }
        images.push_back(img);
    }
    
}
void read_image_flat(std::string& path, std::vector<std::vector<double>>& images,int& no_of_images ){
    std::ifstream file(path);
    if(!file) std::cout<<"File read failed"<<std::endl;
    unsigned char ele =1;
    int magicNumbers = 0;
    file.read((char*)&magicNumbers,sizeof(magicNumbers));
    magicNumbers = reverseInt(magicNumbers);
    file.read((char*)&no_of_images,sizeof(no_of_images));
    no_of_images = reverseInt(no_of_images);
    int rows = 0 ;
    file.read((char*)&rows,sizeof(rows));
    rows = reverseInt(rows);
    int cols = 0;
    file.read((char*)&cols,sizeof(cols));
    cols = reverseInt(cols);
    for(size_t k=0 ; k<no_of_images;k++){
        std::vector<double> img;
        for(size_t i =0 ; i<28*28; i++){
            file.read((char*)(&ele),sizeof(ele));
                double normal = (((double)ele)/255);
                img.push_back(normal);
        }
        images.push_back(img);
    }
}
void flatten_image(std::string& path, std::vector<Eigen::VectorXd>& images,int& no_of_images ){
    std::ifstream file(path);
    if(!file) std::cout<<"File read failed"<<std::endl;
    unsigned char ele =1;
    int magicNumbers = 0;
    file.read((char*)&magicNumbers,sizeof(magicNumbers));
    magicNumbers = reverseInt(magicNumbers);
    file.read((char*)&no_of_images,sizeof(no_of_images));
    no_of_images = reverseInt(no_of_images);
    int rows = 0 ;
    file.read((char*)&rows,sizeof(rows));
    rows = reverseInt(rows);
    int cols = 0;
    file.read((char*)&cols,sizeof(cols));
    cols = reverseInt(cols);
    for(size_t k=0 ; k<no_of_images;k++){
        Eigen::VectorXd img(rows*cols);
        for(size_t i =0 ; i<rows*cols; i++){
            file.read((char*)(&ele),sizeof(ele));
            double normal = (((double)ele)/255);
            img(i) = normal;
        }
        images.push_back(img);
    }
}

void read_image_as_batch(std::string& path, std::vector<Eigen::MatrixXd>& images,int& no_of_images, int batch_size){
    std::ifstream file(path);
    if(!file) std::cout<<"File read failed"<<std::endl;
    unsigned char ele =1;
    int magicNumbers = 0;
    file.read((char*)&magicNumbers,sizeof(magicNumbers));
    magicNumbers = reverseInt(magicNumbers);
    file.read((char*)&no_of_images,sizeof(no_of_images));
    no_of_images = reverseInt(no_of_images);
    int rows = 0 ;
    file.read((char*)&rows,sizeof(rows));
    rows = reverseInt(rows);
    int cols = 0;
    file.read((char*)&cols,sizeof(cols));
    cols = reverseInt(cols);
    for (size_t k = 0; k < no_of_images/batch_size; k++)
    {
        
        Eigen::MatrixXd img(batch_size, rows * cols);
        for (size_t j = 0; j < batch_size; j++)
        {
            for (size_t i = 0; i < rows * cols; i++)
            {
                file.read((char *)(&ele), sizeof(ele));
                double normal = (((double)ele) / 255);
                img(j,i) = normal;
            }
        }
        images.push_back(img);
    }
}

void write_image(Eigen::MatrixXd image, std::string path){
    std::ofstream out(path);
    out<< 2;
    out<<std::endl;
    out<< image.rows();
    out<<std::endl;
    out<< image.cols();
    out<<std::endl;
    for(int i=0;i<28;i++){
        for(int j=0;j<28;j++){
            out<< image(i,j);
            out<<std::endl;
        }
    }
    
}

void read_label(std::vector<std::vector<double>>& labels, std::string path){
    std::ifstream file(path);
    if(!file) std::cout<<"File read failed"<<std::endl;
    int magicNumbers = 0;
    file.read((char*)&magicNumbers,sizeof(magicNumbers));
    magicNumbers = reverseInt(magicNumbers);
    int no_of_items = 0 ;
    file.read((char*)&no_of_items,sizeof(no_of_items));
    no_of_items = reverseInt(no_of_items);
    unsigned char ele =1;
    for(int i=0;i<no_of_items;i++){
        std::vector<double> temp_vector(10,0.0);
        file.read((char*)(&ele),sizeof(ele));
        double temp = (double)ele;
        temp_vector[temp] = 1;
        labels.push_back(temp_vector);
    }
}

void read_eigen_label(std::vector<Eigen::VectorXd>& labels, std::string path){
    std::ifstream file(path);
    if(!file) std::cout<<"File read failed"<<std::endl;
    int magicNumbers = 0;
    file.read((char*)&magicNumbers,sizeof(magicNumbers));
    magicNumbers = reverseInt(magicNumbers);
    int no_of_items = 0;
    file.read((char*)&no_of_items,sizeof(no_of_items));
    no_of_items = reverseInt(no_of_items);
    unsigned char ele = 1;
    for(int i = 0; i < no_of_items; i++) {
        Eigen::VectorXd temp_vector = Eigen::VectorXd::Zero(10);
        file.read((char*)(&ele),sizeof(ele));
        int label_index = (int)ele;
        temp_vector(label_index) = 1.0;
        labels.push_back(temp_vector);
    }
}

void read_label_as_batch(std::vector<Eigen::MatrixXd>& labels, std::string path, int batch_size){
    std::ifstream file(path);
    if(!file) std::cout<<"File read failed"<<std::endl;
    int magicNumbers = 0;
    file.read((char*)&magicNumbers,sizeof(magicNumbers));
    magicNumbers = reverseInt(magicNumbers);
    int no_of_items = 0;
    file.read((char*)&no_of_items,sizeof(no_of_items));
    no_of_items = reverseInt(no_of_items);
    unsigned char ele = 1;
    for(int i = 0; i < no_of_items/batch_size; i++) {
        Eigen::MatrixXd temp_matrix = Eigen::MatrixXd::Zero(batch_size,10);
        for(int j=0 ; j<batch_size;j++){
            file.read((char*)(&ele),sizeof(ele));
            int label_index = (int)ele;
            temp_matrix(j,label_index) = 1.0;
        }
        labels.push_back(temp_matrix);
    }
}

void write_label(std::vector<double>& label, std::string label_path){
    std::ofstream out(label_path);
    out<< 1;
    out<<std::endl;
    out<< label.size();
    out<<std::endl;
    for(auto i : label){
        out<< i;
        out<< std::endl;
    }
}

void read_config_file(std::string path, std::unordered_map<std::string, std::string>& configMap)
{
    std::cout<<"in read config func : "<<path<<std::endl;
    
    std::ifstream config(path);
    std::string line;

    if (config.is_open())
    {
        while (std::getline(config, line))
        {
            std::stringstream ss(line);
            std::string key, value;
            if (std::getline(ss, key, '=') && std::getline(ss, value))
            {
                remove_spaces(key);
                remove_spaces(value);
                configMap[key] = value;
            }
        }
        config.close();
    }
    else
    {
        std::cerr << "Unable to open the file." << std::endl;
    }
}

