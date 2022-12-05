#include <chrono>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <eigen3/Eigen/Eigen>
#include <algorithm>

Eigen::MatrixXd csv2mat(std::ifstream &ifs)
{
    if (!ifs.good())
    {
        throw std::runtime_error("failed to open file!");
    }

    int rows, cols;
    ifs >> rows;
    ifs >> cols;
    Eigen::MatrixXd mat(rows, cols);

    int row = 0;
    int col = 0;
    while (ifs.peek() != ifs.eof())
    {
        double x;
        ifs >> x;
        mat(row, col) = x;
        ++col;
        if (col == cols)
        {
            col = 0;
            ++row;
        }
        if (row == rows)
        {
            break;
        }
    }
    return mat;
}

class Layer
{
public:
    Layer(){};

    virtual Eigen::MatrixXd forward(Eigen::MatrixXd const &x) const = 0;
};

class Linear : public Layer
{
public:
    Linear(std::string const &A_filename, std::string const &b_filename)
    {
        std::ifstream A_file(A_filename);
        std::ifstream b_file(b_filename);
        A = csv2mat(A_file);
        b = csv2mat(b_file);
    }

    Eigen::MatrixXd forward(Eigen::MatrixXd const &x) const override
    {

            Eigen::MatrixXd y = A * x + b;
            return y;
    };

private:
    Eigen::MatrixXd A;
    Eigen::MatrixXd b;
};


class ReLU : public Layer
{
public:
    ReLU(){};

    Eigen::MatrixXd forward(Eigen::MatrixXd const &x) const override
    {

        Eigen::Matrix<double, 128, 1> R_y;

        for (int i = 0; i < x.rows(); i++)
        {
            if (x(i, 0) <= 0)
            {
                R_y(i, 0) = 0;
            }
            else
            {
                R_y(i, 0) = x(i, 0);
            }
        }
        return R_y;
    }
};

class Softmax : public Layer
{
public:
    Softmax(){};

    Eigen::MatrixXd forward(Eigen::MatrixXd const &x) const override
    {

        Eigen::MatrixXd soft = x;
        double sum_s = 0;
        for (int i = 0; i < x.rows(); i++)
        {
            double denom = exp(x(i,0));
            sum_s += denom;
        }
        for (int i = 0; i < x.rows(); i++)
        {
            soft(i,0) = exp(x(i,0))/sum_s;
            
        }
        return soft;
    }
    
};



int main(int argc, char *argv[])
{
    const Eigen::IOFormat vec_csv_format(3, Eigen::DontAlignCols, ", ", ", ");
    std::ofstream ofs("output.csv");

    // load in the weights, biases, and the data from files
    std::vector<std::string> data_filenames{"data1.csv", "data2.csv", "data3.csv", "data4.csv"};
    if (argc >= 2)
    {
        data_filenames.clear();
        for (int i{1}; i < argc; ++i)
        {
            data_filenames.push_back(argv[i]);
        }
    }

    Linear l1("A1.csv", "b1.csv");
    ReLU r;
    Linear l2("A2.csv", "b2.csv");
    Softmax s;

    for (std::string const &data_filename : data_filenames)
    {
        std::cout << "Evalua(ting " << data_filename << '\n';
        std::ifstream ifs{data_filename};
        Eigen::MatrixXd X = csv2mat(ifs);

        Eigen::MatrixXd ans = l1.forward(X);
        Eigen::MatrixXd R_ans = r.forward(ans);
        Eigen::MatrixXd l_ans = l2.forward(R_ans);
        Eigen::MatrixXd probabilities = s.forward(l_ans);

        ofs << probabilities.format(vec_csv_format) << std::endl;
    }
}
