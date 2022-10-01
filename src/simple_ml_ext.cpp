#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <vector>
#include <iostream>
#include <stdexcept>

namespace py = pybind11;

typedef std::vector<float> float_vector_t;
typedef std::vector<float_vector_t> float_matrix_t;
typedef std::vector<std::uint8_t> labels_t;

float_matrix_t read_matrix_2d(const float *ptr, std::size_t row_num, std::size_t col_num) {
    float_matrix_t matrix;
    for(std::size_t row_idx = 0; row_idx < row_num; ++row_idx) {
        float_vector_t row(ptr, ptr + col_num);
        matrix.push_back(row);
        ptr += col_num;
    }
    return matrix;
}

labels_t read_vector_1d(const unsigned char *ptr, std::size_t col_num) {
    labels_t row(ptr, ptr + col_num);
    return row;
}

template <typename T>
T get_minibatch(T const& input, std::size_t idx_start, std::size_t idx_end) {
    T result;
    for (auto idx = idx_start; idx < idx_end; ++idx) {
        result.push_back(input[idx]);
    }
    return result;
}

void print_matrix_sizes(float_matrix_t const& m, std::string const& name) {
    auto rows = m.size();
    auto cols = m[0].size();
    std::cout << name << ".shape = (" << rows << ", " << cols << ")" << std::endl;
}

void print_matrix_2d(float_matrix_t const& m, std::string const& name) {
    auto rows = m.size();
    auto cols = m[0].size();
    std::cout << name << ".shape = (" << rows << ", " << cols << ")" << std::endl;
    for (std::size_t i = 0; i < rows; ++i) {
        if ((i >= 3) and (i < rows-3)) {
            continue;
        }
        for (std::size_t j = 0; j < cols; ++j) {
            std::cout << "\t[" << i << "][" << j << "]=" << m[i][j] << "\t";
        }
        std::cout << std::endl;
    }
}

void print_vector_1d(labels_t const& m, std::string const& name) {
    auto cols = m.size();
    std::cout << name << ".shape = (" << cols << ")" << std::endl;
    for (std::size_t i = 0; i < cols; ++i) {
        std::cout << "\t[" << i << "] = " << static_cast<int>(m[i]) << std::endl;
    }
}

float_matrix_t multiply_matrices_2d(float_matrix_t const& a, float_matrix_t const& b) {
    std::size_t a_rows = a.size();
    std::size_t a_cols = a[0].size();
    std::size_t b_cols = b[0].size();
    float_matrix_t result(a_rows, float_vector_t(b_cols, 0));
    for (std::size_t i = 0; i < a_rows; ++i) {
        for (std::size_t j = 0; j < b_cols; ++j) {
            for (std::size_t n = 0; n < a_cols; ++n) {
                result[i][j] += (a[i][n] * b[n][j]);
            }
        }
    }
    return result;
}

float_matrix_t multiply_matrix_2d_by_scalar(float_matrix_t const& a, float scalar) {
    std::size_t a_rows = a.size();
    std::size_t a_cols = a[0].size();
    float_matrix_t result(a_rows, float_vector_t(a_cols, 0));
    for (std::size_t i = 0; i < a_rows; ++i) {
        for (std::size_t j = 0; j < a_cols; ++j) {
            result[i][j] += (a[i][j] * scalar);
        }
    }
    return result;
}

float_matrix_t subtract_matrices_2d(float_matrix_t const& a, float_matrix_t const& b) {
    std::size_t a_rows = a.size();
    std::size_t b_rows = b.size();
    std::size_t a_cols = a[0].size();
    std::size_t b_cols = b[0].size();
    if ((a_rows != b_rows) or (a_cols != b_cols)) {
        throw std::runtime_error("shape of the arguments does not match");
    }
    float_matrix_t result(a_rows, float_vector_t(a_cols, 0));
    for (std::size_t i = 0; i < a_rows; ++i) {
        for (std::size_t j = 0; j < a_cols; ++j) {
            result[i][j] += a[i][j] - b[i][j];
        }
    }
    return result;
}

float_matrix_t transpose_matrix(float_matrix_t const& values) {
    std::size_t rows = values.size();
    std::size_t cols = values[0].size();
    float_matrix_t result(cols, float_vector_t(rows, 0));
    for (std::size_t i = 0; i < rows; ++i) {
        for (std::size_t j = 0; j < cols; ++j) {
            result[j][i] = values[i][j];
        }
    }
    return result;
}

float_matrix_t exponentiate_matrix(float_matrix_t const& a) {
    std::size_t rows = a.size();
    std::size_t cols = a[0].size();
    // std::cout << "DEBUG === exponentiate_matrix(...) -> rows = " << rows << "; cols = " << cols << std::endl;
    float_matrix_t result(rows, float_vector_t(cols, 0));
    for (std::size_t i = 0; i < rows; ++i) {
        float_vector_t result_row;
        for (std::size_t j = 0; j < cols; ++j) {
            result[i][j] = exp(a[i][j]);
        }
    }
    return result;
}

// values / values.sum(axis=1)[:, np.newaxis]
float_matrix_t normalize_rows(float_matrix_t const& values) {
    std::size_t rows = values.size();
    std::size_t cols = values[0].size();
    float_matrix_t result(rows, float_vector_t(cols, 0));
    for (std::size_t i = 0; i < rows; ++i) {
        float norm_constant = 0;
        for (std::size_t j = 0; j < cols; ++j) {
            norm_constant += values[i][j];
        }
        for (std::size_t j = 0; j < cols; ++j) {
            result[i][j] = values[i][j] / norm_constant;
        }
    }
    return result;    
}

float_matrix_t one_hot(labels_t const& labels, std::size_t dims = 0) {
    std::size_t rows = labels.size();
    if (dims == 0) {
        labels_t::value_type max_label = 0;
        for (std::size_t i = 0; i < rows; ++i) {
            if (labels[i] > max_label) {
                max_label = labels[i];
            }
        }
        dims = max_label + 1;
    }
    
    std::size_t cols = dims;
    float_matrix_t result(rows, float_vector_t(cols, 0));
    for (std::size_t i = 0; i < rows; ++i) {
        auto label = labels[i];
        result[i][label] = 1;
    }
    return result;    
}

void softmax_regression_epoch_cpp(const float *X_ptr, const unsigned char *y_ptr,
								  float *theta_ptr, size_t m, size_t n, size_t k,
								  float lr, size_t batch)
{
    /**
     * A C++ version of the softmax regression epoch code.  This should run a
     * single epoch over the data defined by X and y (and sizes m,n,k), and
     * modify theta in place.  Your function will probably want to allocate
     * (and then delete) some helper arrays to store the logits and gradients.
     *
     * Args:
     *     X (const float *): pointer to X data, of size (num_examples * input_dim), stored in row
     *          major (C) format
     *     y (const unsigned char *): pointer to y data, of size num_examples
     *     theta (float *): pointer to theta data, of size (input_dim * num_classes), stored in row
     *          major (C) format
     *     m (size_t): number of examples
     *     n (size_t): input dimension
     *     k (size_t): number of classes
     *     lr (float): learning rate / SGD step size
     *     batch (int): SGD minibatch size
     *
     * Returns:
     *     (None)
     */

    // Reference code in Python:
    
    // # X.shape: (num_examples * input_dim)
    // # y.shape: num_examples
    // # theta.shape: (input_dim * num_classes)

    // num_classes = y.max() + 1
    // for i in range(0, X.shape[0], batch):
    //     # print(f"processing minibatch [{i}, {i+batch})...")
    //     minibatch_X = X[i:i+batch]
    //     minibatch_y = y[i:i+batch]
    //     logodds = np.matmul(minibatch_X, theta) # shape = (num_examples x num_classes)
    //     Z = normalize_rows(np.exp(logodds)) # shape = (num_examples x num_classes)
    //     Z_sub = (Z - one_hot(minibatch_y, dims=num_classes))  # shape = (num_examples x num_classes)
    //     theta_grad = np.matmul(minibatch_X.T, Z_sub) # (input_dim x num_classes)
    //     theta -= (lr / batch) * theta_grad

    /// BEGIN YOUR CODE

    std::size_t input_dim = n;
    std::size_t num_classes = k;
    std::size_t num_examples = m;
    std::size_t minibatch_size = batch;

    // std::cout << "=== START ===" << std::endl;

    auto X = read_matrix_2d(X_ptr, num_examples, input_dim);
    auto y = read_vector_1d(y_ptr, num_examples);
    auto theta = read_matrix_2d(theta_ptr, input_dim, num_classes);

    for (std::size_t batch_idx_start = 0; batch_idx_start < num_examples; batch_idx_start += minibatch_size) {
        auto batch_idx_end = batch_idx_start + minibatch_size;

        // std::cout << "processing minibatch [" 
        //           << batch_idx_start << ", " << batch_idx_end
        //           << ")..." << std::endl;

        auto minibatch_X = get_minibatch(X, batch_idx_start, batch_idx_end);
        auto minibatch_y = get_minibatch(y, batch_idx_start, batch_idx_end);

        // print_matrix_2d(minibatch_X, "minibatch_X");
        // print_vector_1d(minibatch_y, "minibatch_y");
        // print_matrix_2d(theta, "theta");
        
        auto logodds = multiply_matrices_2d(minibatch_X, theta);
        // print_matrix_2d(logodds, "logodds");
        
        auto logodds_exp = exponentiate_matrix(logodds);
        // print_matrix_2d(logodds_exp, "logodds_exp");
        
        auto logodds_exp_norm = normalize_rows(logodds_exp);
        // print_matrix_2d(logodds_exp_norm, "logodds_exp_norm");
        
        auto Y_onehot = one_hot(minibatch_y, num_classes);
        // print_matrix_2d(Y_onehot, "Y_onehot");
        
        auto logodds_exp_norm_sub = subtract_matrices_2d(logodds_exp_norm, Y_onehot);
        // print_matrix_2d(logodds_exp_norm_sub, "logodds_exp_norm_sub");
        
        auto minibatch_X_T = transpose_matrix(minibatch_X);
        // print_matrix_2d(minibatch_X, "minibatch_X");
        // print_matrix_2d(minibatch_X_T, "minibatch_X_T");
        
        auto theta_grad = multiply_matrices_2d(minibatch_X_T, logodds_exp_norm_sub);
        // print_matrix_2d(theta_grad, "theta_grad");
        
        auto theta_grad_scaled = multiply_matrix_2d_by_scalar(theta_grad, (lr / batch));
        // print_matrix_2d(theta_grad_scaled, "theta_grad_scaled");

        theta = subtract_matrices_2d(theta, theta_grad_scaled);
        // print_matrix_2d(theta, "theta");
    }

    // print_matrix_2d(theta, "== FINISH === final theta");

    for (std::size_t i = 0; i < theta.size(); ++i) {
        for (std::size_t j = 0; j < theta[i].size(); ++j) {
            theta_ptr[i*num_classes+j] = theta[i][j];
        }
    }
    /// END YOUR CODE
}


/**
 * This is the pybind11 code that wraps the function above.  It's only role is
 * wrap the function above in a Python module, and you do not need to make any
 * edits to the code
 */
PYBIND11_MODULE(simple_ml_ext, m) {
    m.def("softmax_regression_epoch_cpp",
    	[](py::array_t<float, py::array::c_style> X,
           py::array_t<unsigned char, py::array::c_style> y,
           py::array_t<float, py::array::c_style> theta,
           float lr,
           int batch) {
        softmax_regression_epoch_cpp(
        	static_cast<const float*>(X.request().ptr),
            static_cast<const unsigned char*>(y.request().ptr),
            static_cast<float*>(theta.request().ptr),
            X.request().shape[0],
            X.request().shape[1],
            theta.request().shape[1],
            lr,
            batch
           );
    },
    py::arg("X"), py::arg("y"), py::arg("theta"),
    py::arg("lr"), py::arg("batch"));
}
