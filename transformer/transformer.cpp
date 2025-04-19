#include <bits/stdc++.h>
#include <Eigen/Dense>
using namespace std;
using namespace chrono;
using namespace Eigen;

/*

_________________________________________________________________________________________________________

This is the C++ implementation of Transformer.

IDE: CLion (Downloaded using my Queen's email)


This is entirely for learning. There is no emphasis on test accuracy or optimisation. Implementation
of Transformer from scratch without resorting to any library such as PyTorch or Tensorflow is the aim here.

_________________________________________________________________________________________________________

*/


mt19937 rng(steady_clock::now().time_since_epoch().count());
normal_distribution normal_dist(0.0, .01);

using MatrixVariant = variant<MatrixXd, MatrixXi>;

struct Tensor {
    vector<MatrixVariant> data;

    // Product of two tensors
    Tensor operator*(const Tensor& tensor) const {
        const int batch_size = data.size();

        Tensor res_tensor;

        for(int i = 0; i < batch_size; i++) {
            MatrixXd mat1 = get<MatrixXd>(data[i]);
            MatrixXd mat2 = get<MatrixXd>(tensor.data[i]);
            MatrixXd res = mat1 * mat2;
            res_tensor.data.emplace_back(res);
        }

        return res_tensor;
    }

    // Product of a tensor and a matrix
    Tensor operator*(const MatrixXd& matrix) const {
        const int batch_size = data.size();

        Tensor res_tensor;

        for(int i = 0; i < batch_size; i++) {
            MatrixXd res = get<MatrixXd>(data[i]) * matrix;
            res_tensor.data.emplace_back(res);
        }
        return res_tensor;
    }

    // Addition of two tensors of same dimension
    Tensor operator+(const Tensor& tensor) const {
        const int batch_size = data.size();

        Tensor res_tensor;

        for(int i = 0; i < batch_size; i++) {
            MatrixXd res = get<MatrixXd>(data[i]) + get<MatrixXd>(tensor.data[i]);
            res_tensor.data.emplace_back(res);
        }
        return res_tensor;
    }

    // Addition of a tensor and a matrix
    Tensor operator+(const MatrixXd& matrix) const {
        const int batch_size = data.size();

        Tensor res_tensor;

        for(int i = 0; i < batch_size; i++) {
            MatrixXd res = get<MatrixXd>(data[i]) + matrix;
            res_tensor.data.emplace_back(res);
        }
        return res_tensor;
    }

    // Addition of a tensor and a scalar
    Tensor operator+(const double& number) const {
        const int batch_size = data.size();

        Tensor res_tensor;

        for(int i = 0; i < batch_size; i++) {
            MatrixXd res = get<MatrixXd>(data[i]).array() + number;
            res_tensor.data.emplace_back(res);
        }
        return res_tensor;
    }

    // Divide the tensor by a scalar
    Tensor operator/(const double& number) const {
        const int batch_size = data.size();

        Tensor res_tensor;

        for(int i = 0; i < batch_size; i++) {
            MatrixXd res = get<MatrixXd>(data[i]).array() / number;
            res_tensor.data.emplace_back(res);
        }
        return res_tensor;
    }

    // Transpose all the matrices in a tensor
    Tensor transpose() const {
        Tensor transposedTensor;

        for(auto &mat: data) {
            MatrixXd transposed = get<MatrixXd>(mat).transpose();
            transposedTensor.data.emplace_back(transposed);
        }
        return transposedTensor;
    }

    // Print the tensor
    void print() const {
        for(auto& var: data) {
            cout << get<MatrixXd>(var) << "\n\n";
        }
    }

    // Print the shape of the tensor
    void printShape() const {
        cout << data.size() << "  " << get<MatrixXd>(data[0]).rows() << "  " << get<MatrixXd>(data[0]).cols() << "\n";
    }

    // Get the shape of the tensor
    tuple<int, int, int> shape() const {
        return make_tuple(data.size(), get<MatrixXd>(data[0]).rows(), get<MatrixXd>(data[0]).cols());
    }
};

Tensor softmax(const Tensor& x) {

    /*
     Returns softmax of the given tensor along the last dimension.

     @Params:
     x: (batch_size, max_sequence_length, columns)

     @Returns:
     result: (batch_size, max_sequence_length, columns)
    */

    const int batch_size = get<0>(x.shape());
    const int max_sequence_length = get<1>(x.shape());
    const int columns = get<2>(x.shape());

    Tensor probabilities;

    for(int i = 0; i < batch_size; i++) {

        auto mat = get<MatrixXd>(x.data[i]);
        MatrixXd probabilities_mat(max_sequence_length, columns);

        for(int j = 0; j < max_sequence_length; j++) {

            const double max_value = mat.row(j).maxCoeff();

            MatrixXd exp_values(1, columns);

            for(int k = 0; k < columns; k++) {
                exp_values(0, k) = exp(mat(j, k) - max_value);
            }

            const double sum_exp_values = exp_values.sum();

            for(int k = 0; k < columns; k++) {
                probabilities_mat(j, k) = exp_values(0, k) / sum_exp_values;
            }
        }
        probabilities.data.emplace_back(probabilities_mat);
    }

    return probabilities;
}

double cross_entropy_loss(const Tensor& probabilities, const MatrixXi& target) {

    /*
    Computes the sequence-to-sequence loss.

    @Params:
    probabilities: (batch_size, max_sequence_length, target_vocab_size) - predicted scores
    target: (batch_size, max_sequence_length) - actual token indices

    @Returns:
    loss: Scalar loss value
    */

    const int batch_size = get<0>(probabilities.shape());
    const int max_sequence_length = get<1>(probabilities.shape());

    vector correct_probabilities(batch_size, vector(max_sequence_length, 0.0));

    for(int i = 0; i < batch_size; i++) {
        auto mat = get<MatrixXd>(probabilities.data[i]);
        for(int j = 0; j < max_sequence_length; j++) {
            correct_probabilities[i][j] = mat(j, target(i, j));
        }
    }

    double loss = 0;

    for(int i = 0; i < batch_size; i++) {
        for(int j = 0; j < max_sequence_length; j++) {
            loss += -log(correct_probabilities[i][j]);
        }
    }

    loss /= (batch_size * max_sequence_length);

    return loss;
}

Tensor cross_entropy_gradient(const Tensor& probabilities, const MatrixXi& target) {

    /*
    Calculates the gradient of each probability.

    @Params:
    probabilities: (batch_size, max_sequence_length, target_vocab_size) - predicted scores
    target: (batch_size, max_sequence_length) - actual token indices

    @Returns:
    gradients: (batch_size, max_sequence_length, target_vocab_size)
    */

    const int batch_size = get<0>(probabilities.shape());
    const int max_sequence_length = get<1>(probabilities.shape());
    const int vocab_size = get<2>(probabilities.shape());

    Tensor gradient;

    for(int i = 0; i < batch_size; i++) {

        MatrixXd probabilities_mat = get<MatrixXd>(probabilities.data[i]);
        MatrixXd gradient_i(max_sequence_length, vocab_size);

        for(int j = 0; j < max_sequence_length; j++) {
            gradient_i.row(j) = probabilities_mat.row(j);
            gradient_i(j, target(i, j)) -= 1;
        }
        gradient.data.emplace_back(gradient_i);
    }

    return gradient / batch_size;
}


MatrixXd randomUniformMatrix(int rows, int cols, double min, double max) {
    uniform_real_distribution dis(min, max);

    MatrixXd mat(rows, cols);

    for(int i = 0; i < rows; i++) {
        for(int j = 0; j < cols; j++) {
            mat(i, j) = dis(rng);
        }
    }
    return mat;
}


class Embedding {
public:
    int vocab_size;
    int d_model;
    MatrixXd embeddings;
    MatrixXi input;  // shape: (batch_size, max_sequence_length)

    Embedding(const int vocab_size, const int d_model) : vocab_size(vocab_size), d_model(d_model) {
        embeddings = randomUniformMatrix(vocab_size, d_model, 0, 1);
    }

    Tensor forward(const Tensor& x) {

        /*
        @Params:
        x: (1, batch_size, max_sequence_length)

        @Returns:
        output: (batch_size, max_sequence_length, d_model)
        */

        Tensor output;

        // Check if the tensor x contains expected type
        if(const MatrixXi* pIndices = get_if<MatrixXi>(&x.data[0])) {

            // Let's call it indices because it contains the indices of the tokens in the vocabulary
            const MatrixXi& indices = *pIndices;

            // Cache the indices for backpropagation
            this->input = indices;
            const int batch_size = indices.rows();
            const int max_sequence_length = indices.cols();

            vector<MatrixXd> embedding_matrices(batch_size);

            for(int i = 0; i < batch_size; i++) {
                MatrixXd emb_i(max_sequence_length, d_model);
                for(int j = 0; j < max_sequence_length; j++) {
                    // Get the index for the word in the vocabulary
                    const int token = indices(i, j);

                    //  Get the corresponding embedding
                    emb_i.row(j) = embeddings.row(token);
                }
                embedding_matrices[i] = emb_i;
            }

            // Loop through the vector of embedding matrices and add them to the output tensor
            for(const auto& m: embedding_matrices) {
                output.data.emplace_back(m);
            }
        }
        else {
            throw runtime_error("Tensor does not contain MatrixXi as expected.");
        }

        return output;  // shape: (batch_size, max_sequence_length, d_model)
    }

    Tensor backward(const Tensor& gradient_output, const double learning_rate) {

        /*
        Update embedding vectors using gradient.

        @Params:
        gradient_output: (batch_size, max_sequence_length, d_model)
        learning_rate: float
        */

        // Convert gradient_output Tensor to a vector of MatrixXd
        vector<MatrixXd> gradient_matrices;

        for(const auto& var: gradient_output.data) {
            // Use get to extract the MatrixXd
            try {
                const MatrixXd& mat = get<MatrixXd>(var);
                gradient_matrices.emplace_back(mat);
            }
            catch(const bad_variant_access&) {
                throw runtime_error("Expected MatrixXd in gradient_output Tensor in Embedding layer backward.");
            }
        }
        const int batch_size = gradient_matrices.size();
        // Each matrix in gradient_matrices has shape: (max_sequence_length, d_model)
        const int max_sequence_length = gradient_matrices[0].rows();

        // Collect unique tokens from the saved input indices
        set<int> unique_tokens;
        for (int i = 0; i < batch_size; i++) {
            for (int j = 0; j < max_sequence_length; j++) {
                unique_tokens.insert(input(i, j));
            }
        }

        // Iterate over each unique token to accumulate gradients and update embeddings
        for (const int token : unique_tokens) {
            VectorXd sum_gradient = VectorXd::Zero(d_model);
            for (int i = 0; i < batch_size; i++) {
                for (int j = 0; j < max_sequence_length; j++) {
                    if (input(i, j) == token) {
                        // gradient_matrices[i].row(j) is a row vector; transpose it to add as a column vector.
                        sum_gradient += gradient_matrices[i].row(j).transpose();
                    }
                }
            }
            // Update the embedding for the current token
            // (Assuming embeddings rows correspond to tokens.)
            for (int i = 0; i < d_model; i++) {
                embeddings(token, i) -= learning_rate * sum_gradient(i);
            }
        }
        Tensor tensor;
        return tensor;
    }
};

MatrixXd positional_encoding(const int length, const int depth) {

    const int half_depth = depth / 2;

    MatrixXd positions(length, 1);
    for(int i = 0; i < length; i++) {
        positions(i, 0) = i;
    }

    MatrixXd depths(1, half_depth);
    for(int i = 0; i < half_depth; i++) {
        depths(0, i) = i;
        depths(0, i) /= half_depth;
    }

    MatrixXd angle_rates(1, half_depth);
    for(int i = 0; i < half_depth; i++) {
        angle_rates(0, i) = 1.0 / pow(10000, depths(0, i));
    }

    MatrixXd angle_rads = positions * angle_rates;  // (length, half_depth)

    MatrixXd pos_encoding(length, depth);

    for(int i = 0; i < length; i++) {
        for(int j = 0; j < half_depth; j++) {
            pos_encoding(i, j) = sin(angle_rads(i, j));
            pos_encoding(i, j + half_depth) = cos(angle_rads(i, j));
        }
    }
    return pos_encoding;
}

class PositionalEmbedding {
public:

    int vocab_size;
    int d_model;
    Embedding embedding;
    MatrixXd pos_encoding;

    PositionalEmbedding(const int vocab_size, const int d_model) : vocab_size(vocab_size), d_model(d_model), embedding(Embedding(vocab_size, d_model)) {
        pos_encoding = positional_encoding(2048, d_model);
    }

    Tensor forward(const Tensor& x) {
        /*
        @Params
        x: (1, batch_size, max_sequence_length)

        @Returns
        output: (batch_size, max_sequence_length, d_model)
        */

        Tensor output;

        if(const MatrixXi* pIndices = get_if<MatrixXi>(&x.data[0])) {
            // Let's call it indices because it contains the indices of the tokens in the vocabulary
            const MatrixXi& indices = *pIndices;

            const int length = indices.cols();

            output = embedding.forward(x);

            for(auto& mat_variant: output.data) {
                auto& mat = get<MatrixXd>(mat_variant);
                mat *= sqrt(d_model);

                mat += pos_encoding.topRows(length);
            }
        }
        else {
            throw runtime_error("Tensor does not contain MatrixXi as expected.");
        }

        return output;  // shape: (batch_size, max_sequence_length, d_model)
    }

    Tensor backward(const Tensor& gradient_output, const double learning_rate) {
        /*
        @Param:
        gradient_output: (batch_size, max_sequence_length, d_model)
        learning_rate: float
        */

        vector<MatrixXd> gradient_matrices;

        for(const auto& mat_variant: gradient_output.data) {
            // Use get to extract the MatrixXd
            try {
                const auto& mat = get<MatrixXd>(mat_variant);

                // Scale the gradient to get the gradient of embedding
                gradient_matrices.emplace_back(mat * sqrt(d_model));
            }
            catch(const bad_variant_access&) {
                throw runtime_error("Expected MatrixXd in gradient_output Tensor in Embedding layer backward.");
            }
        }

        embedding.backward(gradient_output, learning_rate);
        return gradient_output;
    }
};

class ScaledDotProductAttention {
public:
    Tensor query;
    Tensor key;
    Tensor value;
    Tensor Q;
    Tensor K;
    Tensor V;
    Tensor A;
    int d_model;
    int d;

    MatrixXd WQ;
    MatrixXd WK;
    MatrixXd WV;

    ScaledDotProductAttention(const int d_model, const int d) : d_model(d_model), d(d) {
        // Glorot initialisation
        const double limit = sqrt(6.0 / (d_model + d));
        WQ = randomUniformMatrix(d_model, d, -limit, limit);
        WK = randomUniformMatrix(d_model, d, -limit, limit);
        WV = randomUniformMatrix(d_model, d, -limit, limit);
    }

    Tensor forward(const Tensor& query, const Tensor& key, const Tensor& value, const bool use_causal_mask=false) {
        /*
        @Params
        query: (batch_size, max_sequence_length, d_model)
        key: (batch_size, max_sequence_length, d_model)
        value: (batch_size, max_sequence_length, d_model)

        @Returns
        output: (batch_size, max_sequence_length, d)
        */
        this->query = query;
        this->key = key;
        this->value = value;

        this->Q.data.clear();
        this->K.data.clear();
        this->V.data.clear();
        this->A.data.clear();

        const int batch_size = get<0>(query.shape());
        const int max_sequence_length = get<1>(query.shape());

        for(int i = 0; i < batch_size; i++) {
            auto& mat_q = get<MatrixXd>(query.data[i]);  // shape: (max_sequence_length, d)
            auto& mat_k = get<MatrixXd>(key.data[i]);  // shape: (max_sequence_length, d)
            auto& mat_v = get<MatrixXd>(value.data[i]);  // shape: (max_sequence_length, d)

            MatrixXd res = mat_q * this->WQ;  // shape: (max_sequence_length, d)
            this->Q.data.emplace_back(res);

            res = mat_k * this->WK;  // shape: (max_sequence_length, d)
            this->K.data.emplace_back(res);

            res = mat_v * this->WV;  // shape: (max_sequence_length, d)
            this->V.data.emplace_back(res);
        }

        Tensor QK = this->Q * this->K.transpose();  // shape: (batch_size, max_sequence_length, max_sequence_length)
        QK = QK / sqrt(this->d);

        if(use_causal_mask) {
            MatrixXd mask = MatrixXd::Constant(max_sequence_length, max_sequence_length, -1e9);
            mask = mask.triangularView<StrictlyUpper>();
            QK = QK + mask;
        }

        this->A = softmax(QK);  // shape: (batch_size, max_sequence_length, d)

        Tensor output = this->A * this->V;
        return output;  // shape: (batch_size, max_sequence_length, d)
    }

    vector<Tensor> backward(const Tensor& gradient_output, const double learning_rate) {
        /*
        @Params:
        gradient_output: (batch_size, max_sequence_length, d)
        learning_rate: float

        @Returns
        gradient_query, gradient_key, gradient_value: (batch_size, max_sequence_length, d_model)
        */

        double scale = 1 / sqrt(this->d);
        int batch_size = get<0>(gradient_output.shape());

        Tensor gradient_V = this->A.transpose() * gradient_output;  // shape: (batch_size, max_sequence_length, d)

        Tensor gradient_A = gradient_output * this->V.transpose(); // shape: (batch_size, max_sequence_length, max_sequence_length)

        // For softmax, derivative is: dZ (dQK) = A * (dA - sum(dA * A, axis=-1, keepdims=True))
        Tensor sum_gradient_A_A;  // shape: (batch_size, max_sequence_length, 1)

        for(int i = 0; i < batch_size; i++) {
            MatrixXd sum = (get<MatrixXd>(gradient_A.data[i]).array() * get<MatrixXd>(this->A.data[i]).array()).rowwise().sum();  // shape: (max_sequence_length, 1)
            sum_gradient_A_A.data.emplace_back(sum);
        }

        Tensor gradient_QK;  // shape: (batch_size, max_sequence_length, max_sequence_length)

        for(int i = 0; i < batch_size; i++) {
            MatrixXd mat_A = get<MatrixXd>(this->A.data[i]);  // shape: (max_sequence_length, max_sequence_length)
            MatrixXd mat_gradient_A = get<MatrixXd>(gradient_A.data[i]);  // shape: (max_sequence_length, max_sequence_length)
            MatrixXd mat_sum_gradient_A_A = get<MatrixXd>(sum_gradient_A_A.data[i]).replicate(1, mat_gradient_A.cols()); // shape: (max_sequence_length, max_sequence_length)

            MatrixXd mat = mat_A.array() * (mat_gradient_A - mat_sum_gradient_A_A).array(); // shape: (max_sequence_length, max_sequence_length)
            mat *= scale;
            gradient_QK.data.emplace_back(mat);
        }

        Tensor gradient_Q = gradient_QK * this->K;  // shape: (batch_size, max_sequence_length, d)
        Tensor gradient_K = gradient_QK.transpose() * this->Q;  // shape: (batch_size, max_sequence_length, d)

        MatrixXd gradient_WQ = MatrixXd::Zero(this->d_model, this->d);  // shape: (d_model, d)
        for(int i = 0; i < batch_size; i++) {
            gradient_WQ += get<MatrixXd>(this->query.data[i]).transpose() * get<MatrixXd>(gradient_Q.data[i]);
        }
        Tensor gradient_query = gradient_Q * this->WQ.transpose();  // shape: (batch_size, max_sequence_length, d_model)

        MatrixXd gradient_WK = MatrixXd::Zero(this->d_model, this->d);  // shape: (d_model, d)
        for(int i = 0; i < batch_size; i++) {
            gradient_WK += get<MatrixXd>(this->key.data[i]).transpose() * get<MatrixXd>(gradient_K.data[i]);
        }
        Tensor gradient_key = gradient_K * this->WK.transpose();  // shape: (batch_size, max_sequence_length, d_model)

        MatrixXd gradient_WV = MatrixXd::Zero(this->d_model, this->d);  // shape: (d_model, d)
        for(int i = 0; i < batch_size; i++) {
            gradient_WV += get<MatrixXd>(this->value.data[i]).transpose() * get<MatrixXd>(gradient_V.data[i]);
        }
        Tensor gradient_value = gradient_V * this->WV.transpose();  // shape: (batch_size, max_sequence_length, d_model)

        this->WQ -= learning_rate * gradient_WQ;
        this->WK -= learning_rate * gradient_WK;
        this->WV -= learning_rate * gradient_WV;

        return {gradient_query, gradient_key, gradient_value};
    }
};

class MultiHeadAttention {
public:

    int n_heads;
    int d_model;
    int head_dim;
    vector<ScaledDotProductAttention> attention_heads;

    MatrixXd W;
    Tensor concatenated_output;

    MultiHeadAttention(const int n_heads, const int d_model) : n_heads(n_heads), d_model(d_model) {
        head_dim = d_model / n_heads;
        for(int i = 0; i < n_heads; i++) {
            attention_heads.emplace_back(d_model, head_dim);
        }

        double limit = sqrt(6.0 / (d_model + d_model));

        W = randomUniformMatrix(d_model, d_model, -limit, limit);
    }

    Tensor forward(const Tensor& query, const Tensor& key, const Tensor& value, const bool use_causal_mask = false) {
        /*
        @Params
        query, key, value: (batch_size, max_sequence_length, d_model)

        @Returns
        output: (batch_size, max_sequence_length, d_model)
        */

        const int batch_size = get<0>(query.shape());
        const int max_sequence_length = get<1>(query.shape());

        vector<Tensor> head_outputs;

        for(auto& head: attention_heads) {
            head_outputs.emplace_back(head.forward(query, key, value, use_causal_mask));  // Each head output: (batch_size, max_sequence_length, head_dim)
        }

        // Iterate over all the sequences in the batch
        for(int i = 0; i < batch_size; i++) {
            MatrixXd mat(max_sequence_length, this->d_model);

            for(int j = 0; j < this->n_heads; j++) {
                mat.block(0, j * head_dim, max_sequence_length, head_dim) = get<MatrixXd>(head_outputs[j].data[i]);
            }
            this->concatenated_output.data.emplace_back(mat);
        }

        // concatenated_output shape: (batch_size, max_sequence_length, d_model)

        Tensor final_output = this->concatenated_output * this->W;  // shape: (batch_size, max_sequence_length, d_model)

        return final_output;
    }

    vector<Tensor> backward(const Tensor& gradient_output, const double learning_rate) {
        /*
        @Params
        gradient_output: (batch_size, max_sequence_length, d_model)

        @Returns
        gradient_query, gradient_key, gradient_value: (batch_size, max_sequence_length, d_model)
        */

        const int batch_size = get<0>(gradient_output.shape());
        const int max_sequence_length = get<1>(gradient_output.shape());

        // During forward pass, same weight matrix was applied for all the instances in a batch.
        // Therefore, during backward pass gradients of all the instances should be used to calculate
        // the gradient of the weight matrix. To perform that operation, we convert concatenated_output
        // and gradient_output, which are 3D tensors, to 2D matrices.

        MatrixXd concatenated_output_2d(batch_size * max_sequence_length, this->d_model);
        MatrixXd gradient_output_2d(batch_size * max_sequence_length, this->d_model);

        for(int i = 0; i < batch_size; i++) {
            MatrixXd mat = get<MatrixXd>(this->concatenated_output.data[i]);
            concatenated_output_2d.block(i * max_sequence_length, 0, max_sequence_length, d_model) = mat;

            mat = get<MatrixXd>(gradient_output.data[i]);
            gradient_output_2d.block(i * max_sequence_length, 0, max_sequence_length, d_model) = mat;
        }

        MatrixXd gradient_weights = concatenated_output_2d.transpose() * gradient_output_2d;  // shape: (d_model, d_model)

        Tensor gradient_concatenated_output = gradient_output * this->W.transpose();  // shape: (batch_size, max_sequence_length, d_model)

        Tensor gradient_query;
        Tensor gradient_key;
        Tensor gradient_value;

        for(int i = 0; i < batch_size; i++) {
            MatrixXd mat = MatrixXd::Zero(max_sequence_length, this->d_model);
            gradient_query.data.emplace_back(mat);
            gradient_key.data.emplace_back(mat);
            gradient_value.data.emplace_back(mat);
        }

        for(int i = 0; i < n_heads; i++) {
            Tensor gradient_head;

            // Get the appropriate gradient of the attention head currently being processed over the entire batch
            for(int j = 0; j < batch_size; j++) {
                MatrixXd mat = get<MatrixXd>(gradient_concatenated_output.data[j]).block(0, i * head_dim, max_sequence_length, head_dim);
                gradient_head.data.emplace_back(mat);
            }

            // List of gradient_query, gradient_key, and gradient_value
            auto gradient_list = this->attention_heads[i].backward(gradient_head, learning_rate);

            gradient_query = gradient_query + gradient_list[0];
            gradient_key = gradient_key + gradient_list[1];
            gradient_value = gradient_value + gradient_list[2];
        }

        this->W -= learning_rate * gradient_weights;

        return {gradient_query, gradient_key, gradient_value};
    }
};

class Add {
public:
    Tensor forward(vector<Tensor> x) {
        /*
        @Params
        x: a list of two matrices of shape (batch_size, max_sequence_length, d_model)
        */
        return x[0] + x[1];
    }

    vector<Tensor> backward(Tensor gradient_output, double learning_rate) {
        /*
        This function passes the same gradient it receives. It's not even necessary.
        Just added for consistency.

        @Params
        gradient_output: (batch_size, max_sequence_length, d_model)

        @Returns
        gradients: a list of gradients corresponding to x[0] and x[1]
        */

        return {gradient_output, gradient_output};
    }
};

class LayerNormalization {
public:
    MatrixXd g;
    MatrixXd b;
    Tensor normalised_input;  // shape: (batch_size, max_sequence_length, d_model)
    Tensor standard_deviation;  // shape: (batch_size, max_sequence_length, 1)
    int d_model;
    double epsilon;

    LayerNormalization(const int d_model, const double epsilon = 1e-6) : d_model(d_model), epsilon(epsilon) {
        g = MatrixXd::Ones( 1, d_model);
        b = MatrixXd::Zero(1, d_model);
    }

    Tensor forward(const Tensor& x) {
        /*
        @Params
        x: (batch_size, max_sequence_length, d_model)

        @Returns
        output: (batch_size, max_sequence_length, d_model)
        */

        const int batch_size = get<0>(x.shape());

        Tensor mean;
        Tensor output;

        for(int i = 0; i < batch_size; i++) {
            auto& mat = get<MatrixXd>(x.data[i]);  // shape: (max_sequence_length, d_model)
            MatrixXd mean_mat = mat.rowwise().mean();  // shape: (batch_size, max_sequence_length, 1)
            mean.data.emplace_back(mean_mat);

            MatrixXd centred_mat = mat - mean_mat.replicate(1, mat.cols());  // shape: (max_sequence_length, d_model)

            MatrixXd variance_mat = (centred_mat.array().square().matrix()).rowwise().mean();  // shape: (max_sequence_length, 1)

            MatrixXd standard_deviation_mat = (variance_mat.array() + this->epsilon).sqrt();  // shape: (max_sequence_length, 1)

            this->standard_deviation.data.emplace_back(standard_deviation_mat);

            MatrixXd normalised_mat = ((mat - mean_mat.replicate(1, mat.cols())).array()
                    / (standard_deviation_mat.replicate(1, mat.cols())).array()).matrix();  // shape: (max_sequence_length, d_model)

            this->normalised_input.data.emplace_back(normalised_mat);

            MatrixXd output_mat = (this->g.replicate(mat.rows(), 1).array()
                    * normalised_mat.array() + this->b.replicate(mat.rows(), 1).array()).matrix();  // shape: (max_sequence_length, d_model)
            output.data.emplace_back(output_mat);
        }

        return output;
    }

    Tensor backward(const Tensor& gradient_output, const double learning_rate) {
        /*
        @Params
        gradient_output: (batch_size, max_sequence_length, d_model)

        @Returns
        gradient_input: (batch_size, max_sequence_length, d_model)
        */

        const int batch_size = get<0>(gradient_output.shape());

        MatrixXd gradient_g = MatrixXd::Zero(1, d_model);
        MatrixXd gradient_b = MatrixXd::Zero(1, d_model);

        Tensor gradient_normalised_input;  // shape: (batch_size, max_sequence_length, d_model)

        Tensor gradient_input;

        for(int i = 0; i < batch_size; i++) {
            MatrixXd gradient_output_mat = get<MatrixXd>(gradient_output.data[i]);  // shape: (max_sequence_length, d_model)

            MatrixXd normalised_input_mat = get<MatrixXd>(this->normalised_input.data[i]);  // shape: (max_sequence_length, d_model)

            MatrixXd gradient_g_mat = gradient_output_mat.array() * normalised_input_mat.array();  // shape: (max_sequence_length, d_model)
            gradient_g += gradient_g_mat.colwise().sum();
            gradient_b += gradient_output_mat.colwise().sum();

            // Following the formula for the gradient of layer norm:
            // d_x = (1/σ) * (d_normalized - mean(d_normalized) - normalized * mean(d_normalized * normalized))

            MatrixXd gradient_normalised_input_mat = (gradient_output_mat.array() *
                this->g.replicate(gradient_output_mat.rows(), 1).array()).matrix();  // shape: (max_sequence_length, d_model)
            gradient_normalised_input.data.emplace_back(gradient_normalised_input_mat);

            MatrixXd mean_normalised_input_mat = gradient_normalised_input_mat.rowwise().mean();  // shape: (max_sequence_length, 1)

            MatrixXd mean_gradient_normalised_input_normalised_mat = (gradient_normalised_input_mat.array()
                * normalised_input_mat.array()).matrix().rowwise().mean();  // shape: (max_sequence_length, 1)

            MatrixXd standard_deviation_mat = get<MatrixXd>(this->standard_deviation.data[i]);  // shape: (max_sequence_length, 1)

            MatrixXd gradient_input_mat = standard_deviation_mat.array().inverse().matrix().replicate(1, gradient_output_mat.cols()).array()
                * (gradient_normalised_input_mat.array()
                - mean_normalised_input_mat.replicate(1, gradient_output_mat.cols()).array()
                - normalised_input_mat.array()
                * mean_gradient_normalised_input_normalised_mat.replicate(1, gradient_output_mat.cols()).array());

            gradient_input.data.emplace_back(gradient_input_mat);  // gradient_input_mat shape: (max_sequence_length, d_model)
        }

        this->g -= learning_rate * gradient_g;
        this->b -= learning_rate * gradient_b;

        return gradient_input;  // gradient_input shape: (batch_size, max_sequence_length, d_model)
    }
};

class Linear {
public:
    int input_channels;
    int output_channels;        // number of output units
    MatrixXd weights, biases;          // W: (hidden_size x input_size), b: (hidden_size x 1)
    Tensor input;         // Stored input from forward pass (shape: (batch_size, input_size))

    Linear(int input_channels, int output_channels) : input_channels(input_channels), output_channels(output_channels) {
        double limit = sqrt(6.0 / (input_channels + output_channels));

        weights = randomUniformMatrix(input_channels, output_channels, -limit, limit);
        biases = MatrixXd::Zero(1, output_channels);
    }

    Tensor forward(const Tensor& x) {
        /*
        @Params
        x: (batch_size, max_sequence_length, input_channels)

        @Returns
        output: (batch_size, max_sequence_length, output_channels)
        */

        const int max_sequence_length = get<1>(x.shape());

        this->input = x;
        Tensor output = x * weights + biases.replicate(max_sequence_length, 1);
        return output;
    }

    Tensor backward(const Tensor& gradient_output, const double learning_rate) {
        /*
        @Params
        gradient_output: (batch_size, max_sequence_length, output_channels)

        @Returns
        gradient_input: (batch_size, max_sequence_length, input_channels)
        */

        const int batch_size = get<0>(gradient_output.shape());
        const int max_sequence_length = get<1>(gradient_output.shape());

        Tensor gradient_input = gradient_output * weights.transpose();

        // During forward pass, same weight matrix was applied for all the instances in a batch.
        // Therefore, during backward pass gradients of all the instances should be used to calculate
        // the gradient of the weight matrix. To perform that operation, we convert concatenated_output
        // and gradient_output, which are 3D tensors, to 2D matrices.

        MatrixXd gradient_output_2d(batch_size * max_sequence_length, this->output_channels);
        MatrixXd input_2d(batch_size * max_sequence_length, this->input_channels);

        for(int i = 0; i < batch_size; i++) {
            MatrixXd mat = get<MatrixXd>(gradient_output.data[i]);
            gradient_output_2d.block(i * max_sequence_length, 0, max_sequence_length, output_channels) = mat;

            mat = get<MatrixXd>(this->input.data[i]);
            input_2d.block(i * max_sequence_length, 0, max_sequence_length, input_channels) = mat;
        }

        MatrixXd gradient_weights = (gradient_output_2d.transpose() * input_2d).transpose();  // shape: (input_channels, output_channels)
        MatrixXd gradient_biases = gradient_output_2d.colwise().sum();  // shape: (1, output_channels)

        this->weights -= learning_rate * gradient_weights;
        this->biases -= learning_rate * gradient_biases;

        return gradient_input;
    }
};

class FeedForward {
public:
    int d_model;
    int d_ff;
    Linear linear_1;
    Linear linear_2;
    Add add;
    LayerNormalization layer_normalisation;
    vector<vector<vector<bool>>> relu_mask;

    FeedForward(const int d_model, const int d_ff) :
        d_model(d_model),
        d_ff(d_ff),
        linear_1(d_model, d_ff),
        linear_2(d_ff, d_model),
        layer_normalisation(d_model) {}

    Tensor forward(const Tensor& x) {
        /*
        @Params
        x: (batch_size, max_sequence_length, d_model)

        @Returns
        output: (batch_size, max_sequence_length, d_model)
        */

        const int batch_size = get<0>(x.shape());
        const int max_sequence_length = get<1>(x.shape());

        relu_mask = vector(batch_size, vector(max_sequence_length, vector<bool>(d_ff)));

        Tensor x_normalised = this->layer_normalisation.forward(x);  // shape: (batch_size, max_sequence_length, d_model)
        Tensor output1 = this->linear_1.forward(x_normalised);  // shape: (batch_size, max_sequence_length, d_ff)
        Tensor relu_out;

        for(int i = 0; i < batch_size; i++) {
            auto& output1_mat = get<MatrixXd>(output1.data[i]);  // shape: (max_sequence_length, d_ff)
            MatrixXd relu_out_mat(max_sequence_length, this->d_ff);

            for(int j = 0; j < max_sequence_length; j++) {
                for(int k = 0; k < this->d_ff; k++) {
                    relu_mask[i][j][k] = output1_mat(j, k) > 0;  // Cache mask for ReLU
                    relu_out_mat(j, k) = max(0.0, output1_mat(j, k));  // Apply ReLU
                }
            }
            relu_out.data.emplace_back(relu_out_mat);
        }

        Tensor output2 = this->linear_2.forward(relu_out);  // shape: (batch_size, max_sequence_length, d_model)
        Tensor add_output = this->add.forward({x, output2}); // shape: (batch_size, max_sequence_length, d_model)
        return add_output;
    }

    Tensor backward(const Tensor& gradient_output, const double learning_rate) {
        /*
        @Params
        gradient_output: (batch_size, max_sequence_length, d_model)

        @Returns
        gradient_input: (batch_size, max_sequence_length, d_model)
        */

        // Backprop through the addition layer: it splits the gradient equally to both inputs.
        const auto gradient_list = this->add.backward(gradient_output, learning_rate);
        const Tensor& gradient_x_residual = gradient_list[0];  // each of shape: (batch_size, max_sequence_length, d_model)
        const Tensor& gradient_output2 = gradient_list[1];  // each of shape: (batch_size, max_sequence_length, d_model)

        // Backprop through Dense2.
        // gradient_output2 is gradient with respect to Dense2 output.
        // linear_2.backward returns gradient with respect to its input (i.e., relu_out)
        Tensor gradient_relu = this->linear_2.backward(gradient_output2, learning_rate);  // shape: (batch_size, max_sequence_length, d_ff)

        // Backprop through ReLU.
        // The derivative of ReLU is 1 for positive out1, 0 otherwise.

        const int batch_size = get<0>(gradient_output.shape());
        const int max_sequence_length = get<1>(gradient_output.shape());

        Tensor gradient_output1;

        for(int i = 0; i < batch_size; i++) {
            const MatrixXd& gradient_relu_mat = get<MatrixXd>(gradient_relu.data[i]);
            MatrixXd gradient_output1_mat(max_sequence_length, d_ff);
            for(int j = 0; j < max_sequence_length; j++) {
                for(int k = 0; k < this->d_ff; k++) {
                    gradient_output1_mat(j, k) = gradient_relu_mat(j, k) * relu_mask[i][j][k];
                }
            }
            gradient_output1.data.emplace_back(gradient_output1_mat);
        }

        // Backprop through Dense1.
        // linear_1.backward returns gradient with respect to its input x_normalised.
        Tensor gradient_normalised = this->linear_1.backward(gradient_output1, learning_rate);  // shape: (batch_size, max_sequence_length, d_model)

        // Backprop through Layer Normalisation
        Tensor gradient_input_layer_norm = this->layer_normalisation.backward(gradient_normalised, learning_rate);  // shape: (batch_size, max_sequence_length, d_model)

        Tensor gradient_input =  gradient_x_residual + gradient_input_layer_norm;  // shape: (batch_size, max_sequence_length, d_model)
        return gradient_input;
    }
};


class BaseAttention {
public:
    int n_heads;
    int d_model;
    MultiHeadAttention mha;
    Add add;
    LayerNormalization layer_normalisation;

    BaseAttention(const int n_heads, const int d_model) :
    n_heads(n_heads),
    d_model(d_model),
    mha(n_heads, d_model),
    layer_normalisation(d_model) {}
};


//  This is the self-attention layer of encoder
class GlobalSelfAttention : BaseAttention {
public:
    GlobalSelfAttention(const int n_heads, const int d_model) : BaseAttention(n_heads, d_model) {}

    Tensor forward(const Tensor& x) {
        /*
        @Params
        x: (batch_size, max_sequence_length, d_model)

        @Returns
        output: (batch_size, max_sequence_length, d_model)
        */

        Tensor x_normalised = this->layer_normalisation.forward(x);  // shape: (batch_size, max_sequence_length, d_model)
        Tensor attention_output = this->mha.forward(x_normalised, x_normalised, x_normalised);  // shape: (batch_size, max_sequence_length, d_model)
        Tensor output = this->add.forward({x, attention_output});  // shape: (batch_size, max_sequence_length, d_model)
        return output;
    }
    Tensor backward(const Tensor& gradient_output, const double learning_rate) {
        /*
        @Params
        gradient_output: (batch_size, max_sequence_length, d_model)

        @Returns
        gradient_input: (batch_size, max_sequence_length, d_model)
        */

        // Gradient of input to this attention layer/residual connection and gradient of output of MultiHeadAttention
        auto gradient_list = this->add.backward(gradient_output, learning_rate);
        Tensor gradient_x_residual = gradient_list[0];
        Tensor gradient_attention_output = gradient_list[1];

        // As MultiHeadAttention receives 3 inputs (query, key, value), it returns 3 gradients during backpropagation
        gradient_list = this->mha.backward(gradient_attention_output, learning_rate);
        Tensor gradient_query_normalised = gradient_list[0];
        Tensor gradient_key_normalised = gradient_list[1];
        Tensor gradient_value_normalised = gradient_list[2];

        // Sum the gradients to get the gradient of the output of layer normalisation
        Tensor gradient_output_layer_norm = gradient_query_normalised + gradient_key_normalised + gradient_value_normalised;

        // Gradient of the input of layer normalisation
        Tensor gradient_input_layer_norm = this->layer_normalisation.backward(gradient_output_layer_norm, learning_rate);

        // Sum the gradients to get the gradient of input which is the output of previous layer
        return gradient_x_residual + gradient_input_layer_norm;
    }
};


// This is the self attention layer of decoder
class CausalSelfAttention : BaseAttention {
public:
    CausalSelfAttention(const int n_heads, const int d_model) : BaseAttention(n_heads, d_model) {}

    Tensor forward(const Tensor& x) {
        /*
        @Params
        x: (batch_size, max_sequence_length, d_model)

        @Returns
        output: (batch_size, max_sequence_length, d_model)
        */

        Tensor x_normalised = this->layer_normalisation.forward(x);  // shape: (batch_size, max_sequence_length, d_model)
        Tensor attention_output = this->mha.forward(x_normalised, x_normalised, x_normalised, true);  // shape: (batch_size, max_sequence_length, d_model)
        Tensor output = this->add.forward({x, attention_output});  // shape: (batch_size, max_sequence_length, d_model)
        return output;
    }

    Tensor backward(const Tensor& gradient_output, const double learning_rate) {
        /*
        @Params
        gradient_output: (batch_size, max_sequence_length, d_model)

        @Returns
        gradient_input: (batch_size, max_sequence_length, d_model)
        */

        // Gradient of input to this attention layer/residual connection and gradient of output of MultiHeadAttention
        auto gradient_list = this->add.backward(gradient_output, learning_rate);
        Tensor gradient_x_residual = gradient_list[0];
        Tensor gradient_attention_output = gradient_list[1];

        // As MultiHeadAttention receives 3 inputs (query, key, value), it returns 3 gradients during backpropagation
        gradient_list = this->mha.backward(gradient_attention_output, learning_rate);
        Tensor gradient_query_normalised = gradient_list[0];
        Tensor gradient_key_normalised = gradient_list[1];
        Tensor gradient_value_normalised = gradient_list[2];

        // Sum the gradients to get the gradient of the output of layer normalisation
        Tensor gradient_output_layer_norm = gradient_query_normalised + gradient_key_normalised + gradient_value_normalised;

        // Gradient of the input of layer normalisation
        Tensor gradient_input_layer_norm = this->layer_normalisation.backward(gradient_output_layer_norm, learning_rate);

        // Sum the gradients to get the gradient of input which is the output of previous layer
        return gradient_x_residual + gradient_input_layer_norm;
    }
};


class CrossAttention : BaseAttention {
public:
    CrossAttention(const int n_heads, const int d_model) : BaseAttention(n_heads, d_model) {}

    Tensor forward(const Tensor& x, const Tensor& context) {
        /*
        @Params
        x: (batch_size, max_sequence_length, d_model)    comes from decoder
        context: (batch_size, max_sequence_length, d_model)    comes from encoder

        @Returns
        output: (batch_size, max_sequence_length, d_model)
        */

        Tensor x_normalised = this->layer_normalisation.forward(x);  // shape: (batch_size, max_sequence_length, d_model)
        Tensor context_normalised = this->layer_normalisation.forward(context);  // shape: (batch_size, max_sequence_length, d_model)
        Tensor attention_output = this->mha.forward(x_normalised, context_normalised, context_normalised);  // shape: (batch_size, max_sequence_length, d_model)
        Tensor output = this->add.forward({x, attention_output});  // shape: (batch_size, max_sequence_length, d_model)
        return output;
    }

    vector<Tensor> backward(const Tensor& gradient_output, const double learning_rate) {
        /*
        @Params
        gradient_output: (batch_size, max_sequence_length, d_model)

        @Returns
        gradient_x: (batch_size, max_sequence_length, d_model)
        gradient_context: (batch_size, max_sequence_length, d_model)
        */

        // Gradient of input to this attention layer/residual connection and gradient of output of MultiHeadAttention
        auto gradient_list = this->add.backward(gradient_output, learning_rate);
        Tensor gradient_x_residual = gradient_list[0];
        Tensor gradient_attention_output = gradient_list[1];

        // As MultiHeadAttention receives 3 inputs (query, key, value), it returns 3 gradients during backpropagation
        gradient_list = this->mha.backward(gradient_attention_output, learning_rate);
        Tensor gradient_query_normalised = gradient_list[0];
        Tensor gradient_key_normalised = gradient_list[1];
        Tensor gradient_value_normalised = gradient_list[2];

        // Since context was passed as key and value, sum their gradients to get the gradient of context
        Tensor gradient_context_normalised = gradient_key_normalised + gradient_value_normalised;

        // Gradient of context
        Tensor gradient_context = this->layer_normalisation.backward(gradient_context_normalised, learning_rate);

        // Gradient of x that goes into layer normalisation
        Tensor gradient_x_input_layer_norm = this->layer_normalisation.backward(gradient_query_normalised, learning_rate);

        // Summing the gradients of residual connection and input of layer normalisation yields gradient of the
        // input of this layer, which is the output of previous layer. Gradient of residual connection should only
        // be added to the gradient of input of layer normalisation because the residual connection is between the
        // output of the causal attention layer and the output of MultiHeadAttention of cross attention layer.
        // There is no residual connection that involves the output of encoder

        return {gradient_x_residual + gradient_x_input_layer_norm, gradient_context};
    }
};


class EncoderLayer {
public:
    int d_model;
    int d_ff;
    int n_heads;
    FeedForward feed_forward;
    GlobalSelfAttention self_attention;

    EncoderLayer(const int d_model, const int d_ff, const int n_heads) :
    d_model(d_model),
    d_ff(d_ff),
    n_heads(n_heads),
    feed_forward(d_model, d_ff),
    self_attention(n_heads, d_model) {}

    Tensor forward(const Tensor& x) {
        /*
        @Params
        x: (batch_size, max_sequence_length, d_model)

        @Returns
        output: (batch_size, max_sequence_length, d_model)
        */

        Tensor output = this->self_attention.forward(x);  // shape: (batch_size, max_sequence_length, d_model)
        output = this->feed_forward.forward(output);  // shape: (batch_size, max_sequence_length, d_model)
        return output;
    }

    Tensor backward(const Tensor& gradient_output, const double learning_rate) {
        /*
        @Params
        gradient_output: (batch_size, max_sequence_length, d_model)

        @Returns
        gradient: (batch_size, max_sequence_length, d_model)
        */

        // Gradient of the output of self attention layer
        Tensor gradient = this->feed_forward.backward(gradient_output, learning_rate);

        // Gradient of the input of self attention layer which was the output of
        // previous layer during forward pass
        gradient = this->self_attention.backward(gradient, learning_rate);
        return gradient;
    }
};


class Encoder {
public:
    int d_model;
    int n_layers;
    int n_heads;
    int d_ff;
    int vocab_size;
    PositionalEmbedding positional_embedding;
    vector<EncoderLayer> encoder_layers;

    Encoder(const int d_model, const int n_layers, const int n_heads, const int d_ff, const int vocab_size) :
        d_model(d_model),
        n_layers(n_layers),
        n_heads(n_heads),
        d_ff(d_ff),
        vocab_size(vocab_size),
        positional_embedding(PositionalEmbedding(vocab_size, d_model)) {
            for(int i = 0; i < n_layers; i++) {
                encoder_layers.emplace_back(d_model, d_ff, n_heads);
            }
        }

    Tensor forward(const Tensor& x) {
        /*
        @Params
        x: (batch_size, max_sequence_length)

        @Returns
        output: (batch_size, max_sequence_length, d_model)
        */
        Tensor x_embedded = this->positional_embedding.forward(x);  // shape: (batch_size, max_sequence_length, d_model)
        Tensor output = x_embedded;

        for(auto& layer : encoder_layers) {
            output = layer.forward(output);
        }
        return output;
    }

    Tensor backward(const Tensor& gradient_output, const double learning_rate) {
        /*
        @Params
        gradient_output: (batch_size, max_sequence_length, d_model)

        @Returns
        gradient: (batch_size, max_sequence_length, d_model)
        */

        Tensor gradient = gradient_output;
        for(int i = this->encoder_layers.size() - 1; i >= 0; i--) {
            gradient = this->encoder_layers[i].backward(gradient, learning_rate);
        }

        gradient = this->positional_embedding.backward(gradient, learning_rate);

        return gradient;
    }
};


class DecoderLayer {
public:
    int d_model;
    int d_ff;
    int n_heads;
    FeedForward feed_forward;
    CausalSelfAttention causal_self_attention;
    CrossAttention cross_attention;

    DecoderLayer(const int d_model, const int d_ff, const int n_heads) :
    d_model(d_model),
    d_ff(d_ff),
    n_heads(n_heads),
    feed_forward(FeedForward(d_model, d_ff)),
    causal_self_attention(CausalSelfAttention(n_heads, d_model)),
    cross_attention(CrossAttention(n_heads, d_model)) {}

    Tensor forward(const Tensor& x, const Tensor& context) {
        /*
        @Params
        x: (batch_size, max_sequence_length, d_model)    comes from positional embedding of decoder for the first decoder layer, otherwise previous decoder layer
        context: (batch_size, max_sequence_length, d_model)    comes from encoder

        @Returns
        x: (batch_size, max_sequence_length, d_model)
        */

        Tensor output = this->causal_self_attention.forward(x);  // shape: (batch_size, max_sequence_length, d_model)
        output = this->cross_attention.forward(output, context);  // shape: (batch_size, max_sequence_length, d_model)
        output = this->feed_forward.forward(output);  // shape: (batch_size, max_sequence_length, d_model)
        return output;
    }

    vector<Tensor> backward(const Tensor& gradient_output, const double learning_rate) {
        /*
        @Params
        gradient_output: (batch_size, max_sequence_length, d_model)

        @Returns
        gradient_x_causal_self_attention: (batch_size, max_sequence_length, d_model)
        gradient_context: (batch_size, max_sequence_length, d_model)
        */

        // gradient_output is the gradient of the output of feed-forward layer of every decoder layer
        // gradient_cross_attention_output stores the gradient of the output of cross attention layer
        // which was the input of feed-forward during forward pass

        Tensor gradient_cross_attention_output = this->feed_forward.backward(gradient_output, learning_rate);

        // During forward pass, there were two inputs, x and context, which came from positional embedding
        // of decoder, and encoder respectively. Therefore, during backpropagation we calculate their gradients
        // and pass gradient of context to decoder layer which passes it to encoder because it came from
        // encoder, and gradient of x to positional embedding of decoder because it came from that layer.

        auto gradient_list = this->cross_attention.backward(gradient_cross_attention_output, learning_rate);
        Tensor gradient_causal_self_attention_output = gradient_list[0];
        Tensor gradient_context = gradient_list[1];

        // Gradient of input that goes into causal self-attention
        Tensor gradient_causal_self_attention_input = this->causal_self_attention.backward(gradient_causal_self_attention_output, learning_rate);

        return {gradient_causal_self_attention_input, gradient_context};
    }
};


class Decoder {
public:
    int d_model;
    int n_layers;
    int n_heads;
    int d_ff;
    int vocab_size;
    PositionalEmbedding positional_embedding;
    vector<DecoderLayer> decoder_layers;

    Decoder(const int d_model, const int n_layers, const int n_heads, const int d_ff, const int vocab_size) :
        d_model(d_model),
        n_layers(n_layers),
        n_heads(n_heads),
        d_ff(d_ff),
        vocab_size(vocab_size),
        positional_embedding(PositionalEmbedding(vocab_size, d_model)) {
            for(int i = 0; i < n_layers; i++) {
                decoder_layers.emplace_back(d_model, d_ff, n_heads);
            }
        }

    Tensor forward(const Tensor& x, const Tensor& context) {
        /*
        @Params
        x: (batch_size, max_sequence_length)
        context: (batch_size, max_sequence_length, d_model)    comes from encoder

        @Returns
        x: (batch_size, max_sequence_length, d_model)
        */

        Tensor x_embedded = this->positional_embedding.forward(x);  // shape: (batch_size, max_sequence_length, d_model)
        Tensor output = x_embedded;

        for(auto& layer : decoder_layers) {
            output = layer.forward(output, context);
        }
        return output;
    }

    Tensor backward(const Tensor& gradient_output, const double learning_rate) {
        /*
        @Params
        gradient_output: (batch_size, max_sequence_length, d_model)

        @Returns
        gradient_context: (batch_size, max_sequence_length, d_model)
        */

        const int batch_size = get<0>(gradient_output.shape());
        const int max_sequence_length = get<1>(gradient_output.shape());
        const int d_model = get<2>(gradient_output.shape());

        // During forward pass, for every decoder layer, x comes from previous decoder layer while
        // context is injected from outside, in this case, encoder. Therefore, gradient_x should be
        // passed to the previous layer during backpropagation.

        Tensor gradient_x = gradient_output;

        // We need to pass this to encoder. It's the summation of the gradients of context of all the cross attention layers
        Tensor gradient_context;
        for(int i = 0; i < batch_size; i++) {
            MatrixXd mat = MatrixXd::Zero(max_sequence_length, d_model);
            gradient_context.data.emplace_back(mat);
        }

        Tensor gradient_context_;

        for(int i = this->decoder_layers.size() - 1; i >= 0; i--) {
            auto gradient_list = this->decoder_layers[i].backward(gradient_x, learning_rate);
            gradient_x = gradient_list[0];
            gradient_context_ = gradient_list[1];

            gradient_context = gradient_context + gradient_context_;
        }

        this->positional_embedding.backward(gradient_x, learning_rate);

        return gradient_context;
    }
};


class Transformer {
public:
    int d_model;
    int n_layers;
    Encoder encoder;
    Decoder decoder;
    Linear final_layer;

    Transformer(const int d_model, const int n_layers, const int n_heads, const int d_ff, const int input_vocab_size, const int target_vocab_size) :
    d_model(d_model),
    n_layers(n_layers),
    encoder(d_model, n_layers, n_heads, d_ff, input_vocab_size),
    decoder(d_model, n_layers, n_heads, d_ff, target_vocab_size),
    final_layer(Linear(d_model, target_vocab_size)) {}

    Tensor forward(const vector<MatrixXi>& inputs) {
        /*
        @Params
        inputs: A list of two matrices of shape (batch_size, max_sequence_length)

        @Returns
        logits: (batch_size, max_sequence_length, target_vocab_size)
        */

        Tensor context, x;
        context.data.emplace_back(inputs[0]);
        x.data.emplace_back(inputs[1]);

        context = this->encoder.forward(context);
        x = this->decoder.forward(x, context);
        Tensor logits = this->final_layer.forward(x);
        return logits;
    }

    double backward(const MatrixXi& input_data, const MatrixXi& target_data, const double learning_rate) {
        /*
        @Params
        input_data: (batch_size, max_sequence_length)
        target_data: (batch_size, max_sequence_length)
        learning_rate: scalar

        @Returns
        loss: scalar
        */

        Tensor logits = this->forward({input_data, target_data});

        Tensor probabilities = softmax(logits);

        double loss = cross_entropy_loss(probabilities, target_data);

        Tensor gradient_for_final_layer = cross_entropy_gradient(probabilities, target_data);  // shape: (batch_size, max_sequence_length, target_vocab_size)

        Tensor gradient_for_decoder = this->final_layer.backward(gradient_for_final_layer, learning_rate);  // shape: (batch_size, max_sequence_length, d_model)

        Tensor gradient_for_encoder = this->decoder.backward(gradient_for_decoder, learning_rate);  // shape: (batch_size, max_sequence_length, d_model)

        this->encoder.backward(gradient_for_encoder, learning_rate);

        return loss;
    }

    Tensor fit(MatrixXi input_data, MatrixXi target_data, int epochs, int batch_size, double learning_rate = 0.001, bool print_predictions = true) {
        int num_samples = input_data.rows();

        for(int epoch = 1; epoch <= epochs; epoch++) {
            double total_loss = 0;
            for(int i = 0; i < num_samples; i += batch_size) {

                int current_batch_size = min(batch_size, num_samples - i);

                MatrixXi input_batch = input_data.block(i, 0, current_batch_size, input_data.cols());
                MatrixXi target_batch = target_data.block(i, 0, current_batch_size, target_data.cols());

                double loss = this->backward(input_batch, target_batch, learning_rate);
                total_loss += loss;
            }
            if(epoch == 1 || epoch % 10 == 0 || epoch == epochs) {
                printf("Epoch %02d  Loss: %.4f\n", epoch, total_loss);
            }
        }
        cout << "\n";

        Tensor predictions;

        for(int i = 0; i < num_samples; i++) {
            Tensor logits = this->forward({input_data.row(i), target_data.row(i)});
            Tensor prediction = softmax(logits);

            if(print_predictions) {
                auto prediction_mat = get<MatrixXd>(prediction.data[0]);

                for(int i = 0; i < prediction_mat.rows(); i++) {
                    cout << prediction_mat.row(i) << "  ---------  " << prediction_mat.row(i).maxCoeff() << "\n";
                }
                cout << "\n";
            }
            predictions.data.emplace_back(prediction.data[0]);
        }
        return predictions;
    }
};

int main() {
    int input_vocab_size = 10;
    int target_vocab_size = 15;
    int num_sequences = 5;
    int max_sequence_length = 10;
    int epochs = 30;
    int batch_size = 32;

    // Model specifications
    int n_layers = 6;
    int n_heads = 8;
    int d_model = 32;
    int d_ff = d_model * 4;

    MatrixXi input_data(num_sequences, max_sequence_length);
    MatrixXi target_data(num_sequences, max_sequence_length);

    for(int i = 0; i < num_sequences; i++) {
        for(int j = 0; j < max_sequence_length; j++) {
            input_data(i, j) = uniform_int_distribution(0, input_vocab_size - 1)(rng);
            target_data(i, j) = uniform_int_distribution(0, target_vocab_size - 1)(rng);
        }
    }

    Transformer transformer(d_model, n_layers, n_heads, d_ff, input_vocab_size, target_vocab_size);

    auto start = high_resolution_clock::now();

    Tensor predictions = transformer.fit(input_data, target_data, epochs, batch_size);

    auto end = high_resolution_clock::now();
    auto totalTime = duration_cast<milliseconds>(end - start);

    int correct_generated_tokens = 0;

    for(int i = 0; i < num_sequences; i++) {
        vector<int> generated_tokens(max_sequence_length);
        const MatrixXd prediction = get<MatrixXd>(predictions.data[i]);

        for(int j = 0; j < max_sequence_length; j++) {

            prediction.row(j).maxCoeff(&generated_tokens[j]);

            if(generated_tokens[j] == target_data(i, j)) {
                correct_generated_tokens++;
            }
        }
        for(int j = 0; j < max_sequence_length; j++) {
            printf("%3d ", target_data(i, j));
        }
        cout  << "    ";
        for(auto token : generated_tokens) {
            printf("%3d ", token);
        }
        cout << "\n";
    }

    printf("Correct generated tokens: %d  Accuracy: %2f%%\n", correct_generated_tokens, 1.0 * correct_generated_tokens / (num_sequences * max_sequence_length) * 100);

    cout << "Total time = " << static_cast<double>(totalTime.count()) / 1000 << " seconds\n";
}