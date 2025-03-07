#include <bits/stdc++.h>
#include <Eigen/Dense>
using namespace std;
using namespace chrono;
using namespace Eigen;

mt19937 rng(steady_clock::now().time_since_epoch().count());
normal_distribution normal_dist(0.0, .01);

// Cross-Entropy Loss function
double cross_entropy_loss(const vector<vector<double>>& predictions, const vector<int>& labels) {
    const int batch_size = static_cast<int>(predictions.size());

    double loss = 0;
    // Calculate loss for each sample in the batch
    for (int i = 0; i < batch_size; i++) {
        const int label = labels[i];
        // Extract the probability corresponding to the correct label
        const double correct_prob = predictions[i][label];
        loss -= log(correct_prob + 1e-15f); // Add epsilon for numerical stability
    }

    return loss / batch_size;  // Average loss
}

// Cross-Entropy Gradient function
vector<vector<double>> cross_entropy_gradient(const vector<vector<double>>& predictions, const vector<int>& labels) {
    const int batch_size = static_cast<int>(predictions.size());

    const int num_classes = static_cast<int>(predictions[0].size());

    vector gradient(batch_size, vector<double>(num_classes));

    // Calculate gradient for each sample in the batch
    for(int i = 0; i < batch_size; i++) {
        const int label = labels[i];

        // Copy the predictions and subtract 1 from the correct class
        for(int j = 0; j < num_classes; j++) {
            gradient[i][j] = predictions[i][j];
        }
        gradient[i][label] -= 1;
    }

    // Normalize the gradient by batch size
    for (int i = 0; i < batch_size; i++) {
        for (int j = 0; j < num_classes; j++) {
            gradient[i][j] /= batch_size;
        }
    }

    return gradient;
}

MatrixXd randomUniformMatrix(int rows, int cols, double min, double max) {
    uniform_real_distribution<double> dis(min, max);

    MatrixXd mat(rows, cols);

    for(int i = 0; i < rows; i++) {
        for(int j = 0; j < cols; j++) {
            mat(i, j) = dis(rng);
        }
    }
    return mat;
}

// activation function (ReLU)
MatrixXd relu(const MatrixXd& z) {
    return z.unaryExpr([](double val) {
            return max(0.0, val);
        }
    );
}


class Embedding {
public:
    int vocab_size;
    int embedding_dim;
    MatrixXd embeddings;
    MatrixXi input;  // shape: (batch_size, sequence_length)

    Embedding(int vocab_size, int embedding_dim) : vocab_size(vocab_size), embedding_dim(embedding_dim) {
        embeddings = MatrixXd(vocab_size, embedding_dim);  // Embeddings matrix

        // Initialise embeddings with random values
        for(int i = 0; i < vocab_size; i++) {
            for(int j = 0; j < embedding_dim; j++) {
                embeddings(i, j) = normal_dist(rng);
            }
        }
    }

    vector<MatrixXd> forward(const MatrixXi& x) {
        this->input = x;
        int batch_size = x.rows();
        int sequence_length = x.cols();

        vector output(batch_size, MatrixXd(sequence_length, embedding_dim));

        for(int i = 0; i < batch_size; i++) {
            for(int j = 0; j < sequence_length; j++) {
                int token = x(i, j);  // Get the index for the word in the vocabulary

                //  Get the corresponding embedding
                output[i].row(j) = embeddings.row(token);
            }
        }

        return output;  // shape: (batch_size, sequence_length, embedding_dim)
    }

    void backward(const vector<MatrixXd>& grad, double learning_rate) {

        // Update embedding vectors using gradient.
        // grad has shape (batch_size, sequence_length, embedding_dim).

        int batch_size = grad.size();
        int sequence_length = grad[0].rows();

        cout << batch_size << "   " << sequence_length << "\n";

        learning_rate *= batch_size;

        set<int> unique_tokens;

        // Collect unique tokens from input sequences
        for(int i = 0; i < batch_size; i++) {
            for(int j = 0; j < sequence_length; j++) {
                unique_tokens.insert(input(i, j));
            }
        }

        // Iterate over each unique token
        for(int token: unique_tokens) {
            VectorXd sum_grad = VectorXd::Zero(embedding_dim);  // Initialise gradient vector for the token

            // Accumulate gradients for this token
            for(int i = 0; i < batch_size; i++) {
                for(int j = 0; j < sequence_length; j++) {
                    if(input(i, j) == token) {
                        // Accumulate gradients for the corresponding token's embedding
                        sum_grad += grad[i].row(j).transpose();
                    }
                }
            }

            // Update the embedding for the current token
            for(int i = 0; i < embedding_dim; i++) {
                embeddings(token, i) -= learning_rate * sum_grad(i);
            }
        }
    }
};


class SimpleRNN {
public:
    int hidden_size;
    int input_size;
    bool return_sequences;

    MatrixXd W_xh;  // Weight matrix from input to hidden (shape: hidden_size x input_size)
    MatrixXd W_hh;  // Recurrent weight matrix (shape: hidden_size x hidden_size)
    VectorXd b_h;   // Bias for hidden state (shape: hidden_size)

    // Stored for backward pass
    vector<MatrixXd> inputs;   // Each element: (sequence_length x input_size)

    // This will hold the computed hidden states for each sequence in the batch.
    vector<MatrixXd> h_states; // Each element: (sequence_length x hidden_size)

    // Constructor
    SimpleRNN(int input_size, int hidden_size, bool return_sequences) : input_size(input_size), hidden_size(hidden_size), return_sequences(return_sequences) {
        double limit = sqrt(6.0 / (input_size + hidden_size));

        // Glorot uniform initialisation
        W_xh = randomUniformMatrix(hidden_size, input_size, -limit, limit);

        // Orthogonal initialisation
        W_hh = orthogonalInit(hidden_size);

        // Initialise bias to zero
        b_h = VectorXd::Zero(hidden_size);
    }

    // Forward pass: x is a vector of matrices, one per batch instance.
    // Each matrix has shape (sequence_length, input_size).
    // Returns a vector of matrices where each matrix is either:
    // - (sequence_length, hidden_size) if return_sequences==true, or
    // - (1, hidden_size) if false (only last hidden state).
    vector<MatrixXd> forward(const vector<MatrixXd>& x) {

        /*
        x shape: (batch_size, sequence_length, input_size)
        */

        int batch_size = x.size();
        int sequence_length = x[0].rows();

        // Store input for backward pass
        inputs = x;

        // Prepare the output container.
        vector<MatrixXd> output(batch_size);

        for(int i = 0; i < batch_size; i++) {
            // Create a matrix to store hidden states for this sequence.
            // Each row will correspond to the hidden state at one timestep.
            MatrixXd h_seq(sequence_length, hidden_size);

            // Initialize the hidden state h as a zero vector (shape: hidden_size)
            VectorXd h = VectorXd::Zero(hidden_size);

            // Iterate over time steps
            for(int t = 0; t < sequence_length; t++) {
                // Ge the t-th input token as a row vector (shape: 1 x input_size)
                // We then transpose it to have a column vector (input_size x 1)
                VectorXd x_t = x[i].row(t);

                // Compute the new hidden state:
                // h_new = tanh(W_xh * x_t + W_hh * h + b_h)
                VectorXd h_new = (W_xh * x_t + W_hh * h + b_h).array().tanh();
                h = h_new;

                // Store the hidden state as a row (transpose h to get 1 x hidden_size)
                h_seq.row(t) = h.transpose();
            }

            // Save the full hidden state sequence for this instance
            h_states.emplace_back(h_seq);

            // If we return full sequences, output is the entire h_seq.
            // Otherwise, output only the last hidden state as a 1 x hidden_size matrix.
            if (return_sequences) {
                output[i] = h_seq;
            }
            else {
                output[i] = h_seq.row(sequence_length - 1);
            }
        }
        return output;
    }

    vector<MatrixXd> backward(const vector<MatrixXd>& dL_dh_last, double learning_rate) {
        // dL_dh_last shape: (batch_size, sequence_length, hidden_size)

        int batch_size = inputs.size();
        int sequence_length = inputs[0].rows();
        learning_rate *= batch_size;

        MatrixXd dW_xh = MatrixXd::Zero(hidden_size, input_size);
        MatrixXd dW_hh = MatrixXd::Zero(hidden_size, hidden_size);
        MatrixXd db = MatrixXd::Zero(hidden_size, 1);

        vector dL_dx = vector<MatrixXd>(batch_size, MatrixXd::Zero(sequence_length, input_size));

        MatrixXd dL_dh_next = MatrixXd::Zero(batch_size, hidden_size);

        MatrixXd dL_dh_t;

        for(int t = sequence_length - 1; t >= 0; t--) {
            MatrixXd dL_dh_last_at_t(batch_size, hidden_size);
            MatrixXd h_states_at_t(batch_size, hidden_size);
            MatrixXd h_states_at_t_minus_1(batch_size, hidden_size);
            MatrixXd inputs_at_t(batch_size, input_size);

            for(int i = 0; i < batch_size; i++) {
                dL_dh_last_at_t.row(i) = dL_dh_last[i].row(t);
                h_states_at_t.row(i) = h_states[i].row(t);
                inputs_at_t.row(i) = inputs[i].row(t);

                if(t > 0) {
                    h_states_at_t_minus_1.row(i) = h_states[i].row(t - 1);
                }
            }
            dL_dh_t = dL_dh_last_at_t + dL_dh_next;

            // cout << "dl_dh_t\n" << dL_dh_t << "\n\n";
            // cout << "h_states_at_t\n" << h_states_at_t << "\n\n";

            MatrixXd dtanh = (1 - h_states_at_t.array().square()) * dL_dh_t.array();
            // cout << "dtanh\n" << dtanh << "\n\n";

            dW_xh += dtanh.transpose() * inputs_at_t;
            // cout << "dw_xh\n" << dW_xh << "\n\n";
            dW_hh += dtanh.transpose() * (t > 0 ? h_states_at_t_minus_1 : MatrixXd::Zero(batch_size, hidden_size));
            // cout << "dw_hh\n" << dW_hh << "\n\n";
            db += dtanh.colwise().sum().transpose();
            // cout << "db\n" << db << "\n\n";

            // dtanh shape:   (batch_size, hidden_size)
            // W_xh shape:    (hidden_size, input_size)
            // dL_dx_t shape: (batch_size, input_size)

            MatrixXd dL_dx_t = dtanh * W_xh;

            for(int i = 0; i < batch_size; i++) {
                dL_dx[i].row(t) = dL_dx_t.row(i);
            }
            // cout << "dl_dx\n";
            // for(auto& mat: dL_dx) {
            //     cout << mat << "\n\n";
            // }

            dL_dh_next = dtanh * W_hh;
            // cout << "dl_dh_next\n" << dL_dh_next << "\n\n";
        }

        W_xh -= learning_rate * dW_xh;
        W_hh -= learning_rate * dW_hh;
        b_h -= learning_rate * db;

        return dL_dx;
    }


    // This function will be called if the next layer is a Dense layer. In that case, during forward pass
    // the output for only the last time step was passed to the Dense layer. Therefore, during backpropagation
    // dL_dh_last is a 2D matrix, since there is no time step

    vector<MatrixXd> backward(const MatrixXd& dL_dh_last, double learning_rate) {
        // dL_dh_last shape: (batch_size, hidden_size)

        int batch_size = inputs.size();
        int sequence_length = inputs[0].rows();
        learning_rate *= batch_size;

        MatrixXd dW_xh = MatrixXd::Zero(hidden_size, input_size);
        MatrixXd dW_hh = MatrixXd::Zero(hidden_size, hidden_size);
        MatrixXd db = MatrixXd::Zero(hidden_size, 1);

        vector dL_dx = vector<MatrixXd>(batch_size, MatrixXd::Zero(sequence_length, input_size));

        MatrixXd dL_dh_next = MatrixXd::Zero(batch_size, hidden_size);

        MatrixXd dL_dh_t = dL_dh_last;

        for(int t = sequence_length - 1; t >= 0; t--) {
            MatrixXd h_states_at_t(batch_size, hidden_size);
            MatrixXd h_states_at_t_minus_1(batch_size, hidden_size);
            MatrixXd inputs_at_t(batch_size, input_size);

            for(int i = 0; i < batch_size; i++) {
                h_states_at_t.row(i) = h_states[i].row(t);
                inputs_at_t.row(i) = inputs[i].row(t);

                if(t > 0) {
                    h_states_at_t_minus_1.row(i) = h_states[i].row(t - 1);
                }
            }

            MatrixXd dtanh = (1 - h_states_at_t.array().square()) * dL_dh_t.array();

            dW_xh += dtanh.transpose() * inputs_at_t;
            dW_hh += dtanh.transpose() * (t > 0 ? h_states_at_t_minus_1 : MatrixXd::Zero(batch_size, hidden_size));
            db += dtanh.colwise().sum().transpose();

            MatrixXd dL_dx_t = dtanh * W_xh;
            for(int i = 0; i < batch_size; i++) {
                dL_dx[i].row(t) = dL_dx_t.row(i);
            }
            dL_dh_t = dtanh * W_hh;
        }

        W_xh -= learning_rate * dW_xh;
        W_hh -= learning_rate * dW_hh;
        b_h -= learning_rate * db;

        return dL_dx;
    }
private:
    MatrixXd orthogonalInit(int size) {
        MatrixXd A = randomUniformMatrix(size, size, -1.0, 1.0);
        JacobiSVD svd(A, ComputeFullU);
        return svd.matrixU();
    }
};

int main() {

    int vocab_size = 10;
    int embedding_dim = 4;
    int max_sequence_length = 5;
    int num_sequences = 3;
    int classes = 5;
    int epochs = 10;
    int batch_size = 32;

    MatrixXi data(num_sequences, max_sequence_length);
    vector<int> labels = vector<int>(num_sequences);

    for(int i = 0; i < num_sequences; i++) {
        labels[i] = uniform_int_distribution(0, classes - 1)(rng);
        for(int j = 0; j < max_sequence_length; j++) {
            data(i, j) = uniform_int_distribution(0, vocab_size - 1)(rng);
        }
    }

    cout << data << "\n\n";

    Embedding embedding = Embedding(vocab_size, embedding_dim);

    int hidden_size = 4;

    SimpleRNN rnn1 = SimpleRNN(embedding_dim, hidden_size, true);
    SimpleRNN rnn2 = SimpleRNN(hidden_size, hidden_size, false);

    auto output_embed = embedding.forward(data);
    auto output_rnn1 = rnn1.forward(output_embed);
    auto output_rnn2 = rnn2.forward(output_rnn1);

    cout << "--------------------------------\noutput_rnn2\n\n";
    for(auto& mat: output_rnn2) {
        cout << mat << "\n\n";
    }

    MatrixXd grad(num_sequences, hidden_size);

    for(int j = 0; j < num_sequences; j++) {
        for(int k = 0; k < hidden_size; k++) {
            grad(j, k) = uniform_real_distribution(0.0, 1.0)(rng);
        }
    }

    auto grad_rnn2 = rnn2.backward(grad, .01);

    cout << "--------------------------------\ngrad_rnn2\n";
    for(auto& mat: grad_rnn2) {
        cout << mat << "\n\n";
    }

    auto grad_rnn1 = rnn1.backward(grad_rnn2, .01);

    cout << "--------------------------------\ngrad_rnn1\n";
    for(auto& mat: grad_rnn1) {
        cout << mat << "\n\n";
    }
}
