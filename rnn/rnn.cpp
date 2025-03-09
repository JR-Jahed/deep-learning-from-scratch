#include <bits/stdc++.h>
#include <Eigen/Dense>
using namespace std;
using namespace chrono;
using namespace Eigen;


mt19937 rng(steady_clock::now().time_since_epoch().count());
normal_distribution normal_dist(0.0, .01);

using MatrixVariant = variant<MatrixXd, MatrixXi>;

struct Tensor {
    vector<MatrixVariant> data;

    void print() {
        for(auto& var: data) {
            cout << get<MatrixXd>(var) << "\n\n";
        }
    }
};


// Cross-Entropy Loss function
double cross_entropy_loss(const Tensor& predictions, const vector<int>& labels) {

    // Expect the predictions Tensor to hold a MatrixXd as its first element.
    MatrixXd pred;
    try {
        pred = get<MatrixXd>(predictions.data[0]);
    }
    catch(const bad_variant_access&) {
        throw runtime_error("cross_entropy_loss expects predictions to contain a MatrixXd");
    }

    const int batch_size =  pred.rows();
    double loss = 0;

    // Compute the loss for each sample.
    for(int i = 0; i < batch_size; i++) {
        int label = labels[i];
        double correct_prob = pred(i, label);
        loss -= log(correct_prob + 1e-15); // Add epsilon for numerical stability.
    }

    return loss / batch_size;  // Average loss
}

// Cross-Entropy Gradient function
Tensor cross_entropy_gradient(const Tensor& predictions, const vector<int>& labels) {
    MatrixXd pred;
    try {
        pred = get<MatrixXd>(predictions.data[0]);
    }
    catch(const bad_variant_access&) {
        throw runtime_error("cross_entropy_gradient expects predictions to contain a MatrixXd");
    }

    int batch_size = pred.rows();
    int num_classes = pred.cols();
    MatrixXd grad = pred; // Start with a copy of the predictions.

    // For each sample, subtract 1 from the probability of the correct class.

    for(int i = 0; i < batch_size; i++) {
        int label = labels[i];
        grad(i, label) -= 1.0;
    }

    // Normalize the gradient by the batch size.
    grad /= static_cast<double>(batch_size);

    // Pack the resulting gradient matrix into a Tensor.
    Tensor grad_tensor;
    grad_tensor.data.emplace_back(grad);
    return grad_tensor;
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
    return z.unaryExpr([](double val) { return max(0.0, val); });
}


class Layer {
public:
    virtual Tensor forward(const Tensor& input) = 0;
    virtual Tensor backward(const Tensor& grad, double learning_rate) = 0;
    virtual ~Layer() {}
};

class Embedding: public Layer {
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

    Tensor forward(const Tensor& x) override {

        Tensor output;

        if(const MatrixXi* pIndices = get_if<MatrixXi>(&x.data[0])) {
            const MatrixXi& indices = *pIndices;

            this->input = indices;
            int batch_size = indices.rows();
            int sequence_length = indices.cols();

            vector<MatrixXd> embedded(batch_size);

            for(int i = 0; i < batch_size; i++) {
                MatrixXd emb_i(sequence_length, embedding_dim);
                for(int j = 0; j < sequence_length; j++) {
                    int token = indices(i, j);  // Get the index for the word in the vocabulary

                    //  Get the corresponding embedding
                    emb_i.row(j) = embeddings.row(token);
                }
                embedded[i] = emb_i;
            }

            for(const auto& m: embedded) {
                output.data.emplace_back(m);
            }
        }
        else {
            throw runtime_error("Tensor does not contain MatrixXi as expected.");
        }

        return output;  // shape: (batch_size, sequence_length, embedding_dim)
    }

    Tensor backward(const Tensor& grad, double learning_rate) {

        // Update embedding vectors using gradient.
        // grad has shape (batch_size, sequence_length, embedding_dim).

        // Convert grad Tensor to a vector of MatrixXd
        vector<MatrixXd> grad_mats;

        for(const auto& var: grad.data) {
            // Use get to extract the MatrixXd
            try {
                const MatrixXd& mat = get<MatrixXd>(var);
                grad_mats.emplace_back(mat);
            }
            catch(const bad_variant_access&) {
                throw runtime_error("Expected MatrixXd in grad Tensor in Embedding layer backward.");
            }
        }
        int batch_size = grad_mats.size();
        // Each matrix in grad_mats has shape: (sequence_length, embedding_dim)
        int sequence_length = grad_mats[0].rows();

        learning_rate *= batch_size;

        cout << "embedding grad\n" << grad_mats[0] << "\n\n";

        // Collect unique tokens from the saved input indices
        set<int> unique_tokens;
        for (int i = 0; i < batch_size; i++) {
            for (int j = 0; j < sequence_length; j++) {
                unique_tokens.insert(input(i, j));
            }
        }

        // Iterate over each unique token to accumulate gradients and update embeddings
        for (int token : unique_tokens) {
            VectorXd sum_grad = VectorXd::Zero(embedding_dim);
            for (int i = 0; i < batch_size; i++) {
                for (int j = 0; j < sequence_length; j++) {
                    if (input(i, j) == token) {
                        // grad_mats[i].row(j) is a row vector; transpose it to add as a column vector.
                        sum_grad += grad_mats[i].row(j).transpose();
                    }
                }
            }
            // Update the embedding for the current token
            // (Assuming embeddings rows correspond to tokens.)
            for (int i = 0; i < embedding_dim; i++) {
                embeddings(token, i) -= learning_rate * sum_grad(i);
            }
        }
        Tensor tensor;
        return tensor;
    }
};


class SimpleRNN : public Layer {
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
    Tensor forward(const Tensor& input) override {

        /*
        x shape: (batch_size, sequence_length, input_size)
        */

        vector<MatrixXd> x;
        x.reserve(input.data.size());
        for (const auto &var : input.data) {
            try {
                const MatrixXd &mat = get<MatrixXd>(var);
                x.emplace_back(mat);
            }
            catch (const bad_variant_access&) {
                throw runtime_error("SimpleRNN expects input Tensor to contain MatrixXd.");
            }
        }

        int batch_size = x.size();
        int sequence_length = x[0].rows();

        // Store input for backward pass
        inputs = x;

        // Prepare the output container.
        vector<MatrixXd> output(batch_size);

        // Iterate over each sequence (batch element).
        for(int i = 0; i < batch_size; i++) {
            // Create a matrix to store hidden states for this sequence.
            // Each row will correspond to the hidden state at one timestep.
            MatrixXd h_seq(sequence_length, hidden_size);

            // Initialize the hidden state h as a zero vector (shape: hidden_size)
            VectorXd h = VectorXd::Zero(hidden_size);

            // Process the sequence time step by time step.
            for(int t = 0; t < sequence_length; t++) {
                // Ge the t-th input token as a row vector (shape: 1 x input_size)
                // We then transpose it to have a column vector (input_size x 1)
                VectorXd x_t = x[i].row(t);

                // Compute the new hidden state: h_new = tanh(W_xh * x_t + W_hh * h + b_h)
                VectorXd h_new = (W_xh * x_t + W_hh * h + b_h).array().tanh();
                h = h_new;

                // Store the hidden state as a row (transpose h to get 1 x hidden_size)
                h_seq.row(t) = h.transpose();
            }

            // Save the full hidden state sequence for backward pass
            h_states.emplace_back(h_seq);

            // If we return full sequences, output is the entire h_seq.
            // Otherwise, output only the last hidden state as a 1 x hidden_size matrix.
            if (return_sequences) {
                output[i] = h_seq;  // shape: (sequence_length, hidden_size)
            }
            else {
                output[i] = h_seq.row(sequence_length - 1);  // shape: (1, hidden_size)
            }
        }

        // Pack the output vector into a Tensor.
        Tensor out;
        for (const auto &mat : output) {
            out.data.emplace_back(mat);
        }
        return out;
    }

    Tensor backward(const Tensor& grad, double learning_rate) override {
        // grad shape: (batch_size, sequence_length, hidden_size)

        // Extract gradient matrices from the input Tensor.
        vector<MatrixXd> gradVec;
        gradVec.reserve(grad.data.size());
        for (const auto &var : grad.data) {
            try {
                gradVec.emplace_back(get<MatrixXd>(var));
            }
            catch (const bad_variant_access&) {
                throw runtime_error("SimpleRNN backward expects Tensor to contain MatrixXd.");
            }
        }

        // Number of samples in the batch and sequence length from stored inputs.
        int batch_size = inputs.size();
        int sequence_length = inputs[0].rows();

        learning_rate *= batch_size;

        // cout << "rnn grad\n" <<

        // Initialize gradient accumulators for weights and bias.
        MatrixXd dW_xh = MatrixXd::Zero(hidden_size, input_size);
        MatrixXd dW_hh = MatrixXd::Zero(hidden_size, hidden_size);
        MatrixXd db = MatrixXd::Zero(hidden_size, 1);

        // dL_dx will hold gradients with respect to the RNN inputs.
        vector<MatrixXd> dL_dx(batch_size, MatrixXd::Zero(sequence_length, input_size));

        // dL_dh_next will hold the gradient propagated from future time steps.
        MatrixXd dL_dh_next = MatrixXd::Zero(batch_size, hidden_size);
        MatrixXd dL_dh_t;  // Gradient at current time step

        // If not full sequence, then gradVec is of shape (1, batch_size, hidden_size).
        // Initialize dL_dh_t
        if (!return_sequences) {
            dL_dh_t = gradVec[0];
        }

        // Loop backwards through time.
        for(int t = sequence_length - 1; t >= 0; t--) {
            MatrixXd dL_dh_at_t = MatrixXd::Zero(batch_size, hidden_size);
            MatrixXd h_states_at_t = MatrixXd::Zero(batch_size, hidden_size);
            MatrixXd h_states_at_t_minus_1 = MatrixXd::Zero(batch_size, hidden_size);
            MatrixXd inputs_at_t = MatrixXd::Zero(batch_size, input_size);

            // Gather the necessary slices for each sample.
            for(int i = 0; i < batch_size; i++) {
                // h_states[i] is stored during forward pass.
                h_states_at_t.row(i) = h_states[i].row(t);
                inputs_at_t.row(i) = inputs[i].row(t);
                if(return_sequences) {
                    // For full sequence, the gradient for time t is provided.
                    dL_dh_at_t.row(i) = gradVec[i].row(t);
                }
                if(t > 0) {
                    h_states_at_t_minus_1.row(i) = h_states[i].row(t - 1);
                }
            }

            // If full sequence, combine the gradient from the output and the backpropagated gradient.
            if(return_sequences) {
                dL_dh_t = dL_dh_at_t + dL_dh_next;
            }
            else {
                // For the non-full sequence case, only the last time step has an external gradient.
                // For earlier time steps, dL_dh_t comes solely from backpropagation.
                if (t < sequence_length - 1) {
                    dL_dh_t = dL_dh_next;
                }
            }

            // Compute dtanh = (1 - h_t^2) ∘ dL_dh_t elementwise.
            MatrixXd dtanh = (1 - h_states_at_t.array().square()).matrix().cwiseProduct(dL_dh_t);

            // Accumulate gradients.
            dW_xh += dtanh.transpose() * inputs_at_t;
            dW_hh += dtanh.transpose() * (t > 0 ? h_states_at_t_minus_1 : MatrixXd::Zero(batch_size, hidden_size));
            db += dtanh.colwise().sum().transpose();

            // Compute gradient with respect to input at time t.
            MatrixXd dL_dx_t = dtanh * W_xh;
            for(int i = 0; i < batch_size; i++) {
                dL_dx[i].row(t) = dL_dx_t.row(i);
            }
            // Propagate gradient to previous hidden state.
            dL_dh_next = dtanh * W_hh;

            // In the non-full sequence case, for t == sequence_length - 1 the initial
            // dL_dh_t was already set from the external gradient.
        }
        W_xh -= learning_rate * dW_xh;
        W_hh -= learning_rate * dW_hh;
        b_h -= learning_rate * db;

        // Package the gradient with respect to input into a Tensor.
        Tensor ret;
        for (const auto &mat : dL_dx) {
            ret.data.emplace_back(mat);
        }
        return ret;
    }

private:
    MatrixXd orthogonalInit(int size) {
        MatrixXd A = randomUniformMatrix(size, size, -1.0, 1.0);
        JacobiSVD svd(A, ComputeFullU);
        return svd.matrixU();
    }
};


class Linear : public Layer {
public:
    int hidden_size;        // number of output units
    string activation;      // e.g., "relu", "sigmoid", etc.
    MatrixXd W, b;          // W: (hidden_size x input_size), b: (hidden_size x 1)
    MatrixXd inputs;         // Stored inputs from forward pass (shape: (batch_size, input_size))
    MatrixXd output;        // Stored pre-activation output (shape: (hidden_size, batch_size))

    Linear(int input_size, int hidden_size, string activation) : hidden_size(hidden_size), activation(activation) {
        double limit = sqrt(6.0 / (input_size + hidden_size));

        W = randomUniformMatrix(hidden_size, input_size, -limit, limit);
        b = MatrixXd::Zero(hidden_size, 1);
    }

    // Forward pass: accepts a Tensor containing a vector of MatrixVariant holding MatrixXd.
    // We assume that each element in the Tensor represents one sample (row vector) with shape (1 x input_size).
    // We combine them into a single MatrixXd of shape (batch_size, input_size).
    Tensor forward(const Tensor& input) override {

        // Extract MatrixXd from Tensor.
        vector<MatrixXd> x;
        x.reserve(input.data.size());
        for (const auto &var : input.data) {
            try {
                x.emplace_back(get<MatrixXd>(var));
            }
            catch (const bad_variant_access&) {
                throw runtime_error("Linear layer expects input Tensor to contain MatrixXd.");
            }
        }
        int batch_size = x.size();
        int input_cols = x[0].cols(); // assuming each sample is a row vector
        MatrixXd mat_x(batch_size, input_cols);
        for (int i = 0; i < batch_size; i++) {
            mat_x.row(i) = x[i];
        }

        // Save input for backward pass.
        this->inputs = mat_x;

        // Compute z = W * x^T + b, where:
        //   - W has shape (hidden_size, input_size)
        //   - x^T has shape (input_size, batch_size)
        //   - b is (hidden_size, 1) and is replicated to (hidden_size, batch_size)
        MatrixXd z = W * mat_x.transpose() + b.replicate(1, batch_size);

        // Apply activation function.
        // Here, activation_function(z) returns a matrix with the same shape as z (i.e. (hidden_size, batch_size)).
        this->output = activation_function(z);

        // We return the output as a Tensor, transposing z so that the output shape becomes (batch_size, hidden_size).
        MatrixXd output_mat = this->output.transpose();
        Tensor output;
        output.data.emplace_back(output_mat);
        return output;  // Output shape: (batch_size, hidden_size)
    }

    // Backward pass: accepts a Tensor whose first element is a MatrixXd with shape (batch_size, hidden_size)
    // representing the gradient from the next layer. It returns a Tensor containing the gradient with respect to
    // the Linear layer's input.
    Tensor backward(const Tensor& grad, double learning_rate) override {
        // Extract the gradient matrix.
        vector<MatrixXd> gradVec;
        for (const auto &var : grad.data) {
            try {
                gradVec.emplace_back(get<MatrixXd>(var));
            }
            catch (const bad_variant_access&) {
                throw runtime_error("Linear backward expects grad Tensor to contain MatrixXd.");
            }
        }
        if (gradVec.empty()) {
            throw runtime_error("Linear backward: empty grad Tensor.");
        }

        // grad_mat has shape (batch_size, hidden_size)
        MatrixXd grad_mat = gradVec[0];

        int batch_size = grad_mat.rows();
        learning_rate *= batch_size;

        // Compute derivative of the activation function.
        // Note: this->output is the activated z of shape (hidden_size, batch_size)
        // grad_mat.transpose() is (hidden_size, batch_size), so elementwise multiplication works.
        MatrixXd dz = grad_mat.transpose().array() * activation_derivative(output).array();

        // Compute gradients for weights and biases.
        MatrixXd dW = dz * inputs; // (hidden_size, batch_size) * (batch_size, input_size) = (hidden_size, input_size)
        dW /= batch_size;
        MatrixXd db = dz.rowwise().sum(); // sum over batch: shape (hidden_size, 1)
        db /= batch_size;  // Averaging over batch

        // Update parameters
        W -= learning_rate * dW;
        b -= learning_rate * db;

        // Compute gradient for the previous layer: dL/dx = (W^T * dz)^T.
        MatrixXd dL_dx = (W.transpose() * dz).transpose(); // shape: (batch_size, input_size)
        Tensor output;
        output.data.emplace_back(dL_dx);
        return output;  // Shape: (batch_size, input_size)
    }

private:
    MatrixXd activation_function(const MatrixXd& z) {
        // Softmax activation
        MatrixXd exp_z = (z.array().rowwise() - z.colwise().maxCoeff().array()).exp();
        exp_z.array().rowwise() /= exp_z.colwise().sum().array();
        return exp_z;
    }

    MatrixXd activation_derivative(const MatrixXd& a) {
        // Softmax derivative (element-wise)
        // For simplicity, we assume that `a` is the output from softmax activation
        return a.array() * (1 - a.array());
    }
};

class Sequential_ {
public:
    vector<shared_ptr<Layer>> layers;

    Sequential_() {}

    void add(const shared_ptr<Layer>& layer) {
        layers.emplace_back(layer);
    }

    Tensor forward(Tensor x) {
        // cout << "forward\n";
        for(const auto& layer: layers) {
            x = layer->forward(x);
            // x.print();
        }
        return x;
    }
    double backward(Tensor X, vector<int> y_true, double learning_rate) {
        auto y_pred = forward(X);
        // cout << "y_pred\n";
        // y_pred.print();
        auto loss_grad = cross_entropy_gradient(y_pred, y_true);

        for(int i = layers.size() - 1; i >= 0; i--) {
            loss_grad = layers[i]->backward(loss_grad, learning_rate);
        }

        double loss = cross_entropy_loss(y_pred, y_true);
        return loss;
    }

    MatrixXd fit(MatrixXi X_train, vector<int> y_train, int epochs, int batch_size, int num_classes, double learning_rate = 0.01, bool print_predictions=true) {

        int num_samples = X_train.rows();

        for(int epoch = 1; epoch <= epochs; epoch++) {

            double totalLoss = 0;

            for(int i = 0; i < num_samples; i++) {
                const int start_index = i * batch_size;
                const int end_index = min(start_index + batch_size, num_samples);

                if(start_index >= num_samples)
                    break;

                MatrixXi mat = X_train.block(start_index, 0, end_index - start_index, X_train.cols());

                Tensor tensor;
                tensor.data = vector<MatrixVariant>(1, mat);
                vector labels(y_train.begin() + start_index, y_train.begin() + end_index);

                double loss = backward(tensor, labels, learning_rate);
                totalLoss += loss * (end_index - start_index);
            }

            if(epoch == 1 || epoch % 10 == 0 || epoch == epochs) {
                printf("Epoch = %02d    Loss: %.5lf\n", epoch, totalLoss / num_samples);
            }
        }
        cout << "\n\n";

        MatrixXd predictions(num_samples, num_classes);

        for(int i = 0; i < num_samples; i++) {
            Tensor tensor;
            MatrixXi mat = X_train.row(i);
            tensor.data = vector<MatrixVariant>(1, mat);
            Tensor predictionTensor = forward(tensor);
            const MatrixXd prediction = get<MatrixXd>(predictionTensor.data[0]);

            if(print_predictions) {
                cout << prediction << "  ------  " << prediction.maxCoeff() << "\n";
            }
            predictions.row(i) = prediction.row(0);
        }
        return predictions;
    }
};


int main() {

    int vocab_size = 1000;
    int embedding_dim = 32;
    int max_sequence_length = 10;
    int num_sequences = 50;
    int num_classes = 5;
    int epochs = 20;
    int batch_size = 32;

    MatrixXi data(num_sequences, max_sequence_length);
    vector<int> labels = vector<int>(num_sequences);

    for(int i = 0; i < num_sequences; i++) {
        labels[i] = uniform_int_distribution(0, num_classes - 1)(rng);
        for(int j = 0; j < max_sequence_length; j++) {
            data(i, j) = uniform_int_distribution(0, vocab_size - 1)(rng);
        }
    }

    cout << data << "\n\n";

    Embedding embedding = Embedding(vocab_size, embedding_dim);

    vector hidden_size = {64, 128, 128};

    int linear_size = 5;

    Sequential_ model = Sequential_();

    model.add(make_shared<Embedding>(Embedding(vocab_size, embedding_dim)));

    for(int i = 0; i < hidden_size.size(); i++) {
        model.add(make_shared<SimpleRNN>(SimpleRNN(i == 0 ? embedding_dim : hidden_size[i - 1], hidden_size[i], i != hidden_size.size() - 1)));
    }

    model.add(make_shared<Linear>(hidden_size.back(), linear_size, "softmax"));

    auto start = high_resolution_clock::now();

    MatrixXd predictions = model.fit(data, labels, epochs, batch_size, num_classes);

    auto end = high_resolution_clock::now();

    auto totalTime = duration_cast<milliseconds>(end - start);

    vector<int> predictedLabels(num_sequences);

    for(int i = 0; i < num_sequences; i++) {
        predictions.row(i).maxCoeff(&predictedLabels[i]);
    }

    cout << "correct labels =       ";

    for(auto& val: labels) {
        cout << val  << " ";
    }

    cout << "\npredicted labels =     ";

    for(auto& val: predictedLabels) {
        cout << val << " ";
    }
    cout << "\n";

    int correct_prediction = 0;

    for(int i = 0; i < num_sequences; i++) {
        if(predictedLabels[i] == labels[i]) {
            correct_prediction += 1;
        }
    }

    cout << "Correct prediction = " << correct_prediction << "\n";

    cout << "Total time = " << static_cast<double>(totalTime.count()) / 1000 << " seconds\n";
}
