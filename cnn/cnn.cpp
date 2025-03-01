#include <bits/stdc++.h>
using namespace std;
using namespace chrono;

/*

_________________________________________________________________________________________________________

This is the C++ implementation of CNN. It's significantly faster than Python.

IDE: CLion (Downloaded using my Queen's email)


This is entirely for learning. There is no emphasis on accuracy or optimisation. Implementation
of CNN from scratch without resorting to any library such as PyTorch or Tensorflow is the aim here.

_________________________________________________________________________________________________________

*/


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

class ConvPool {
public:
    virtual ~ConvPool() = default;  // Virtual destructor for proper cleanup
    virtual vector<vector<vector<vector<double>>>> forward(vector<vector<vector<vector<double>>>>& input) = 0;     // Pure virtual forward method
    virtual vector<vector<vector<vector<double>>>> backward(vector<vector<vector<vector<double>>>>& dL_dout, double learning_rate) = 0;    // Pure virtual backward method
};

class Conv2d : public ConvPool {
public:
    vector<vector<vector<vector<double>>>> input;
    vector<vector<vector<vector<bool>>>> relu_mask;
    int input_channels;  // Number of input channels
    int output_channels; // Number of output channels
    pair<int, int> kernel_size;  // Kernel size (3x3)
    vector<vector<vector<vector<double>>>> weights;  // Filters, each of size 3x3
    vector<double> biases;  // Bias for each filter
    string initializer;

    Conv2d() {

    }

    Conv2d(const int input_channels, const int output_channels, const auto kernel_size={3, 3}, const string& initializer = "glorot") :
    input_channels(input_channels), output_channels(output_channels), kernel_size(kernel_size), initializer(initializer) {

        weights = vector(output_channels, vector(input_channels, vector(kernel_size.first, vector<double>(kernel_size.second))));
        biases = vector<double>(output_channels);

        if(initializer == "glorot") {
            const double limit = sqrt(6.0f / static_cast<float>(input_channels + output_channels));
            const uniform_real_distribution dist(-limit, limit);

            initializeWeights(dist);
            initializeBiases(dist);
        }
        else if(initializer == "he") {
            const double limit = sqrt(6.0f / static_cast<float>(input_channels));
            const uniform_real_distribution dist(-limit, limit);

            initializeWeights(dist);
            initializeBiases(dist);
        }
        else {
            initializeWeights(normal_dist);
            initializeBiases(normal_dist);
        }
    }

    vector<vector<vector<vector<double>>>> forward(vector<vector<vector<vector<double>>>>& input) override {

        // Save input for backpropagation
        this->input = input;

        const int batch_size = input.size();
        const int input_height = input[0].size();
        const int input_width = input[0][0].size();
        const int input_channels = input[0][0][0].size();

        const int kernel_height = kernel_size.first;
        const int kernel_width = kernel_size.second;

        const int output_height = input_height - kernel_height + 1;
        const int output_width = input_width - kernel_width + 1;

        auto output = vector(batch_size, vector(output_height, vector(output_width, vector<double>(output_channels))));

        for(int i = 0; i < batch_size; i++) {
            for(int h = 0; h < output_height; h++) {
                for(int w = 0; w < output_width; w++) {
                    for(int oc = 0; oc < output_channels; oc++) {


                        /*

                        For each position of the kernel on the input channels, the output for that particular position is the sum of
                        elementwise multiplication for all input channels. The kernel starts to slide over the input from [0, 0].
                        If there are 32 input channels, the output will be calculated as the sum of the hadamard product when kernel
                        is placed on the same position for all 32 channels.

                        */


                        double sum_value = 0.0;

                        for(int ic = 0; ic < input_channels; ic++) {
                            for(int kh = 0; kh < kernel_height; kh++) {
                                for(int kw = 0; kw < kernel_width; kw++) {
                                    const double input_value  = input[i][h + kh][w + kw][ic];
                                    const double kernel_value = weights[oc][ic][kh][kw];

                                    sum_value += input_value * kernel_value;
                                }
                            }
                        }
                        output[i][h][w][oc] = sum_value + biases[oc];
                    }
                }
            }
        }

        relu_mask = vector(batch_size, vector(output_height, vector(output_width, vector<bool>(output_channels))));

        for(int i = 0; i < batch_size; i++) {
            for(int h = 0; h < output_height; h++) {
                for(int w = 0; w < output_width; w++) {
                    for(int oc = 0; oc < output_channels; oc++) {
                        // Mark the output positions for which ReLU produced 0
                        relu_mask[i][h][w][oc] = output[i][h][w][oc] > 0;

                        // ReLU
                        output[i][h][w][oc] = max(0.0, output[i][h][w][oc]);
                    }
                }
            }
        }

        return output;
    }

    vector<vector<vector<vector<double>>>> backward(vector<vector<vector<vector<double>>>>& dL_dout, const double learning_rate) override {

        const int batch_size = input.size();
        const int input_height = input[0].size();
        const int input_width = input[0][0].size();
        const int input_channels = input[0][0][0].size();

        const int kernel_height = kernel_size.first;
        const int kernel_width = kernel_size.second;

        // Gradients of all the weights of this layer
        auto dL_dweights = vector(output_channels, vector(input_channels, vector(kernel_size.first, vector<double>(kernel_size.second, 0))));

        // Gradients of all the biases
        auto dL_dbiases = vector<double>(output_channels);

        // Gradients w.r.t. the inputs of this layer which will be used to calculate the gradients of the previous layer unless it's the first layer
        auto dL_din = vector(batch_size, vector(input_height, vector(input_width, vector<double>(input_channels, 0))));

        for(int i = 0; i < batch_size; i++) {
            for(int h = 0; h < dL_dout[0].size(); h++) {
                for(int w = 0; w < dL_dout[0][0].size(); w++) {
                    for(int oc = 0; oc < output_channels; oc++) {
                        // If ReLU produced 0 during forward pass for an output, its gradient will not be counted
                        dL_dout[i][h][w][oc] *= relu_mask[i][h][w][oc] ? 1 : 0;
                    }
                }
            }
        }

        /*

        First loop goes through all the images in the batch
        Second loop goes through the rows of the activation maps of this layer
        Third loop goes through the columns of the activation maps of this layer
        Fourth loop goes through the output channels of this layer
        Fifth loop goes through the input channels of this layer
        Sixth and Seventh loops go through the kernel height and width respectively

        */

        for(int i = 0; i < batch_size; i++) {
            for(int h = 0; h < dL_dout[0].size(); h++) {
                for(int w  = 0; w < dL_dout[0][0].size(); w++) {
                    for(int oc = 0; oc < output_channels; oc++) {

                        /*

                        The bias value of a neuron was used to calculate all the output values of an activation map.
                        Therefore, its gradient is calculated by summing the gradients of all the output values.

                        */

                        dL_dbiases[oc] += dL_dout[i][h][w][oc];

                        for(int ic = 0; ic < input_channels; ic++) {
                            for(int kh = 0; kh < kernel_height; kh++) {
                                for(int kw = 0; kw < kernel_width; kw++) {
                                    const double input_value = input[i][h + kh][w + kw][ic];

                                    /*

                                    During forward pass kernel slid over all the possible positions of the input channels.

                                    Therefore, the gradient of a weight (dL_dweights[oc][ic][kh][kw]) will be calculated using the gradients of all
                                    the outputs that were calculated using this weight and its corresponding input value

                                    In other words, the gradient of a particular output position (dL_dout[i][h][w][oc]) will be used to calculate the
                                    gradients of all the weights that were used to calculate this particular output position

                                    */

                                    dL_dweights[oc][ic][kh][kw] += input_value * dL_dout[i][h][w][oc];


                                    /*

                                    Similarly, the gradient of a particular input position (dL_din[i][h + kh][w + kw][ic]) will be calculated using the
                                    weight (weights[oc][ic][kh][kw]) associated with that input position and the gradient of the output position which
                                    was calculated using this input value

                                    In other words, gradient of every output value (dL_dout[i][h][w][oc]) will be used to calculate the gradient of all
                                    the input values that were used to calculate this output value

                                    */

                                    dL_din[i][h + kh][w + kw][ic] += weights[oc][ic][kh][kw] * dL_dout[i][h][w][oc];
                                }
                            }
                        }
                    }
                }
            }
        }

        for(int oc = 0; oc < output_channels; oc++) {
            for(int ic = 0; ic < input_channels; ic++) {
                for(int kh = 0; kh < kernel_height; kh++) {
                    for(int kw = 0; kw < kernel_width; kw++) {
                        weights[oc][ic][kh][kw] -= learning_rate * dL_dweights[oc][ic][kh][kw];
                    }
                }
            }
            biases[oc] -= learning_rate * dL_dbiases[oc];
        }

        return dL_din;  // Pass gradient to the previous layer
    }

private:
    void initializeWeights(auto dist) {
        for(auto& tensor: weights) {
            for(auto& matrix: tensor) {
                for(auto& row: matrix) {
                    for(auto& val: row) {
                        val = dist(rng);
                    }
                }
            }
        }
    }
    void initializeBiases(auto dist) {
        for(auto& val: biases) {
            val = dist(rng);
        }
    }
};

class Dense {
public:
    vector<vector<double>> input;
    int input_channels;
    int output_channels;
    string activation;
    vector<vector<double>> weights;
    vector<double> biases;
    string initializer;

    Dense(const int input_channels, const int output_channels, const string activation, const string& initializer = "glorot") :
    input_channels(input_channels), output_channels(output_channels), activation(activation), initializer(initializer) {

        weights = vector(input_channels, vector<double>(output_channels));
        biases = vector<double>(output_channels);

        if(initializer == "glorot") {
            const double limit = sqrt(6.0f / static_cast<float>(input_channels + output_channels));
            const uniform_real_distribution dist(-limit, limit);

            initializeWeights(dist);
            initializeBiases(dist);
        }
        else if(initializer == "he") {
            const double limit = sqrt(6.0f / static_cast<float>(input_channels));
            const uniform_real_distribution dist(-limit, limit);

            initializeWeights(dist);
            initializeBiases(dist);
        }
        else {
            initializeWeights(normal_dist);
            initializeBiases(normal_dist);
        }
    }

    vector<vector<double>> forward(const auto& input) {
        this->input = input;

        auto output = vector<vector<double>>();

        for(auto& row: input) {
            vector row_output(output_channels, 0.0);

            for(int i = 0; i < output_channels; i++) {
                for(int j = 0; j < input_channels; j++) {
                    row_output[i] += row[j] * weights[j][i];
                }
                row_output[i] += biases[i];
            }
            output.emplace_back(row_output);
        }

        output = activation_function(output);
        return output;
    }

    vector<vector<double>> backward(const vector<vector<double>>& dL_dout, const double learning_rate) {

        const int batch_size = input.size();
        vector<vector<double>> dL_din(input.size(), vector<double>(input[0].size(), 0.0));  // Gradient w.r.t. input
        vector<vector<double>> dL_dweights = vector(input_channels, vector<double>(output_channels, 0));  // Gradient w.r.t. weights
        vector<double> dL_dbiases(output_channels, 0.0);  // Gradient w.r.t. biases

        for(int i = 0; i < batch_size; i++) {
            for(int oc = 0; oc < output_channels; oc++) {
                dL_dbiases[oc] = dL_dout[i][oc];
                for(int ic = 0; ic < input[0].size(); ic++) {
                    dL_dweights[ic][oc] += dL_dout[i][oc] * input[i][ic];
                    dL_din[i][ic] += weights[ic][oc] * dL_dout[i][oc];
                }
            }
        }

        for(int oc = 0; oc < output_channels; oc++) {
            dL_dbiases[oc] /= batch_size;
            for(int ic = 0; ic < input_channels; ic++) {
                dL_dweights[ic][oc] /= batch_size;
            }
        }

        for(int oc = 0; oc < output_channels; oc++) {
            biases[oc] -= learning_rate * dL_dbiases[oc];
            for(int ic = 0; ic < input.size(); ic++) {
                weights[ic][oc] -= learning_rate * dL_dweights[ic][oc];
            }
        }

        return dL_din;  // Pass gradient to the previous layer
    }

private:
    vector<vector<double>> activation_function(vector<vector<double>>& values) {
        vector result(values.size(), vector<double>(values[0].size(), 0.0));

        if(activation == "relu") {
            for(int i = 0; i < values.size(); i++) {
                for(int j = 0; j < values[0].size(); j++) {
                    result[i][j] = max(0.0, values[i][j]);
                }
            }
        }
        else {
            for(int i = 0; i < values.size(); i++) {
                // Find max value in row for numerical stability
                const double max_value = *max_element(values[i].begin(), values[i].end());

                // Compute exp(values - max_value) for numerical stability
                vector<double> exp_values(values[i].size());

                for(int j = 0; j < values[i].size(); j++) {
                    exp_values[j] = exp(values[i][j] - max_value);
                }

                // Compute the sum of exp_values
                const double sum_exp_values = accumulate(exp_values.begin(), exp_values.end(), 0.0);

                // Normalize the exp_values to get softmax
                for(int j = 0; j < values[i].size(); j++) {
                    result[i][j] = exp_values[j] / sum_exp_values;
                }
            }
        }
        return result;
    }

    void initializeWeights(auto dist) {

        for(auto& row: weights) {
            for(auto& val: row) {
                val = dist(rng);
            }
        }
    }

    void initializeBiases(auto dist) {
        for(auto& val: biases) {
            val = dist(rng);
        }
    }
};

class MaxPooling2D: public ConvPool {
public:
    vector<vector<vector<vector<double>>>> input;

    vector<vector<vector<vector<double>>>> forward(vector<vector<vector<vector<double>>>>& input) override {
        this->input = input;

        const int batch_size = input.size();
        const int height = input[0].size();
        const int width = input[0][0].size();
        const int channels = input[0][0][0].size();

        const int output_height = height / 2;
        const int output_width = width / 2;

        vector output(batch_size, vector(output_height, vector(output_width, vector<double>(channels))));

        for(int i = 0; i < batch_size; i++) {
            for(int j = 0; j < output_height; j++) {
                for(int k = 0; k < output_width; k++) {
                    for(int l = 0; l < channels; l++) {
                        output[i][j][k][l] = max({input[i][j * 2][k * 2][l], input[i][j * 2][k * 2 + 1][l],
                                             input[i][j * 2 + 1][k * 2][l], input[i][j * 2 + 1][k * 2 + 1][l]});
                    }
                }
            }
        }
        return output;
    }

    vector<vector<vector<vector<double>>>> backward(vector<vector<vector<vector<double>>>>& dL_dout, double learning_rate) override {
        const int batch_size = input.size();
        const int input_height = input[0].size();
        const int input_width = input[0][0].size();
        const int input_channels = input[0][0][0].size();

        vector<vector<vector<vector<double>>>> dL_din(batch_size, vector(input_height, vector(input_width, vector<double>(input_channels))));

        for(int i = 0; i < batch_size; i++) {
            for(int h = 0; h < dL_dout[0].size(); h++) {
                for(int w = 0; w < dL_dout[0][0].size(); w++) {
                    for(int oc = 0; oc < dL_dout[0][0][0].size(); oc++) {

                        vector region(2, vector<double>(2));  // 2x2 region
                        for (int i_row = 0; i_row < 2; ++i_row) {
                            for (int i_col = 0; i_col < 2; ++i_col) {
                                region[i_row][i_col] = input[i][h * 2 + i_row][w * 2 + i_col][oc];
                            }
                        }

                        // Find the index of the max value in the region (2x2)
                        int max_row = 0, max_col = 0;
                        double max_val = region[0][0];
                        for (int i_row = 0; i_row < 2; ++i_row) {
                            for (int i_col = 0; i_col < 2; ++i_col) {
                                if (region[i_row][i_col] > max_val) {
                                    max_val = region[i_row][i_col];
                                    max_row = i_row;
                                    max_col = i_col;
                                }
                            }
                        }
                        dL_din[i][h * 2 + max_row][w * 2 + max_col][oc] += dL_dout[i][h][w][oc];
                    }
                }
            }
        }
        return dL_din;
    }
};

class Sequential {
public:
    vector<shared_ptr<ConvPool>> conv_pool_layers;
    vector<Dense> dense_layers;
    vector<vector<vector<vector<vector<double>>>>> conv_pool_outputs;
    vector<vector<vector<double>>> dense_outputs;
    tuple<int, int, int, int> last_conv_output_shape;

    Sequential() {}

    void addConvPoolLayer(const shared_ptr<ConvPool>& layer) {
        conv_pool_layers.push_back(layer);
    }

    void addDenseLayer(const Dense& dense_layer) {
        dense_layers.push_back(dense_layer);
    }

    vector<vector<double>> fit(const int epochs, const vector<vector<vector<vector<double>>>>& input_images,
                                const vector<int>& labels, const int batch_size, const double learning_rate) {

        for(int epoch = 1; epoch <= epochs; epoch++) {

            double total_loss = 0;

            for(int i = 0; i < input_images.size(); i++) {
                const int start_index = i * batch_size;
                const int end_index = min(start_index + batch_size, static_cast<int>(input_images.size()));
                if(start_index >= input_images.size())
                    break;

                vector<vector<double>> batch_predictions = forward(vector(input_images.begin() + start_index, input_images.begin() + end_index));
                vector<vector<double>> dL_dout = cross_entropy_gradient(batch_predictions, vector(labels.begin() + start_index, labels.begin() + end_index));
                backward(dL_dout, learning_rate);
                total_loss += cross_entropy_loss(batch_predictions, vector(labels.begin() + start_index, labels.begin() + end_index)) * (end_index - start_index + 1);
            }

            if(epoch == 1 || epoch % 10 == 0 || epoch == epochs) {
                printf("Epoch = %02d    Loss: %.5lf\n", epoch, total_loss / input_images.size());
            }
        }
        cout << "\n";

        cout << "Predictions\n";

        auto predictions = forward(input_images);

        for(auto& prediction: predictions) {
            for(auto& val: prediction) {
                cout << val << " ";
            }
            cout << "\n";
        }
        cout << "\n";

        return predictions;
    }

    vector<vector<double>> forward(const vector<vector<vector<vector<double>>>>& input) {
        conv_pool_outputs.clear();
        dense_outputs.clear();

        vector<vector<vector<vector<double>>>> current_input = input;

        // Forward pass through convolutional layers
        for(auto& conv_pool_layer : conv_pool_layers) {
            current_input = conv_pool_layer->forward(current_input);
            conv_pool_outputs.emplace_back(current_input); // Cache the output of each conv layer
        }

        last_conv_output_shape = {current_input.size(), current_input[0].size(), current_input[0][0].size(), current_input[0][0][0].size()};

        // Flatten the features maps
        vector<vector<double>> flat_input;

        for(auto& feature_maps: current_input) {
            flat_input.emplace_back(flatten(feature_maps));
        }

        // Forward pass through dense layers
        vector<vector<double>> current_dense_input = flat_input;

        for(auto& dense_layer : dense_layers) {
            current_dense_input = dense_layer.forward(current_dense_input);
            dense_outputs.emplace_back(current_dense_input); // Cache the output of each dense layer
        }

        return current_dense_input;
    }

    void backward(const vector<vector<double>>& dL_dout, double learning_rate) {
        vector<vector<double>> current_dL_dout = dL_dout;

        // Backward pass through dense layers (in reverse order)
        for(int i = dense_layers.size() - 1; i >= 0; i--) {
            current_dL_dout = dense_layers[i].backward(current_dL_dout, learning_rate);
        }

        // Backward pass through convolutional layers (in reverse order)
        vector<vector<vector<vector<double>>>> current_dL_dout_conv = unflatten(current_dL_dout);
        for(int i = conv_pool_layers.size() - 1; i >= 0; i--) {
            current_dL_dout_conv = conv_pool_layers[i]->backward(current_dL_dout_conv, learning_rate);
        }
    }

private:
    vector<double> flatten(const vector<vector<vector<double>>>& tensor) {
        vector<double> flat;
        for(const auto& matrix : tensor) {
            for(const auto& row : matrix) {
                for(const auto& value : row) {
                    flat.push_back(value);
                }
            }
        }
        return flat;
    }

    vector<vector<vector<vector<double>>>> unflatten(const vector<vector<double>>& flat) {

        int dim1 = get<0>(last_conv_output_shape);
        int dim2 = get<1>(last_conv_output_shape);
        int dim3 = get<2>(last_conv_output_shape);
        int dim4 = get<3>(last_conv_output_shape);

        vector tensor(dim1, vector(dim2, vector(dim3, vector<double>(dim4))));

        for(int i = 0; i < dim1; i++) {
            int index = 0;
            for(int j = 0; j < dim2; j++) {
                for(int k = 0; k < dim3; k++) {
                    for(int l = 0; l < dim4; l++) {
                        tensor[i][j][k][l] = flat[i][index];
                        index += 1;
                    }
                }
            }
        }

        return tensor;
    }
};


int main() {

    int width = 64;
    int height = 64;
    int channels = 3;

    int totalImages = 10;
    int classes = 5;
    int epochs = 30;
    int batch_size = 32;

    vector<vector<vector<vector<double>>>> images = vector(totalImages, vector(height, vector(width, vector<double>(channels))));
    vector<int> labels(totalImages);

    for(int i = 0; i < totalImages; i++) {

        labels[i] = uniform_int_distribution(0, classes - 1)(rng);

        for(auto& matrix: images[i]) {
            for(auto& row: matrix) {
                for(auto& val: row) {
                    val = uniform_int_distribution(0, 255)(rng);
                    val /= 255;
                }
            }
        }
    }
    // labels = {3, 1, 4, 2, 0, 1, 4, 3, 2, 0};

    Sequential model;

    int trainable_parameters = 0;

    vector num_output_channels = {16, -1, 32, -1, 64, -1, 64};

    int current_height = height;
    int current_width = width;

    Conv2d conv;

    for(int i = 0; i < num_output_channels.size(); i++) {
        if(num_output_channels[i] > 0) {
            conv = Conv2d(i == 0 ? channels : conv.output_channels, num_output_channels[i], pair{3, 3});
            model.addConvPoolLayer(make_shared<Conv2d>(conv));
            trainable_parameters += conv.weights.size() * conv.weights[0].size() * conv.weights[0][0].size() * conv.weights[0][0][0].size() + conv.biases.size();
            current_height -= 2;
            current_width -= 2;
        }
        else {
            model.addConvPoolLayer(make_shared<MaxPooling2D>());
            current_height = current_height / 2;
            current_width = current_width / 2;
        }
    }

    Dense dense1 = Dense(current_height * current_width * conv.output_channels, 32, "relu");
    model.addDenseLayer(dense1);

    Dense dense2 = Dense(32, classes, "softmax");
    model.addDenseLayer(dense2);

    for(auto& dense_layer: model.dense_layers) {
        trainable_parameters += dense_layer.input_channels * dense_layer.output_channels + dense_layer.output_channels;
    }

    cout << "Total trainable parameters = " << trainable_parameters << "\n\n";

    auto start = high_resolution_clock::now();

    vector<vector<double>> predictions = model.fit(epochs, images, labels, batch_size, .01);

    auto end = high_resolution_clock::now();

    auto totalTime = duration_cast<milliseconds>(end - start);

    vector<int> predicted_labels(totalImages);

    for(int i = 0; i < totalImages; i++) {
        predicted_labels[i] = max_element(predictions[i].begin(), predictions[i].end()) - predictions[i].begin();
    }

    cout << "correct labels =       ";

    for(auto& val: labels) {
        cout << val  << " ";
    }

    cout << "\npredicted labels =     ";

    for(auto& val: predicted_labels) {
        cout << val << " ";
    }
    cout << "\n";

    int correct_prediction = 0;

    for(int i = 0; i < totalImages; i++) {
        if(predicted_labels[i] == labels[i]) {
            correct_prediction += 1;
        }
    }

    cout << "Correct prediction = " << correct_prediction << "\n";

    cout << "Total time = " << static_cast<double>(totalTime.count()) / 1000 << " seconds\n";
}
