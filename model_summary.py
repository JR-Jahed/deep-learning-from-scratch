import torch.nn as nn

def print_shape_hook(module, input, output):
    print(f"{module.__class__.__name__} output shape: {output.shape}")


def check_output_shape_before_fc(model, input):
    for layer in model.children():
        if isinstance(layer, (nn.Conv2d, nn.MaxPool2d)):
            layer.register_forward_hook(print_shape_hook)

    model(input)


def summary(model, input_shape, first_call=True):
    input_width = input_shape[1]

    total_parameters = 0

    for name, module in model.named_children():
        parameters_for_current_layer = 0
        if type(module) is nn.Conv2d:
            parameters_for_current_layer = module.in_channels * module.out_channels * module.kernel_size[0] * module.kernel_size[1] + module.out_channels
            total_parameters += parameters_for_current_layer
            input_width -= (1 - module.padding[0]) * 2
            input_shape = (input_width, input_width, module.out_channels)

        elif type(module) is nn.Linear:
            parameters_for_current_layer = module.in_features * module.out_features + module.out_features
            total_parameters += parameters_for_current_layer
            input_shape = module.out_features

        elif type(module) is nn.MaxPool2d:
            input_width //= 2
            input_shape = (input_width, input_width, input_shape[2])

        elif type(module) is nn.Sequential:
            total_parameters += summary(module, input_shape, False)

        combined_str = f"{name} ({module.__class__.__name__})"

        print(f"{combined_str:<20} {str(input_shape):<20} {parameters_for_current_layer:<15}")

    if first_call:
        print("\nTotal parameters = ", total_parameters, "\n")

    return total_parameters
