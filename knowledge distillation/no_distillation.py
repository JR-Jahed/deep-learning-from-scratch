"""
Train the models and check their performance without knowledge distillation.
"""


import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
import os
import model_summary
from my_dataset import MyDataset
import time
import cnn_models

device = torch.device("cpu")
train_dataset_path = "../cnn custom data/Dataset/Train"
test_dataset_path = "../cnn custom data/Dataset/Test"
model_directory = './Saved Models/Undistilled Models'

num_classes = len(os.listdir(train_dataset_path))

image_height = 128
image_width = 128

transform = transforms.Compose([
    transforms.Resize((image_height, image_width)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

train_dataset = MyDataset(train_dataset_path, transform=transform)
test_dataset = MyDataset(test_dataset_path, transform=transform)

# Create DataLoaders for both training and testing
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

train_size = train_dataset.__len__()
test_size = test_dataset.__len__()


def train_model(model, model_path, epochs):

    # training details is stored in this file
    file_path = model_path[:model_path.rfind('.')] + '.txt'

    if os.path.exists(file_path):
        file = open(file_path, 'r+')
        last_line = file.readlines()[-1]
        epoch_part = last_line[:last_line.find('L')].strip()
        start_epoch = int(epoch_part[epoch_part.rfind(' ') + 1:]) + 1
    else:
        file = open(file_path, 'x')
        start_epoch = 1

    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, weights_only=True))

    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=.001, weight_decay=.0001)

    total_start = time.time()

    for epoch in range(start_epoch, start_epoch + epochs):
        running_loss = 0
        correct = 0
        start = time.time()
        for i, data in enumerate(train_loader):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model.forward(inputs)

            _, predicted = torch.max(outputs.data, 1)

            correct += (predicted == labels).sum().item()

            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() / len(inputs)

        logger = f"Epoch = {epoch:02}   Loss = {running_loss:05.2f}   Accuracy = {100 * correct / train_size:05.2f}   Total time = {time.time() - start:05.2f}\n"
        print(logger)
        file.write(logger)

    torch.save(model.state_dict(), model_path)
    end = time.time()
    print(f"Total time = {end - total_start:05.2f}")


model = cnn_models.Model7(num_classes=num_classes)
# model_summary.check_output_shape_before_fc(model, next(iter(train_loader))[0].to(device))
# model_summary.summary(model, input_shape=(3, image_height, image_width))
# exit(0)

model_path = os.path.join(model_directory, f"{model.name}.pth")

# training = True
training = False

if training:
    train_model(model, model_path, epochs=10)
else:

    correct = 0
    model.load_state_dict(torch.load(model_path, weights_only=True))

    start = time.time()

    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()

    end = time.time()
    file_path = model_path[:model_path.rfind('.')] + '.txt'

    file = open(file_path, 'a')
    logger = f"   Test Accuracy = {100 * correct / test_size:05.2f} Test Time = {end - start:05.2f}"
    print(logger)

    file.write(logger)
