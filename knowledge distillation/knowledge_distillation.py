"""
Train the student models using the softmax outputs of teacher models.
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
train_dataset_path = "../Dataset/Train"
test_dataset_path = "../Dataset/Test"

teacher_model_directory = './Saved Models/Undistilled Models'
student_model_directory = './Saved Models/Distilled Models'

num_classes = len(os.listdir(train_dataset_path))

image_height = 128
image_width = 128


T = 4.0
alpha = 0.75


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


def softmax_with_temperature(logits, temperature):
    return nn.functional.softmax(logits / temperature, dim=-1)


def train_model_distil_knowledge(teacher, student, teacher_path, student_path, epochs):

    # training details is stored in this file
    student_file_path = student_path[:student_path.rfind('.')] + '.txt'

    if os.path.exists(student_file_path):
        student_file = open(student_file_path, 'r+')
        last_line = student_file.readlines()[-1]
        epoch_part = last_line[:last_line.find('L')].strip()
        start_epoch = int(epoch_part[epoch_part.rfind(' ') + 1:]) + 1
    else:
        student_file = open(student_file_path, 'x')
        start_epoch = 1

    if os.path.exists(teacher_path):
        teacher.load_state_dict(torch.load(teacher_path, weights_only=True))
        teacher.eval()
    
    if os.path.exists(student_path):
        student.load_state_dict(torch.load(student_path, weights_only=True))

    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(student.parameters(), lr=.001, weight_decay=.0001)

    total_start = time.time()

    for epoch in range(start_epoch, start_epoch + epochs):
        running_loss = 0
        correct = 0
        start = time.time()

        for i, data in enumerate(train_loader):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Get student outputs
            optimizer.zero_grad()
            student_outputs = student.forward(inputs)

            # Get teacher outputs (soft targets)
            with torch.no_grad():  # No gradient computation for teacher
                teacher_outputs = teacher.forward(inputs)

            # Apply temperature scaling
            soft_targets = softmax_with_temperature(teacher_outputs, T)
            student_softmax = nn.functional.log_softmax(student_outputs, dim=1)

            # Compute losses
            hard_loss = loss_function(student_softmax, labels)
            soft_loss = nn.functional.kl_div(student_softmax, soft_targets, reduction='batchmean') * (T ** 2)  # Scaled KL divergence

            # Combine losses
            loss = alpha * soft_loss + (1 - alpha) * hard_loss
            loss.backward()
            optimizer.step()

            # Compute accuracy
            _, predicted = torch.max(student_outputs.data, 1)
            correct += (predicted == labels).sum().item()
            running_loss += loss.item() / len(inputs)

        logger = f"Epoch = {epoch:02}   Loss = {running_loss:05.2f}   Accuracy = {100 * correct / train_size:05.2f}   Total time = {time.time() - start:05.2f}\n"
        print(logger)
        student_file.write(logger)

    torch.save(student.state_dict(), student_path)
    end = time.time()
    print(f"Total time = {end - total_start:05.2f}")



teacher = cnn_models.Model2(num_classes=num_classes)
student = cnn_models.Model7(num_classes=num_classes)
# model_summary.check_output_shape_before_fc(student, next(iter(train_loader))[0].to(device))
# model_summary.summary(teacher, input_shape=(3, image_height, image_width))
# model_summary.summary(student, input_shape=(3, image_height, image_width))
# exit(0)

teacher_path = os.path.join(teacher_model_directory, f"{teacher.name}.pth")
student_path = os.path.join(student_model_directory, f"{student.name}.pth")

# training = True
training = False

if training:
    train_model_distil_knowledge(teacher, student, teacher_path, student_path, epochs=10)
else:

    model = student
    model_path = student_path

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
    logger = f"   Test Accuracy = {100 * correct / test_size:05.2f}   Test time = {end - start:05.2f}"
    print(logger)

    file.write(logger)
