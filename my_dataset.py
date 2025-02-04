from torch.utils.data import Dataset
from torchvision import transforms
import os
from PIL import Image

class MyDataset(Dataset):
    def __init__(self, dataset_path, transform=None):
        super(MyDataset, self).__init__()

        self.dataset_path = dataset_path
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize(size=(64, 64)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
        else:
            self.transform = transform

        # Walk through all subdirectories and collect image paths and corresponding labels
        self.image_paths = []
        self.labels = []
        self.classes = sorted(os.listdir(dataset_path))  # Assuming each subdirectory is a class

        for label, class_name in enumerate(self.classes):
            class_path = os.path.join(dataset_path, class_name)
            if os.path.isdir(class_path):
                # Find all image files in the class directory
                for img_name in os.listdir(class_path):
                    img_path = os.path.join(class_path, img_name)
                    if img_path.lower().endswith(('png', 'jpg', 'jpeg')):  # You can add more image extensions
                        self.image_paths.append(img_path)
                        self.labels.append(label)

    def __getitem__(self, index):
        img_path = self.image_paths[index]
        label = self.labels[index]

        # Open the image
        img = Image.open(img_path).convert('RGB')

        # Apply transformations
        if self.transform:
            img = self.transform(img)

        return img, label

    def __len__(self):
        return len(self.image_paths)