from torchvision import datasets
from torchvision.transforms import v2
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch

training_losses = []
validation_losses = []


raw_dataset = datasets.OxfordIIITPet(
    root="./data",
    split="trainval",
    target_types="category",
    download=True,
    transform=v2.Compose([
        v2.Resize((224, 224)),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True)
    ])
)

# stat_calc = DataLoader(raw_dataset, batch_size=32, shuffle=False)

# mean = 0
# std = 0

# for images, _ in stat_calc:
#     batch_samples = images.size(0)
#     images = images.view(batch_samples, images.size(1), -1)
#     mean = mean + images.mean(2).sum(0)
#     std = std + images.std(2).sum(0)

# mean = mean / len(stat_calc.dataset)
# std = std / len(stat_calc.dataset)

# print("Mean:"+str(mean))
# print("Std:"+str(std))


# I computed these values using the above code snippet, which I got from an online PyTorch forum. 
mean = [0.4783, 0.4459, 0.3957]
std = [0.2254, 0.2223, 0.2240]


training_dataset_full = datasets.OxfordIIITPet(
    root="./data",
    split="trainval",
    target_types="category",
    download=True,
    transform=v2.Compose([
        v2.RandomResizedCrop(224, scale=(0.8, 1.0)),
        v2.RandomHorizontalFlip(),
        v2.RandomRotation(10),
        v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=mean, std=std)
    ]))

validation_dataset_full = datasets.OxfordIIITPet(
    root="./data",
    split="trainval",
    target_types="category",
    download=True,
    transform=v2.Compose([
        v2.Resize((224, 224)),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=mean, std=std)
    ]))

training_images_num = int(0.8 * len(training_dataset_full))
validation_images_num = int(0.2 * len(training_dataset_full))

training_dataset, not_training = random_split(training_dataset_full, [training_images_num, validation_images_num])
not_validation, validation_dataset = random_split(validation_dataset_full, [training_images_num, validation_images_num])

# Set batch size to 32. If we wanna speed up training, increase to 64 per the documentation recommendations.
training_dataloader = DataLoader(training_dataset, batch_size=32, shuffle=True)
validation_dataloader = DataLoader(validation_dataset, batch_size=32, shuffle=False)

print("Size of training dataset: " + str(len(training_dataset)))
print("Size of validation dataset: " + str(len(validation_dataset)))

class PetClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        # The layers pass from layer to the next to the next so the out of one becomes the in of the next. 
        # There are four layers and I'm doubling the number of filters so that it can identify more complex features as we move deeper into the architecture. 
        # I chose a kernel size of three to represent the 3x3 filter that will be applied to the images. 
        # From the guest lecture on the importance of standardisation and normalisation, I have applied batch normalisation after each convolutional layer.
        # This is to ensure that the values don't get too high or too litttle
        # After applying it through each filter, I'm applying max pooling to reduce spatial dimensions, which flattens the output.
        # It's a 2x2 sliding window which essentially is taking the max val in the window, and sliding it across the image. 
        

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1) 
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(256 * 14 * 14, 512) #14 represents the size of the grid after the pooling layers have hit. 256 is the number of filters in the final convolutional layer. 
        # 512 hidden neurons to learn and identify complex patt
        self.fc2 = nn.Linear(512, 37) # Second param matches no of sub-classes. 
        self.dropout = nn.Dropout(p=0.5) # It randomly sets 50% of the input units to 0 at each update during training time, which helps prevent overfitting. I set it up in anticipation of overfitting as a precautionary measure. 

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x)))) # 32 images get fed in as a batch (which is defined per our batch sizeWe initially start with three filters (RGB) and we apply conv, batch norm, relu activation, and max pooling to increase the number of filters 
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        x = torch.flatten(x, 1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

pet_classifier = PetClassifier().to(device)

# Add loss function
nn_loss = nn.CrossEntropyLoss()

# Add optimiser
optimizer = torch.optim.Adam(pet_classifier.parameters(), lr=0.001)

# Checking if we are overfitting or not
# plt.plot(training_losses, label = "Training Loss")
# plt.plot(validation_losses, label = "Validation Loss")
# plt.legend()
# plt.show()