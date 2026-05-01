## Best performing version: 512 neurons and 0.3 dropout
# Going to try and implement a ResNet architecture on top of this simple architecture to try and improve the accuracy. 
# We skip the "vanishing gradient" problem by adding skip connections. 
# Gonna look into how the DigitalOcean ResNet implementation works and then adapt it to this architecture and see if that makes a difference. 


from torchvision import datasets
from torchvision.transforms import v2
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch

training_losses = []
validation_losses = []
training_accuracies = []
validation_accuracies = []


# raw_dataset = datasets.OxfordIIITPet
#     root="./data",
#     split="trainval",
#     target_types="category",
#     download=True,
#     transform=v2.Compose([
#         v2.Resize((224, 224)),
#         v2.ToImage(),
#         v2.ToDtype(torch.float32, scale=True)
#     ])
# )

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


# I computed these values using the above code snippet, which I got from a PyTorch forum.
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

training_dataset, _ = random_split(training_dataset_full, [training_images_num, validation_images_num])
_, validation_dataset = random_split(validation_dataset_full, [training_images_num, validation_images_num])

training_dataloader = DataLoader(training_dataset, batch_size=64, shuffle=True)
validation_dataloader = DataLoader(validation_dataset, batch_size=64, shuffle=False)

print("Size of training dataset: "+str(len(training_dataset)))
print("Size of validation dataset: "+str(len(validation_dataset)))

class PetClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        # The layers pass from one to the next so the out_channels of one becomes the in_channels of the next.
        # There are four layers and I'm doubling the number of filters so that it can identify more complex features as we move deeper into the architecture.
        # I chose a kernel size of 3 to represent the 3x3 filter applied to the images.
        # From the guest lecture on standardisation and normalisation, I have applied batch normalisation
        # after each convolutional layer to ensure values don't get too high or too low.
        # After each conv block I apply max pooling to reduce spatial dimensions.
        # It's a 2x2 sliding window taking the max value in each window as it slides across the image.

        self.conv1 = nn.Conv2d(in_channels=3,   out_channels=64,  kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(in_channels=64,  out_channels=128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(512 * 14 * 14, 512) # 14 represents the spatial size after 4 rounds of 2x2 max pooling on a 224x224 image
        self.fc2 = nn.Linear(512, 37) # 37 pet breeds
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
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

nn_loss = nn.CrossEntropyLoss()

# Tried to run it at 0.0002 but the validation accuracy and training accuracy was exploding all over the place lowkey. 
optimiser = torch.optim.Adam(pet_classifier.parameters(), lr=0.0001)

# Cosine annealing smoothly decays the learning rate
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=30, eta_min=1e-5)

epoch_limit = 30

for epoch in range(epoch_limit):
    pet_classifier.train()
    running_train_loss = 0.0
    correct_train = 0
    total_train = 0
    for images, labels in training_dataloader:
        images = images.to(device)
        labels = labels.to(device)
        optimiser.zero_grad()

        outputs = pet_classifier(images)
        loss = nn_loss(outputs, labels)
        loss.backward()
        optimiser.step()

        running_train_loss += loss.item()
        _, predicted = torch.max(outputs, dim=1)
        total_train+= labels.size(0)
        correct_train+= (predicted == labels).sum().item()

    epoch_train_loss = running_train_loss / len(training_dataloader)
    epoch_train_accuracy = 100.0 * correct_train / total_train
    training_losses.append(epoch_train_loss)
    training_accuracies.append(epoch_train_accuracy)

    pet_classifier.eval()
    running_val_loss = 0.0
    correct_val = 0
    total_val = 0

    with torch.no_grad():
        for images, labels in validation_dataloader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = pet_classifier(images)
            loss = nn_loss(outputs, labels)
            running_val_loss += loss.item()
            _, predicted = torch.max(outputs, dim=1)
            total_val += labels.size(0)
            correct_val += (predicted == labels).sum().item()

    epoch_val_loss = running_val_loss / len(validation_dataloader)
    epoch_val_accuracy = 100.0 * correct_val / total_val
    validation_losses.append(epoch_val_loss)
    validation_accuracies.append(epoch_val_accuracy)

    # Update larning rate so that it decays more smoothly as we approach the end of training. 
    scheduler.step()

    print("\nEpoch " +str(epoch + 1) + "/"+str(epoch_limit)+" Summary:")
    print("Training Loss: "+str(epoch_train_loss)+"%")
    print("Training Accuracy: "+ str(epoch_train_accuracy)+"%")
    print("Validation Loss: "+str(epoch_val_loss)+ "%")
    print("Validation Accuracy: "+str(epoch_val_accuracy)+ "%")
    print("Learning Rate: "+ str(scheduler.get_last_lr()[0]))

# Checking if we are overfitting or not
# plt.plot(training_losses, label = "Training Loss")
# plt.plot(validation_losses, label = "Validation Loss")
# plt.legend()
# plt.show()