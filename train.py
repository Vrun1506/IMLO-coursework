## Best performing version: 512 neurons and 0.2 dropout
# Going to try and implement a ResNet architecture on top of this simple architecture to try and improve the accuracy. 
# We skip the "vanishing gradient" problem by adding skip connections. 
# Gonna look into how the DigitalOcean ResNet implementation works and then adapt it to this architecture and see if that makes a difference. 


from torchvision import datasets
from torchvision.transforms import v2
from torch.utils.data import DataLoader, random_split
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

training_dataset, _ = random_split(training_dataset_full, [training_images_num, validation_images_num]) # Opted for a 80:20 split, which I used in the past for a Computer Vision project at a hackathon, and it worked quite well. 
_, validation_dataset = random_split(validation_dataset_full, [training_images_num, validation_images_num])

training_dataloader = DataLoader(training_dataset, batch_size=64, shuffle=True) # I upped the batch size from 32 and it improved the model accuracy significantly, most likely because each gradient update is averaged over more samples, so less noise
validation_dataloader = DataLoader(validation_dataset, batch_size=64, shuffle=False)

print("Size of training dataset: "+str(len(training_dataset))) # Sanity check 
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

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64,  kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64) # Nromalise the output of conv layer each time to ensure value size stys within a reasoable range. 
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) # 2x2 sliding window to essentially half the spatial dimensions and this happens four times total (one after each convolutional layer execution), which means we are basically dividing by 16, giving us 14 x 14. 

        self.fc1 = nn.Linear(512 * 14 * 14, 512) # 14 represents the spatial size after 4 rounds of 2x2 max pooling on a 224x224 image
        self.fc2 = nn.Linear(512, 37) # 37 pet breeds
        self.dropout = nn.Dropout(p=0.1) # Modifying this helps to prevent overfitting by setting certain inputs to 0 during training, and thus enforces robustness. Started at 0.5, moved to 0.3, 0.2 and then 0.1.

        # The value still didn't overfit the model, and we observed a significant acc improv. 

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x)))) # Applies all of teh stuff mentioned above. 
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        x = torch.flatten(x, 1) # Flattens the tensor for the fully connected layer into a 1D vector. 
        x = self.dropout(F.relu(self.fc1(x))) 
        x = self.fc2(x)
        return x # We get the scores for the 37 breeds as output, which we can then pass to the cross entropy loss function. 

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu" # MPS was genuinely really slow, so went with the T4 GPU, which trained the model much faster. 
print(f"Using {device} device")

pet_classifier = PetClassifier().to(device) # Move the nn to GPU. 

nn_loss = nn.CrossEntropyLoss() # Loss calc

# Tried to run it at 0.0002 but the validation accuracy and training accuracy was exploding all over the place lowkey. 
# optimiser = torch.optim.Adam(pet_classifier.parameters(), lr=0.0001)

optimiser = torch.optim.AdamW(pet_classifier.parameters(), lr=0.0001, weight_decay=1e-4)
# Better Adam according to research because better generalisation so operates better on unseen data. 
# Kinda like with the objective function J in our lectures, the weight decay penalises the model for having large weights. 

# Until I've established what the most optimal hyperparams are, I'm sticking with AdamW. 

# Once architecture sorted, SWITCH TO SGD because it performs better once hyperparams are optimised because we aren't having to update the learning rate as much. 

# Cosine annealing smoothly decays the learning rate
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=30, eta_min=1e-5)

for epoch in range(30):
    pet_classifier.train()
    running_train_loss = 0.0
    correct_train = 0
    total_train = 0
    for images, labels in training_dataloader:
        images = images.to(device) # imgs + labels to GPU
        labels = labels.to(device)
        optimiser.zero_grad()

        outputs = pet_classifier(images) # Predict the scores for each breed. 
        loss = nn_loss(outputs, labels) # Compute loss for predictions and labels
        
        loss.backward() 
        optimiser.step() # Weight update

        running_train_loss += loss.item() # Total loss for epoch stats
        _, predicted = torch.max(outputs, dim=1) # get the guess we are making based off the data
        total_train+= labels.size(0)
        correct_train+= (predicted == labels).sum().item() # guss == lavel comparison to see behaviour in current epoch. 

    # Stats calc
    epoch_train_loss = running_train_loss / len(training_dataloader)
    epoch_train_accuracy = 100.0 * correct_train / total_train
    training_losses.append(epoch_train_loss)
    training_accuracies.append(epoch_train_accuracy)

    pet_classifier.eval()
    running_val_loss = 0.0
    correct_val = 0
    total_val = 0

    with torch.no_grad(): # Stops the gradient update since we are evaluating the performance of our validation dataset. 
        
        # Pretty much the same process happens but this time with the training dataset. 
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

    scheduler.step() # We update the learning rate as we go with each epoch, which will improve model performance as we go. 

    print("\nEpoch "+str(epoch + 1) + "/ 30 \n Summary:")
    print("Training Loss: "+str(epoch_train_loss)+"%")
    print("Training Accuracy: "+str(epoch_train_accuracy)+"%")
    print("Validation Loss: "+str(epoch_val_loss)+ "%")
    print("Validation Accuracy: "+str(epoch_val_accuracy)+ "%")
    print("Learning Rate: "+str(scheduler.get_last_lr()[0]))