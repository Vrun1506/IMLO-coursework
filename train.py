from torchvision import datasets
from torchvision.transforms import v2
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import torch.nn as nn
import torch

training_losses = []
validation_losses = []


training_dataset_full = datasets.OxfordIIITPet(root = "./data", 
        split = "trainval", 
        target_types = "category",
        download = True, 
        transform = v2.Compose([
            v2.Resize((224, 224)),
            v2.RandomHorizontalFlip(),
            v2.RandomRotation(10),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True)]))

validation_dataset_full = datasets.OxfordIIITPet(root = "./data", 
        split = "trainval", 
        target_types = "category",
        download = True, 
        transform = v2.Compose([
            v2.Resize((224, 224)), 
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True)]))


training_images_num = int(0.8 * len(training_dataset_full))
validation_images_num = int(0.2 * len(training_dataset_full))

training_dataset, not_training = random_split(training_dataset_full, [training_images_num, validation_images_num])
not_validation, validation_dataset = random_split(validation_dataset_full, [training_images_num, validation_images_num])

# Set batch size to 32. If we wanna speed up training, increase to 64 per the documentation recommendations. 
training_dataloader = DataLoader(training_dataset, batch_size = 32, shuffle = True)
validation_dataloader = DataLoader(validation_dataset, batch_size=32, shuffle = False) # Not particularly interested in the order so shuffle is False.

print("Size of training dataset: "+str(len(training_dataset)))
print("Size of validation dataset: "+str(len(validation_dataset)))

class PetClassifier(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
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