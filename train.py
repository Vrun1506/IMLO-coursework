from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt



# Images are coming in different sizes so need to figure out how to resize them to a common size so the model can process them more easily. 
# Need to be mindful of data loss when doing this though. 

# Did some research and found that 224x224 is the conventional size for image classification tasks. If it takes too long to train, reduce the image size so it speeds up. 

# From the week 10 lecture, the model expects data to be in the form of signals. 
# toTensor() converts the pixel into a multi-deminsional array of numbers between 0 and 1 (kinda like a one-hot encoding). 
# The model now has the numerical data it needs to learn from and now we can start training the model (no idea how to do this yet).

training_losses = []
validation_losses = []


training_dataset_full = datasets.OxfordIIITPet(root = "./data", 
        split = "trainval", 
        target_types = "category",
        download = True, 
        transform = transforms.Compose([
            transforms.Resize((224, 224)), 
            transforms.ToTensor()]))

validation_dataset_full = datasets.OxfordIIITPet(root = "./data", 
        split = "trainval", 
        target_types = "category",
        download = True, 
        transform = transforms.Compose([
            transforms.Resize((224, 224)), 
            transforms.ToTensor()]))


training_dataset, not_training = random_split(training_dataset_full, [int(0.8*len(training_dataset_full)), int(0.2*len(training_dataset_full))])
not_validation, validation_dataset = random_split(validation_dataset_full, [int(0.8*len(validation_dataset_full)), int(0.2*len(validation_dataset_full))])

# print(len(training_dataset))

# first_image = training_dataset[0][0]
# print(first_image)

# print(first_image.size)

# image_label = training_dataset[0][1]
# print(image_label)

# Set batch size to 32. If we wanna speed up training, increase to 64 per the documentation recommendations. 
training_dataloader = DataLoader(training_dataset, batch_size = 32, shuffle = True)
validation_dataloader = DataLoader(validation_dataset, batch_size=32, shuffle = False) # Not particularly interested in the order so shuffle is False.

print("Size of training dataset: "+str(len(training_dataset)))
print("Size of validation dataset: "+str(len(validation_dataset)))
# Need to find a way to increase the number of images available to us for training because we don't have enough currently. 



# Add loss function


# Add optimiser

# Coursework paper says to split the dataset into training and validation sets as an option. 


# Checking if we are overfitting or not
# plt.plot(training_losses, label = "Training Loss")
# plt.plot(validation_losses, label = "Validation Loss")
# plt.legend()
# plt.show()