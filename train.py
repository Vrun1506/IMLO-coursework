from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Images are coming in different sizes so need to figure out how to resize them to a common size so the model can process them more easily. 
# Need to be mindful of data loss when doing this though. 

# Did some research and found that 224x224 is the conventional size for image classification tasks. If it takes too long to train, reduce the image size so it speeds up. 

# From the week 10 lecture, the model expects data to be in the form of signals. 
# toTensor() converts the pixel into a multi-deminsional array of numbers between 0 and 1 (kinda like a one-hot encoding). 
# The model now has the numerical data it needs to learn from and now we can start training the model (no idea how to do this yet).

training_dataset = datasets.OxfordIIITPet(root = "./data", 
        split = "trainval", 
        target_types = "category",
        download = True, 
        transform = transforms.Compose([
            transforms.Resize((224, 224)), 
            transforms.ToTensor()]))

print(len(training_dataset))

first_image = training_dataset[0][0]
print(first_image)

print(first_image.size)

image_label = training_dataset[0][1]
print(image_label)

training_dataloader = DataLoader(training_dataset, batch_size = 32, shuffle = True)

# Batch size is the number of images that will be passed to the model at a time. Docs suggest using 64, but I've gone slightly lower to improve accuracy.
# Use shuffle to avoid data bias by random shuffles at the start of each epoch to ensure that the model doesn't learn anything about the order of the images. 


# Need to find a way to increase the number of images available to us for training because we don't have enough currently. 
# Check email to wait for response from lecturer about manipulation with transforms. 


# Add loss function


# Add optimiser

# Coursework paper says to split the dataset into training and validation sets as an option. 
# Number of images is greatly diluted already per the different classes, so I think I'll just keep the test set for validating. 
# Could be interesting to see if it makes a difference if I do the 80:20 split for training and validation (with hyperparam modification to optimise) and then the test set for actually testing the model. 