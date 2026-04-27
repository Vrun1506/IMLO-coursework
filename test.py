from torchvision import datasets, transforms
from torch.utils.data import DataLoader

test_dataset = datasets.OxfordIIITPet(root = "./data", 
        split = "test", 
        target_types = "category",
        download = True, 
        transform = transforms.Compose([
            transforms.Resize((224, 224)), 
            transforms.ToTensor()]))

test_dataloader = DataLoader(test_dataset, batch_size = 32, shuffle = False)

# Not particularly interesting in the order so I've set shuffle to false. 
# If model accuracy drops, maybe change this to true and observe behaviour, but I don't think this will affect the model accuracy. 
# It only affects the order in which the images are fed to the model. 


# Add loss function

