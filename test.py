from torchvision import datasets
from torchvision.transforms import v2
from torch.utils.data import DataLoader
import torch

test_dataset = datasets.OxfordIIITPet(root = "./data", 
        split = "test", 
        target_types = "category",
        download = True, 
        transform = v2.Compose([
            v2.Resize((224, 224)), 
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True)]))

test_dataloader = DataLoader(test_dataset, batch_size = 32, shuffle = False)

# Not particularly interesting in the order so I've set shuffle to false. 
# If model accuracy drops, maybe change this to true and observe behaviour, but I don't think this will affect the model accuracy. 
# It only affects the order in which the images are fed to the model. 


# Add loss function

