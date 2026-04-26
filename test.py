from torchvision import datasets
from torch.utils.data import DataLoader

test_dataset = datasets.OxfordIIITPet(root = "./data", split = "test", target_types = "category",download = True)

test_dataloader = DataLoader(test_dataset, batch_size = 32, shuffle = False)

# Not particularly interesting in the order so I've set shuffle to false. If model accuracy drops, maybe change this to true and observe behaviour. 


# Add loss function

