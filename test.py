from torchvision import datasets

test_dataset = datasets.OxfordIIITPet(root = "./data", split = "test", target_types = "category",download = True)