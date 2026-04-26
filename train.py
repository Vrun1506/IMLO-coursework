from torchvision import datasets

training_dataset = datasets.OxfordIIITPet(root = "./data", split = "trainval", target_types = "category",download = True)

print(len(training_dataset))

first_image = training_dataset[0][0]
print(first_image)

print(first_image.size)

image_label = training_dataset[0][1]
print(image_label)