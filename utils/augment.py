from torchvision import transforms

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(299),
    transforms.RandomRotation(degrees=40),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
])

val_test_transform = transforms.Compose([
    transforms.Resize(326),
    transforms.CenterCrop(299),
    transforms.ToTensor(),
])