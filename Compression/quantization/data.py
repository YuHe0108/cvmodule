from torchvision import datasets, transforms
from torch.utils import data

train_transforms = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
test_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset = datasets.CIFAR10(root='cifar10',
                            train=True,
                            transform=train_transforms,
                            download=True)
trainloader = data.DataLoader(trainset,
                              batch_size=64,
                              shuffle=True,
                              drop_last=True,
                              num_workers=0)

testset = datasets.CIFAR10(root='cifar10',
                           train=False,
                           transform=test_transforms,
                           download=True)
testloader = data.DataLoader(testset,
                             batch_size=64,
                             shuffle=False,
                             num_workers=0)
