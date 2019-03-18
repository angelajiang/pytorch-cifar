import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import lib.cifar
import lib.mnist

class CIFAR10:
    def __init__(self, model, test_batch_size, augment, randomize_labels):

        self.classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        self.model = model
        self.augment = augment

        # Testing set
        transform_test = transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        testset = lib.cifar.CIFAR10(root='./data',
                                    train=False,
                                    download=False,
                                    transform=transform_test,
                                    randomize_labels=randomize_labels)
        self.testloader = torch.utils.data.DataLoader(testset,
                                                      batch_size=test_batch_size,
                                                      shuffle=False,
                                                      num_workers=0)

        # Training set
        if self.augment:
            print("Performing data augmentation on CIFAR10")
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
        else:
            print("Not performing data augmentation on CIFAR10")
            transform_train = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
        self.trainset = lib.cifar.CIFAR10(root='./data',
                                          train=True,
                                          download=False,
                                          transform=transform_train,
                                          randomize_labels=randomize_labels)

        self.num_training_images = len(self.trainset)

        self.unnormalizer = transforms.Compose([transforms.Normalize(mean = [ 0., 0., 0. ],
                                                            std = [ 1/0.2023, 1/0.1994, 1/0.2010 ]),
                                       transforms.Normalize(mean = [ -0.4914, -0.4822, -0.4465 ],
                                                            std = [ 1., 1., 1. ])
                                      ])

class MNIST:
    def __init__(self, device, test_batch_size):

        self.model = MNISTNet().to(device)

        self.classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')

        # Testing set
        self.testloader = torch.utils.data.DataLoader(
            lib.mnist.MNIST('../data', train=False, download=True,
                           transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
            batch_size=test_batch_size, shuffle=False, num_workers=0)

        # Training set
        self.trainset = lib.mnist.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ]))
        self.unnormalizer = transforms.Compose([transforms.Normalize(mean = [ 0., 0., 0. ],
                                                            std = [ 1/0.3081]),
                                       transforms.Normalize(mean = [ -0.1307],
                                                            std = [ 1., 1., 1. ])
                                      ])

        self.num_training_images = len(self.trainset)

class MNISTNet(nn.Module):
    def __init__(self):
        super(MNISTNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

from torch.utils.data import ConcatDataset
from torchvision import datasets
class IndexedSVHN(datasets.SVHN):
    def __getitem__(self, index):
        retval = super(IndexedSVHN, self).__getitem__(index)
        return retval + (index,)

class SVHN:
    def __init__(self, model, test_batch_size, augment):
        self.classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')
        self.model = model
        self.augment = augment

        # Testing set
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        testset = IndexedSVHN(root='./svhn_data',
                              split='test',
                              download=False,
                              transform=transform_test)
        self.testloader = torch.utils.data.DataLoader(testset,
                                                      batch_size=test_batch_size,
                                                      shuffle=False,
                                                      num_workers=0)

        # Training set
        if self.augment:
            print("Performing data augmentation on SVHN")
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
        else:
            print("Not performing data augmentation on SVHN")
            transform_train = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
        self.trainset1 = IndexedSVHN(root='./svhn_data',
                                     split='train',
                                     download=False,
                                     transform=transform_train)
        self.trainset2 = IndexedSVHN(root='./svhn_data',
                                     split='extra',
                                     download=False,
                                     transform=transform_train)
        self.trainset = ConcatDataset([self.trainset1, self.trainset2])

        self.num_training_images = len(self.trainset)

        self.unnormalizer = transforms.Compose([transforms.Normalize(mean = [ 0., 0., 0. ],
                                                            std = [ 1/0.2023, 1/0.1994, 1/0.2010 ]),
                                       transforms.Normalize(mean = [ -0.4914, -0.4822, -0.4465 ],
                                                            std = [ 1., 1., 1. ])
                                      ])

class IndexedImageFolder(datasets.ImageFolder):
    def __getitem__(self, index):
        retval = super(IndexedImageFolder, self).__getitem__(index)
        return retval + (index,)

class ImageNet:
    def __init__(self, model, test_batch_size, traindir, valdir):
        self.classes = [str(i) for i in range(1000)]
        self.model = model
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])


        # Testing set
        testset = IndexedImageFolder(valdir, transforms.Compose([
                                            transforms.Resize(256),
                                            transforms.CenterCrop(224),
                                            transforms.ToTensor(),
                                            normalize]))
        self.testloader = torch.utils.data.DataLoader(testset,
                                                      batch_size=test_batch_size,
                                                      shuffle=False,
                                                      num_workers=0)

        # Training set
        print("Performing data augmentation on ImageNet")
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
        self.trainset = IndexedImageFolder(traindir,
                                           transform_train)
        self.num_training_images = len(self.trainset)
        print(self.num_training_images)
        self.unnormalizer = transforms.Compose([transforms.Normalize(mean = [ 0., 0., 0. ],
                                                        std = [ 1/0.229, 1/0.224, 1/0.225 ]),
                                                transforms.Normalize(mean = [ -0.485, -0.456, -0.406 ],
                                                        std = [ 1., 1., 1. ])
                                               ])
