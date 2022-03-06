from torchvision import datasets
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

import os

class Data:
    def __init__(self, args):
        # pin_memory = False
        # if args.gpu is not None:
        pin_memory = False

        transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])

        trainset = datasets.ImageFolder(
            os.path.join('syndigit', 'imgs_train'), transform=transform)


        self.loader_train = DataLoader(
            trainset, batch_size=args.train_batch_size, shuffle=True, 
            num_workers=2, pin_memory=pin_memory
            )

        testset = datasets.ImageFolder(
            os.path.join('syndigit', 'imgs_valid'), transform=transform)

        self.loader_test = DataLoader(
            testset, batch_size=args.eval_batch_size, shuffle=False, 
            num_workers=2, pin_memory=pin_memory)
        
