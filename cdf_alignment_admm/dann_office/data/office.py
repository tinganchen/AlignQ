from torchvision import datasets
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

import os

class Data:
    def __init__(self, args, dataset_root, category):
        # pin_memory = False
        # if args.gpu is not None:
        pin_memory = False

        transform_train = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])
        
        transform_test = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])

        train_dataset = datasets.ImageFolder(
            os.path.join(dataset_root, 'office31_split', category, 'images', 'train'), transform=transform_train)

        self.loader_train = DataLoader(
            dataset=train_dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=0)
    	
        test_dataset = datasets.ImageFolder(
            os.path.join(dataset_root, 'office31_split', category, 'images', 'test'), transform=transform_test)

        self.loader_test = DataLoader(
            dataset=test_dataset, batch_size=args.eval_batch_size, shuffle=True, num_workers=0)
    
