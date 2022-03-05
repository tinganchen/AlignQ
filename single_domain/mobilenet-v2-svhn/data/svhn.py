from torchvision.datasets import SVHN
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

class Data:
    def __init__(self, args):
        # pin_memory = False
        # if args.gpu is not None:
        pin_memory = False
        
        def target_transform(target):
            return int(target) # - 1

        transform_train = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ])

        transform_test = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ])

        trainset = SVHN(root=args.data_dir, split = 'train', download=True, transform=transform_train, target_transform=target_transform)
        
        self.loader_train = DataLoader(
            trainset, batch_size=args.train_batch_size, shuffle=True, 
            num_workers=2, pin_memory=pin_memory
            )

        testset = SVHN(root=args.data_dir, split='test', download=True, transform=transform_test, target_transform=target_transform)
        self.loader_test = DataLoader(
            testset, batch_size=args.eval_batch_size, shuffle=False, 
            num_workers=2, pin_memory=pin_memory)
