from torchvision.datasets import MNIST
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

class Data:
    def __init__(self, args):
        # pin_memory = False
        # if args.gpu is not None:
        pin_memory = False

        transform = transforms.Compose([
            transforms.Resize((args.img_size, args.img_size)), # different img size settings for mnistm(28) and svhn(32).
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5), std=(0.5))
        ])


        trainset = MNIST(root='mnist', train=True, download=True, transform=transform)
        
        self.loader_train = DataLoader(
            trainset, batch_size=args.train_batch_size, shuffle=True, 
            num_workers=2, pin_memory=pin_memory
            )

        testset = MNIST(root='mnist', train=False, download=True, transform=transform)
        
        self.loader_test = DataLoader(
            testset, batch_size=args.eval_batch_size, shuffle=False, 
            num_workers=2, pin_memory=pin_memory)
        
       
