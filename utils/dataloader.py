import torch
import torchvision
from torchvision import transforms, datasets
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import ConcatDataset



def Dataloader(IMG_SIZE, BATCH_SIZE, debug=False):
    """
    Loads data, transforms data and prepares data for training using DataLoader
    
    :params IMG_SIZE: Assuming height and width of the image are same, IMG_SIZE is the height/width of the image
    :params BATCH_SIZE: Batch size during training
    """
    
    transform = transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)), # Resizes image
                                    transforms.ToTensor(), # Scales data to [0,1]
                                    transforms.Lambda(lambda x: (x * 2) - 1)]) # Rescale data to [-1, 1] range
    train = datasets.MNIST('../data', train=True, transform=transform, download=True) # 60000 images
    test = datasets.MNIST('../data', train=False, transform=transform, download=True) # 10000 images
    
    if debug:
        print('------------------------------Debugging------------------------------')
        # Subset of dataset
        train = torch.utils.data.Subset(train, indices=range(len(train)//100))
        test =torch.utils.data. Subset(test, indices=range(len(test)//100))

    data = ConcatDataset([train, test]) # Combine train and test sets to create a larger dataset

    data = DataLoader(data, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    
    return data