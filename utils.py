import torch
from torchvision import datasets, transforms

def load_data(data_dir):
    ''' Loads data and returns data loaders and dataset. '''
    
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    # Define transforms for the training, validation, and testing sets
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomRotation(30),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'valid': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    # Load datasets
    image_datasets = {
        'train': datasets.ImageFolder(train_dir, transform=data_transforms['train']),
        'valid': datasets.ImageFolder(valid_dir, transform=data_transforms['valid']),
        'test': datasets.ImageFolder(test_dir, transform=data_transforms['test']),
    }

    # Load data loaders
    dataloaders = {
        'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size=64, shuffle=True),
        'valid': torch.utils.data.DataLoader(image_datasets['valid'], batch_size=64),
        'test': torch.utils.data.DataLoader(image_datasets['test'], batch_size=64),
    }

    return dataloaders, image_datasets

def save_checkpoint(model, optimizer, args):
    ''' Save the trained model checkpoint. '''
    checkpoint = {
        'arch': args.arch,
        'hidden_units': args.hidden_units,
        'class_to_idx': model.class_to_idx,
        'state_dict': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'epochs': args.epochs,
    }
    torch.save(checkpoint, args.save_dir + '/checkpoint.pth')

def load_checkpoint(filepath):
    ''' Load a checkpoint and rebuild the model. '''
    checkpoint = torch.load(filepath)
    model = models.__dict__[checkpoint['arch']](pretrained=True)
    model.classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(25088, checkpoint['hidden_units'])),
        ('relu', nn.ReLU()),
        ('dropout', nn.Dropout(0.2)),
        ('fc2', nn.Linear(checkpoint['hidden_units'], 102)),
        ('output', nn.LogSoftmax(dim=1))
    ]))
    model.load_state_dict(checkpoint['state_dict'])
    
    return model
