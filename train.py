import torch
from torch import nn, optim
from torchvision import models
from collections import OrderedDict
import argparse
from workspace_utils import active_session  # Optional utility to keep session active

# Import your custom functions like data loading, model building, etc.
from utils import load_data, save_checkpoint

def get_input_args():
    parser = argparse.ArgumentParser(description='Train a new network on a dataset.')
    
    # Command line arguments
    parser.add_argument('data_dir', type=str, help='Directory of the dataset')
    parser.add_argument('--save_dir', type=str, default='.', help='Directory to save checkpoint')
    parser.add_argument('--arch', type=str, default='vgg16', help='Model architecture (vgg16 or densenet121)')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for training')
    parser.add_argument('--hidden_units', type=int, default=512, help='Number of hidden units in classifier')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs to train for')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for training if available')
    
    return parser.parse_args()

def train():
    args = get_input_args()
    
    # Load data
    dataloaders, image_datasets = load_data(args.data_dir)

    # Load pre-trained model and replace classifier
    if args.arch == 'vgg16':
        model = models.vgg16(pretrained=True)
    elif args.arch == 'densenet121':
        model = models.densenet121(pretrained=True)
    else:
        print(f"Architecture {args.arch} is not supported.")
        return

    for param in model.parameters():
        param.requires_grad = False

    classifier = nn.Sequential(OrderedDict([
                            ('fc1', nn.Linear(25088, args.hidden_units)),
                            ('relu', nn.ReLU()),
                            ('dropout', nn.Dropout(0.2)),
                            ('fc2', nn.Linear(args.hidden_units, 102)),
                            ('output', nn.LogSoftmax(dim=1))
                            ]))

    model.classifier = classifier
    
    # Set device (GPU/CPU)
    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Define criterion and optimizer
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)

    # Training loop
#     with active_session():  # Optional to keep session active in Udacity Workspace
    for epoch in range(args.epochs):
        running_loss = 0
        for inputs, labels in dataloaders['train']:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            logps = model(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch+1}/{args.epochs}.. Training loss: {running_loss/len(dataloaders['train'])}")

    print("Training complete.")
    
    # Save the model
    model.class_to_idx = image_datasets['train'].class_to_idx
    save_checkpoint(model, optimizer, args)

if __name__ == "__main__":
    train()
