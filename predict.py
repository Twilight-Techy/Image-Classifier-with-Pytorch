import torch
from torchvision import models
from PIL import Image
import numpy as np
import argparse
from utils import load_checkpoint, process_image

def get_input_args():
    parser = argparse.ArgumentParser(description='Predict flower name from an image using a trained model.')
    
    # Command line arguments
    parser.add_argument('input', type=str, help='Path to the input image')
    parser.add_argument('checkpoint', type=str, help='Path to the model checkpoint')
    parser.add_argument('--top_k', type=int, default=5, help='Return top K most likely classes')
    parser.add_argument('--category_names', type=str, help='Path to category to names JSON file')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for inference if available')

    return parser.parse_args()

def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model. '''
    model.eval()
    img = process_image(image_path)
    img = torch.from_numpy(img).unsqueeze_(0).float()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    img = img.to(device)

    with torch.no_grad():
        output = model.forward(img)
    
    ps = torch.exp(output)
    top_probs, top_labels = ps.topk(topk)

    return top_probs, top_labels

def main():
    args = get_input_args()
    
    # Load the checkpoint and rebuild model
    model = load_checkpoint(args.checkpoint)

    # Perform inference
    probs, classes = predict(args.input, model, args.top_k)

    print(f"Probabilities: {probs}")
    print(f"Classes: {classes}")

if __name__ == "__main__":
    main()
