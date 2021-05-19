import sys
import pathlib
from PIL import Image
import numpy as np
import torch
from torch import nn

import torchvision.transforms as transforms
import torchvision.models as models

# from model import Net


def main():
    # Load the classes
    data_dir = pathlib.Path('./data/tiny-imagenet-200/train/')
    CLASSES = [item.name for item in data_dir.glob('*')] 

    # im_height, im_width = 64, 64
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    resnet50 = models.resnet50(pretrained=True)
    for param in resnet50.parameters():
        param.requires_grad = False
    resnet50.fc = nn.Linear(resnet50.fc.in_features, 200)
    model = resnet50

    ckpt = torch.load('latest.pt')
    # model = Net(len(CLASSES), im_height, im_width)
    model.load_state_dict(ckpt['net'])
    model.to(device)

    model.eval()

    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)), 
        transforms.ToTensor(),
        # transforms.Normalize((0, 0, 0), tuple(np.sqrt((255, 255, 255)))),
    ])

    # Loop through the CSV file and make a prediction for each line
    with open('eval_classified.csv', 'w') as eval_output_file:  # Open the evaluation CSV file for writing
        for line in pathlib.Path(sys.argv[1]).open():  # Open the input CSV file for reading
            image_id, image_path, image_height, image_width, image_channels = line.strip().split(
                ',')  # Extract CSV info

            print(image_id, image_path, image_height, image_width, image_channels)
            with open(image_path, 'rb') as f:
                img = Image.open(f).convert('RGB')
            img = data_transforms(img)[None, :]
            img = img.to(device)
            
            outputs = model(img)
            _, predicted = outputs.max(1)

            # Write the prediction to the output file
            eval_output_file.write('{},{}\n'.format(image_id, CLASSES[predicted]))


if __name__ == '__main__':
    main()
