import os
import sys

from facenet_pytorch import MTCNN, InceptionResnetV1, fixed_image_standardization, training
from torchvision import datasets, transforms
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn.functional as F

if __name__ == "__main__":
    # Specify the directory where the dataset is stored
    data_dir = 'dataset'

    image = Image.open('test_image.jpg')

    # Define the transformations to be applied to the images
    trans = transforms.Compose([
        np.float32,
        transforms.ToTensor(),
        fixed_image_standardization
    ])

    # Load the dataset from the directory
    dataset = datasets.ImageFolder(data_dir + "_cropped", transform=trans)

    # Load model from file model.pt
    # Load the model
    model = InceptionResnetV1(
        classify=True,
        pretrained="vggface2",
        num_classes=len(dataset.class_to_idx)
    )

    # Set the model to evaluation mode
    model.eval()

    # Find face with MTCNN
    mtcnn = MTCNN()
    image = mtcnn(image)

    # Apply the transformations to the image
    # image = trans(image)

    # Add an extra dimension to represent the batch size
    image = image.unsqueeze(0)

    # Show image
    plt.imshow(image[0].permute(1, 2, 0))
    plt.show()

    # Pass the image through the model
    output = model(image)

    # Get the predicted class
    probabilities = F.softmax(output, dim=1)

    # Convert tensor to numpy array
    probabilities = probabilities.detach().numpy()

    # Print probabilities for each class
    for i, class_name in enumerate(dataset.classes):
        print(f'The image belongs to class {class_name} with probability {probabilities[0][i] * 100:.2f}%')

    pass
