import os

from keras._tf_keras.keras.preprocessing.image import save_img

from PIL import Image


from facenet_pytorch import MTCNN, InceptionResnetV1, fixed_image_standardization, training
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
import numpy as np
import cv2
from augmentation import ImageDataAugmentation
import torch.nn.functional as F

import warnings
from urllib3.exceptions import InsecureRequestWarning

warnings.simplefilter('ignore', InsecureRequestWarning)


class ModelCreator:

    def __init__(self, model, model_name):
        self.model = model
        self.model_name = "model.pt"
        self.user_name = model_name
        self.model_path = f'models/{self.user_name}.pt'
        self.video_path = f'temporary/{self.user_name}.mp4'
        self.image_path = f"temporary/{self.user_name}.jpg"
        self.dataset_path = f'dataset/{self.user_name}'
        self.dataset_cropped_path = f'dataset_cropped/{self.user_name}'
        self.is_loaded = False
        self.save_counter = 0

    def save_image(self, image):
        """Save image to dataset/augmented folder"""
        filename = f"{self.user_name.replace(' ', '_')}{self.save_counter}.jpg"
        if not os.path.exists(f'dataset/{self.user_name}'):
            os.makedirs(f'dataset/{self.user_name}')
        if os.path.exists(f'dataset/{self.user_name}/{filename}'):
            os.remove(f'dataset/{self.user_name}/{filename}')

        save_img(f'dataset/{self.user_name}/{filename}', image, )

        self.save_counter += 1
        pass

    def get_dataset_from_video(self):
        """Get dataset from video."""
        video = cv2.VideoCapture(self.video_path)
        video_dataset = []
        saved_counter = 0

        while video.isOpened():
            ret, frame = video.read()
            if not ret:
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            # frame = cv2.resize(frame, (512, 512))
            video_dataset.append(frame)

            self.save_image(frame)

            saved_counter += 1

        video.release()

        video_dataset = np.array(video_dataset)
        return video_dataset

    def create_model(self):
        # load dataset
        save_dataset = self.get_dataset_from_video()

        augmentor = ImageDataAugmentation(save_dataset, self.user_name)
        augmentor.augment()

        data_dir = 'dataset'
        batch_size = 16
        epochs = 3
        workers = 0 if os.name == 'nt' else os.cpu_count()

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print('Running on device: {}'.format(device))

        # MTCNN module to detect faces
        mtcnn = MTCNN(
            image_size=160, margin=0, min_face_size=20,
            thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
            device=device
        )

        dataset = datasets.ImageFolder(data_dir, transform=transforms.Resize((512, 512)))
        dataset.samples = [
            (p, p.replace(data_dir, data_dir + '_cropped'))
            for p, _ in dataset.samples
        ]

        # Create loader for loading batches of dataset images
        loader = DataLoader(
            dataset,
            num_workers=workers,
            batch_size=batch_size,
            collate_fn=training.collate_pil
        )

        for i, (x, y) in enumerate(loader):
            try:
                # Detect faces
                mtcnn(x, save_path=y)
                print('\rBatch {} of {}'.format(i + 1, len(loader)), end='')
            except Exception as e:
                # Handle errors
                print(f'\nError processing batch {i + 1}: {e}')
                print(f'x: {x}\ny: {y}')

        # Remove mtcnn to reduce GPU memory usage
        del mtcnn

        # Model for face recognition, trained on vggface2 dataset
        resnet = InceptionResnetV1(
            classify=True,
            pretrained='vggface2',
            num_classes=len(dataset.class_to_idx)
        ).to(device)

        optimizer = optim.Adam(resnet.parameters(), lr=0.001)
        scheduler = MultiStepLR(optimizer, [5, 10])

        trans = transforms.Compose([
            np.float32,
            transforms.ToTensor(),
            fixed_image_standardization
        ])
        dataset = datasets.ImageFolder(data_dir + "_cropped", transform=trans)

        img_inds = np.arange(len(dataset))
        np.random.shuffle(img_inds)
        train_inds = img_inds[:int(0.8 * len(img_inds))]
        val_inds = img_inds[int(0.8 * len(img_inds)):]

        train_loader = DataLoader(
            dataset,
            num_workers=workers,
            batch_size=batch_size,
            sampler=SubsetRandomSampler(train_inds)
        )
        val_loader = DataLoader(
            dataset,
            num_workers=workers,
            batch_size=batch_size,
            sampler=SubsetRandomSampler(val_inds)
        )

        loss_fn = torch.nn.CrossEntropyLoss()
        metrics = {
            'fps': training.BatchTimer(),
            'acc': training.accuracy
        }

        writer = SummaryWriter()
        writer.iteration, writer.interval = 0, 10

        print('\n\nInitial')
        print('-' * 10)
        resnet.eval()
        training.pass_epoch(
            resnet, loss_fn, val_loader,
            batch_metrics=metrics, show_running=True, device=device,
            writer=writer
        )

        for epoch in range(epochs):
            print('\nEpoch {}/{}'.format(epoch + 1, epochs))
            print('-' * 10)

            resnet.train()
            training.pass_epoch(
                resnet, loss_fn, train_loader, optimizer, scheduler,
                batch_metrics=metrics, show_running=True, device=device,
                writer=writer
            )

            resnet.eval()
            training.pass_epoch(
                resnet, loss_fn, val_loader,
                batch_metrics=metrics, show_running=True, device=device,
                writer=writer
            )

        writer.close()

        # delete user images in dataset
        for file in os.listdir(f"{self.dataset_path}"):
            os.remove(f"{self.dataset_path}/{file}")

        for file in os.listdir(f"{self.dataset_cropped_path}"):
            os.remove(f"{self.dataset_cropped_path}/{file}")

        torch.save(resnet.state_dict(), self.model_path)

        print('Model created successfully!')

    def load_model(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        try:
            self.model = InceptionResnetV1(
                classify=True,
                pretrained="vggface2",
                num_classes=2,
                device=device
            )

            self.model.load_state_dict(torch.load(self.model_path))
            self.model.eval()
            self.is_loaded = True
            return

        except Exception as ignored:

            try:
                self.model = InceptionResnetV1(
                    classify=True,
                    pretrained="vggface2",
                    num_classes=3,
                    device=device
                )

                self.model.load_state_dict(torch.load(self.model_path))
                self.model.eval()
                self.is_loaded = True
                return

            except Exception as e:
                print(str(e))
                return



    def does_model_exist(self):
        return os.path.exists(self.model_path)

    def verify_image(self):
        if not self.is_loaded:
            self.load_model()

        image = Image.open(self.image_path)


        mtcnn = MTCNN()
        image = mtcnn(image)

        if image is None:
            return False

        image = image.unsqueeze(0)

        output = self.model(image)
        probabilities = F.softmax(output, dim=1)
        probabilities = probabilities.detach().numpy()
        print(probabilities)


        #
        # with torch.no_grad():
        #
        #     predicted_class = torch.argmax(output, dim=1)
        #     print(predicted_class)

        return probabilities[0][0] > probabilities[0][1]
