import os
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset


class DataSet(Dataset):
    def __init__(self, root, train=True):
        self.root = root
        self.train = train
        self.images = os.listdir(self.root)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_index = self.images[index]
        image = Image.open(os.path.join(self.root, image_index))
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.ToTensor()
        ])
        return transform(image)
