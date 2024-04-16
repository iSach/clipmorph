import os

from PIL import Image
from torch.utils import data
from torchvision import transforms as T


class Data(data.Dataset):
    def __init__(self, root_dir, img_size):
        super().__init__()
        self.root_dir = root_dir
        self.img_names = os.listdir(root_dir)
        self.img_size = img_size

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, ind):
        img_name = self.img_names[ind]
        img_path = self.root_dir + "/" + img_name
        img = Image.open(img_path, mode='r')

        # Resize image and rescale pixel values from [0,1] to [0,255].
        transform = T.Compose([
            T.Resize(self.img_size),
            T.CenterCrop(self.img_size),
            T.ToTensor(),
            T.Lambda(lambda x: x.mul(255))
        ])

        # Expand 1-channel images to 3 channels.
        img = transform(img)
        if img.size()[0] != 3:
            img = img.expand(3, -1, -1)
        return img


def load_data(root_dir, img_size, batch_size):
    data_train = Data(root_dir, img_size)

    train_data_loader = data.DataLoader(
        data_train,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True
    )

    return train_data_loader


def load_image(filename, size=None):
    img = Image.open(filename)
    if size is not None:
        img = img.resize((size, size))
    return img
