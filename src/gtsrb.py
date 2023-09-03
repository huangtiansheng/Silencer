import os
from PIL import Image
import csv
import torch.utils.data as data
import numpy as np

class GTSRB(data.Dataset):
    def __init__(self, data_root, train, transform):
        super(GTSRB, self).__init__()
        self.transform = transform
        if train:
            self.data_folder = os.path.join(data_root, "Train")
            self.data, self.targets = self._get_data_train_list()
            if not os.path.isdir(self.data_folder):
                os.makedirs(self.data_folder)
        else:
            self.data_folder = os.path.join(data_root, "Test")
            self.data, self.targets = self._get_data_test_list()
            if not os.path.isdir(self.data_folder):
                os.makedirs(self.data_folder)
        

    def _get_data_train_list(self):
        images = []
        targets = []
        for c in range(0, 43):
            prefix = self.data_folder + "/" + format(c, "05d") + "/"
            if not os.path.isdir(prefix):
                os.makedirs(prefix)
            gtFile = open(prefix + "GT-" + format(c, "05d") + ".csv")
            gtReader = csv.reader(gtFile, delimiter=";")
            next(gtReader)
            for row in gtReader:
                # images.append(prefix + row[0])
                images.append(np.asarray(Image.open(prefix + row[0])))
                # print(images[-1].shape)
                targets.append(int(row[7]))
            gtFile.close()
            print(c)
        return images, targets

    def _get_data_test_list(self):
        images = []
        targets = []
        prefix = os.path.join(self.data_folder, "GT-final_test.csv")
        gtFile = open(prefix)
        gtReader = csv.reader(gtFile, delimiter=";")
        next(gtReader)
        for row in gtReader:
            images.append(np.asarray(Image.open(self.data_folder + '' + "/" + row[0])))
            targets.append(int(row[7]))
        return images,targets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image = self.data[index]
        image = Image.fromarray(image)
        target = self.targets[index]
        image = self.transform(image)
        return image, target