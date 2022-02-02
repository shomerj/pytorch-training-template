import os
import cv2
import json
import torch
import numpy as np
import pandas as pd
from PIL import Image
from pprint import pprint
from torch.utils.data import Dataset
import torchvision.transforms.transforms as T
import torchvision.transforms.functional as F
from config_parser import write_json


class ModelDataset(Dataset):
    def __init__(self, train, args, eval=False):

        if type(args) == str:
            args = json.load(open(args))
        self.img_dir = args['data_loader']['args']['img_dir']
        self.train = train
        if self.train:
            self.data_df = args['data_loader']['args']['train_df']
        else:
            self.data_df = args['data_loader']['args']['test_df']
        self.normalize = args['data_loader']['args']['normalize']
        self.inp_res = args['data_loader']['args']['input_size']
        self.transforms = args['data_loader']['args']['transforms']
        self.balance_classes = args['data_loader']['args']['balance']
        self.sample_size = args['data_loader']['args']['sample_size']
        self.sample_set = args['data_loader']['args']['sample_set']
        self.evaluate = eval
        self.class_mapping = args['data_loader']['args']['class_mapping']
        self.random_crop = args['data_loader']['args']['random_crop']
        self.included_classes = sorted(args['data_loader']['args']['included_classes'])
        self.num_classes = len(self.included_classes)

        if self.class_mapping == None:
            self.class_mapping = {k:i for i, k in enumerate(self.included_classes)}
        self.inv_class_mapping = {v:k for k, v in self.class_mapping.items()}

        self.data_df = pd.read_pickle(self.data_df)
        self.load_dataset()

        if self.train == False: ## creating sample test set
            self.sample_size = int(self.sample_size*0.25)

        if self.sample_set:
            self.create_sample_set()

        if self.normalize:
            self.mean = args['data_loader']['args'].get('mean')
            self.std = args['data_loader']['args'].get('std')
            if self.mean is None or self.std is None:
                self.mean, self.std = self._compute_mean()
                args['data_loader']['args']['mean'] = self.mean
                args['data_loader']['args']['std'] = self.std
                write_json(args, os.path.join(args['trainer']['save_dir'], 'config.json' )

    def return_transforms(self):
        transforms = []
        transforms.append(T.Resize(self.inp_res))

        if self.random_crop and self.train:
            transforms.append(T.RandomCrop(300, pad_if_needed=True))

        if len(self.transforms) > 0 and self.train == True:
            transforms.extend([getattr(T, trans)(**args) for trans, args in self.transforms.items()])
            transforms.append(T.RandomApply(torch.nn.ModuleList(transforms), p=0.1))

        transforms.append(T.ToTensor())
        if self.cutout and self.train:
            transforms.append(T.RandomErasing(p=0.4, scale=(0.01,0.03)))

        transforms = T.Compose(transforms)
        return transforms

    def _compute_mean(self):
        print('==> compute mean')
        mean = torch.zeros(3)
        std = torch.zeros(3)
        cnt = 0
        for img, _ in self.samples:
            cnt += 1
            print( '{} | {}'.format(cnt, len(self.samples)))
            img = cv2.imread(img)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = np.transpose(img, (2,0,1))
            img = img/255
            img = torch.from_numpy(img)
            mean += img.view(img.size(0), -1).mean(1)
            std += img.view(img.size(0), -1).std(1)
        mean /= len(self.samples)
        std /= len(self.samples)

        if self.train:
            print('    Mean: %.4f, %.4f, %.4f' % (mean[0], mean[1], mean[2]))
            print('    Std:  %.4f, %.4f, %.4f' % (std[0], std[1], std[2]))
        torch.save({'mean':mean, 'std':std}, 'meanstd.pkl')
        return mean, std

    def load_dataset(self):
        """
        Loading and printing some info about the dataset
        """
        self.samples = []
        self.class_count = {}
        for cls in self.included_classes:
            count = self.data_df[cls].sum()
            self.class_count[cls] = count

        annot_df = self.data_df[self.included_classes]
        self.class_count['negative'] = annot_df.loc[(annot_df==0).all(1)].shape[0]

        title = 'Train Dataset' if self.train else "Test Dataset"
        for i, row in self.data_df.iterrows():
            label = [0 for i in range(len(self.included_classes))]
            for key, val in row.items():
                if key in self.included_classes:
                    label[self.class_mapping[key]] = int(val)
                elif key == 'filename':
                    path = os.path.join(self.img_dir, val)
            self.samples.append((path, label))

        print('***'*10)
        print(title)
        pprint(self.class_count)
        print('***'*10)

    def __len__(self):
        return len(self.samples)

    def sampling_weights(self):
        class_weights = {cls:1./cnt for cls, cnt in self.class_count.items()}
        weights = []
        for i, (_, label) in enumerate(self.samples):
            label = np.array(label)
            indecies = np.argwhere(label==1)
            if len(indecies) == 0:
                weights.append(class_weights['negative'])
            elif len(indecies) > 1:#oversampling class with lowest count
                min = len(self)
                for idx in indecies:
                    lab = self.inv_class_mapping[int(idx)]
                    count = self.class_count[lab]
                    if count < min:
                        min = count
                        id = idx
                lab = self.inv_class_mapping[int(id)]
                weights.append(class_weights[lab])
            else:
                lab = self.inv_class_mapping[int(indecies)]
                weights.append(class_weights[lab])
        return weights

    def create_sample_set(self):
        samples = []
        indexs = np.arange(len(self.samples))
        weights = self.sampling_weights()
        p = np.exp(weights)/sum(np.exp(weights))
        indexs = np.random.choice(indexs, self.sample_size, replace=False, p=p)
        self.class_count = {cls: 0 for cls in self.included_classes}
        self.class_count['negative'] = 0
        for idx in indexs:
            path, label = self.samples[idx]
            indecies = np.argwhere(np.array(label)==1)
            if len(indecies) == 0:
                self.class_count['negative'] += 1
            else:
                for idx in indecies:
                    lab = self.inv_class_mapping[int(idx)]
                    self.class_count[lab] += 1
            samples.append((path, label))
        self.samples= samples
        print('Sample Set Class Count.....')
        pprint(self.class_count)

    def grab_random_images(self, num_images=4):
        transforms = []
        indexes = np.random.randint(0, len(self), num_images)
        batch = torch.zeros(num_images, 3, self.inp_res[0], self.inp_res[1], dtype=torch.float32)
        target = torch.zeros(num_images, self.num_classes)
        for i, idx in enumerate(indexes):
            img, label = self[idx]
            target[i] = label
            batch[i] = img
        return batch, target

    def crop_center(self, img):
         if type(img) == str:
             img = Image.open(img)
         ###TODO:
         # Make this dynamic
         width_expand = 0.234375
         height_expand = 0.13888888
         ###
         center = img.size[0]//2, img.size[1]//2
         crop_width = int(width_expand*img.size[0])
         crop_height = int(height_expand*img.size[1])
         crop = img.crop((center[0], center[1]-crop_height, center[0]+crop_width, center[1]+crop_height))
         crop = crop.resize((128,128))
         return crop

    def __getitem__(self, idx):
        img, cls = self.samples[idx]
        inp = Image.open(img)
        if inp.mode == 'RGBA':
            inp = inp.convert('RGB')

        transforms = []
        if self.random_crop and self.train:
            transforms.append(T.RandomCrop(300, pad_if_needed=True))
        if len(self.transforms) > 0 and self.train == True:
            t.extend([getattr(T, trans)(**args) for trans, args in self.transforms.items()])

        if self.random_apply:
            transforms.append(T.RandomApply(torch.nn.ModuleList(t), p=0.4))
        transforms.append(T.Resize(self.inp_res))
        transforms.append(T.ToTensor())

        if self.cutout and self.train:
            transforms.append(T.RandomErasing(**self.transforms['cutout']))
        transforms = T.Compose(transforms)
        inp = transforms(inp)

        if self.normalize:
            inp = F.normalize(inp, self.mean, self.std)
        target = torch.tensor(cls, dtype=torch.float32)
        if self.evaluate:
            return inp, target, img
        else:
            return inp, target
