"""VOC Dataset Classes

Original author: Francisco Massa
https://github.com/fmassa/vision/blob/voc_dataset/torchvision/datasets/voc.py

Updated by: Ellis Brown, Max deGroot
"""
from .config import HOME
import os
import os.path as osp
import sys
import torch
import torch.utils.data as data
import cv2
import numpy as np
import random
import json
from os.path import join
from torch.utils.data import Dataset, DataLoader
from data.config import rdvoc
sys.path.append('../')

if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET

RDVOC_CLASSES = (  # always index 0
    'target1', 'target2', 'target3')

# RDVOC_ROOT = "/media/orange/D/HWP/datasets/RDVOC_2019/OTBseq_all/"
# RDVOC_ROOT_test = "/media/orange/D/HWP/datasets/RDVOC_2019/OTBseq_all/"
RDVOC_ROOT = '../VID'
RDVOC_ROOT_test = '../VID'

LABEL_PATH = '../VID/{}.json'

sample_random = random.Random()

cfg = rdvoc

class RDVOCAnnotationTransform(object):
    """Transforms a VOC annotation into a Tensor of bbox coords and label index
    Initilized with a dictionary lookup of classnames to indexes

    Arguments:
        class_to_ind (dict, optional): dictionary lookup of classnames -> indexes
            (default: alphabetic indexing of VOC's 20 classes)
        keep_difficult (bool, optional): keep difficult instances or not
            (default: False)
        height (int): height
        width (int): width
    """

    def __init__(self, class_to_ind=None, keep_difficult=False):
        self.class_to_ind = class_to_ind or dict(
            zip(RDVOC_CLASSES, range(len(RDVOC_CLASSES))))
        self.keep_difficult = keep_difficult

    def __call__(self, target, width, height):
        """
        Arguments:
            target (annotation) : the target annotation to be made usable
                will be an ET.Element
        Returns:
            a list containing lists of bounding boxes  [bbox coords, class name]
        """
        res = []
        for obj in target.iter('object'):
            difficult = int(obj.find('difficult').text) == 1
            if not self.keep_difficult and difficult:
                continue
            name = obj.find('name').text.lower().strip()
            bbox = obj.find('bndbox')

            pts = ['xmin', 'ymin', 'xmax', 'ymax']
            bndbox = []
            for i, pt in enumerate(pts):
                cur_pt = int(bbox.find(pt).text) - 1
                # scale height or width
                cur_pt = cur_pt / width if i % 2 == 0 else cur_pt / height
                bndbox.append(cur_pt)
            label_idx = self.class_to_ind[name]
            bndbox.append(label_idx)
            res += [bndbox]  # [xmin, ymin, xmax, ymax, label_ind]
            # img_id = target.find('filename').text[:-4]

        return res  # [[xmin, ymin, xmax, ymax, label_ind], ... ]


class RDVOCDetection(data.Dataset):
    """VOC Detection Dataset Object

    input is image, target is annotation

    Arguments:
        root (string): filepath to VOCdevkit folder.
        image_set (string): imageset to use (eg. 'train', 'val', 'test')
        transform (callable, optional): transformation to perform on the
            input image
        target_transform (callable, optional): transformation to perform on the
            target `annotation`
            (eg: take in caption string, return tensor of word indices)
        dataset_name (string, optional): which dataset to load
            (default: 'VOC2007')
    """

    def __init__(self, root,
                 image_sets=[('RDVOC_CD0910')],
                 transform=None, target_transform=RDVOCAnnotationTransform(),
                 dataset_name='RDVOC_CD0910',
                 seq_len=5,
                 batch_size=5):
        self.root = root
        self.image_set = image_sets
        self.transform = transform
        self.target_transform = target_transform
        self.name = dataset_name
        self._annopath = osp.join('%s', 'Annotations', '%s', '%s.xml')
        self._imgpath = osp.join('%s', 'JPEGImages', '%s', '%s.jpg')
        self.ids = list()
        self.imagdir = osp.join(self.root, 'JPEGImages')
        rootpath = self.root
        self.seq_len = seq_len
        self.batch_size = batch_size

        # for img in os.listdir(self.imagdir):
        #     imgname = img.split('.')[0]
        #     self.ids.append((rootpath, imgname))

        self.txtpath = osp.join(rootpath, 'ImageSets', 'Main')

        name = image_sets[1]

        # for line in open(osp.join(rootpath, 'ImageSets', 'Main', name + '.txt')):
        #     self.ids.append((rootpath, line.strip()))
        # print(self.ids[:50])
        # self.ids = self._sampler(self.ids, seq_len)
        # print(self.ids[:50])

        self.ids = self.get_ids(name)


    def _sampler(self, ids_list, interval):
        listlen = len(ids_list)
        batch_len = (interval*self.batch_size)
        newlist = [[] for i in range(listlen - listlen % batch_len)]

        for batch in range(int(listlen/batch_len)):
            for i in range(interval):
                for j in range(self.batch_size):
                    newid = batch*batch_len + i*self.batch_size + j
                    oldid = batch*batch_len + j*interval + i
                    # print(newid, oldid)
                    newlist[newid] = ids_list[oldid]

        return newlist


    def get_ids(self, name):
        filelist = os.listdir(self.txtpath)
        filelist.sort()
        ids = []
        for file in filelist:
            filedir = osp.join(self.txtpath, file, '{}.txt'.format(name))
            for line in open(filedir, 'r'):
                ids.append((self.root, file, line.strip()))

        ids = self._sampler(ids, self.seq_len)
        # for item in ids[:200]:
        #     print("self.ids:", item)
        return ids


    def __getitem__(self, index):
        im, gt, h, w = self.pull_item(index)

        return im, gt

    def __len__(self):
        return len(self.ids)

    def pull_item(self, index):
        img_id = self.ids[index]
        target = ET.parse(self._annopath % img_id).getroot()
        img = cv2.imread(self._imgpath % img_id)
        height, width, channels = img.shape

        if self.target_transform is not None:
            target = self.target_transform(target, width, height)

        if self.transform is not None:
            target = np.array(target)
            img, boxes, labels = self.transform(img, target[:, :4], target[:, 4])
            # to rgb
            img = img[:, :, (2, 1, 0)]
            # img = img.transpose(2, 0, 1)
            target = np.hstack((boxes, np.expand_dims(labels, axis=1)))
        return torch.from_numpy(img).permute(2, 0, 1), target, height, width
        # return torch.from_numpy(img), target, height, width

    def pull_image(self, index):
        '''Returns the original image object at index in PIL form

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            PIL img
        '''
        img_id = self.ids[index]
        return cv2.imread(self._imgpath % img_id, cv2.IMREAD_COLOR)

    def pull_anno(self, index):
        '''Returns the original annotation of image at index

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to get annotation of
        Return:
            list:  [img_id, [(label, bbox coords),...]]
                eg: ('001718', [('dog', (96, 13, 438, 332))])
        '''
        img_id = self.ids[index]
        anno = ET.parse(self._annopath % img_id).getroot()
        gt = self.target_transform(anno, 1, 1)
        return img_id[1], gt

    def pull_tensor(self, index):
        '''Returns the original image at an index in tensor form

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            tensorized version of img, squeezed
        '''
        return torch.Tensor(self.pull_image(index)).unsqueeze_(0)

class SeqDataset(Dataset):
    def __init__(self, root, batch_size,
                 transform=None,
                 dataset_name='RDVOC',
                 phase='train'):
        super(SeqDataset, self).__init__()
        self.root = root
        self.anno = LABEL_PATH.format(phase)
        self.batch_size = batch_size
        self.name = dataset_name
        self.transform = transform
        self.phase = phase

        self.labels = json.load(open(self.anno, 'r'))
        self.videos = list(self.labels.keys())
        self.num = len(self.videos)   # video number
        self.frame_range = 100
        self.num_use = 2000 if phase == 'train' else self.num
        if phase == 'train':
            self.pick = self._shuffle()
        else:
            self.pick = list(range(0, self.num))

        self.txtpath = os.path.join(RDVOC_ROOT, 'ImageSets/Main')
        self.ids = self._get_ids('test')


    def __len__(self):
        return self.num_use

    def __getitem__(self, index):
        """
        pick a video/frame --> pairs --> label
        """
        index = self.pick[index] # pick a video
        imgdirs, targets = self._get_pairs(index) # pick a frame

        images = []
        for imgdir in imgdirs:
            im = cv2.imread(imgdir, cv2.IMREAD_COLOR)
            image = cv2.resize(im, (cfg['min_dim'],cfg['min_dim']))
            images.append(image)

        # from PIL image to numpy
        images = np.array(images, np.float32)
        images = torch.from_numpy(images).permute(0,3,1,2).contiguous()
        return images, targets

    # ------------------------------------
    # function groups for selecting pairs
    # ------------------------------------
    def _shuffle(self):
        """
        shuffel to get random pairs index
        """
        lists = list(range(0, self.num))
        m = 0
        pick = []
        while m < self.num_use:
            sample_random.shuffle(lists)
            pick += lists
            m += self.num

        self.pick = pick[:self.num_use]
        return self.pick

    def _target_transform(self, bbox, width, height):
        return [bbox[0]/width, bbox[1]/height, bbox[2]/width, bbox[3]/height]

    def _get_image_anno(self, video_name, frame):
        """
        get image and annotation
        """
        frame = "{:06d}".format(frame)
        image_path = join(self.root, self.phase, video_name, "img/{}.jpg".format(frame))
        image_anno = self.labels[video_name]['gt_rect'][int(frame)] # (xmin, ymin, xmax, ymax)
        image_anno = self._target_transform(image_anno, width=cfg['image_size'][0], height=cfg['image_size'][1])

        return image_path, image_anno

    def _get_pairs(self, index):
        """
        get training pairs
        """
        video_name = self.videos[index]
        video = self.labels[video_name] # selected video's info
        frames = video['image_files']
        name = video['name']
        cls = name.split('_')[1]
        label = RDVOC_CLASSES.index(cls)

        if self.phase == 'train':
            frame_0 = random.randint(0, len(frames) - self.batch_size)  # 随机选取一个序列中的某一帧
            seq = frames[frame_0: frame_0 + self.batch_size]
        else:
            seq = frames
        framenums = [int(frame.split('.')[0]) for frame in seq]

        imgs = []
        targets = []
        for frame in framenums:
            img, bbox = self._get_image_anno(video_name, frame)
            imgs.append(img)
            target = np.hstack((bbox, label))
            # print('target:', target)
            target = torch.from_numpy(target).float()
            targets.append(target)

        # targets = np.hstack((bboxes, np.expand_dims(labels, axis=1)))

        return imgs, targets

    def _get_ids(self, phase):
        filelist = os.listdir(self.txtpath)
        filelist.sort()
        ids = []
        for file in filelist:
            filedir = osp.join(self.txtpath, file, '{}.txt'.format(phase))
            for line in open(filedir, 'r'):
                ids.append((self.root, file, line.strip()))

        return ids