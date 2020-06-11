"""Adapted from:
    @longcw faster_rcnn_pytorch: https://github.com/longcw/faster_rcnn_pytorch
    @rbgirshick py-faster-rcnn https://github.com/rbgirshick/py-faster-rcnn
    Licensed under The MIT License [see LICENSE for details]
"""

from __future__ import print_function
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from data import RDVOC_ROOT_test, RDVOCAnnotationTransform, RDVOCDetection, BaseTransform, SeqDataset
from data import RDVOC_CLASSES as labelmap
import torch.utils.data as data

from ssd import build_ssd
from utils.imgshow import plot
from data.config import rdvoc as cfg

import sys
import os
import time
import argparse
import numpy as np
import pickle
import cv2

if sys.version_info[0] == 2: # get python version
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


parser = argparse.ArgumentParser(
    description='Single Shot MultiBox Detector Evaluation')
parser.add_argument('--trained_model',
                    default='weights/ssd300_COCO_50000.pth', type=str,
                    help='Trained state_dict file path to open')
parser.add_argument('--save_folder', default='eval/', type=str,
                    help='File path to save results')
parser.add_argument('--confidence_threshold', default=0.01, type=float,
                    help='Detection confidence threshold')
parser.add_argument('--top_k', default=5, type=int,
                    help='Further restrict the number of predictions to parse')
parser.add_argument('--cuda', default=True, type=str2bool,
                    help='Use cuda to train model')
parser.add_argument('--voc_root', default=RDVOC_ROOT_test,
                    help='Location of VOC root directory')
parser.add_argument('--cleanup', default=True, type=str2bool,
                    help='Cleanup and remove results files following eval')
parser.add_argument('--visdom', default=False, type=str2bool,
                    help='show the img and box')
parser.add_argument('--use_pickle', default=False, type=str2bool,
                    help='use pickle or not')

args = parser.parse_args()

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)

if torch.cuda.is_available():
    if args.cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    if not args.cuda:
        print("WARNING: It looks like you have a CUDA device, but aren't using \
              CUDA.  Run with --cuda for optimal eval speed.")
        torch.set_default_tensor_type('torch.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

annopath = os.path.join(args.voc_root, 'Annotations', '%s', '%s.xml')
imgsetpath = os.path.join(args.voc_root, "ImageSets/Main/", '%s', '%s.txt')

devkit_path = os.path.join(args.voc_root,'VOC2007')
dataset_mean = (104, 117, 123)
set_type = 'test'
DataSetROOT = args.voc_root
UsePickle = args.use_pickle

classmap = ['target1', 'target2', 'target3']


class Timer(object):
    """A simple timer."""
    def __init__(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.

    def tic(self):
        # using time.time instead of time.clock because time time.clock
        # does not normalize for multithreading
        self.start_time = time.time()

    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        self.average_time = self.total_time / self.calls
        if average:
            return self.average_time
        else:
            return self.diff


def parse_rec(filename):
    """ Parse a PASCAL VOC xml file """
    tree = ET.parse(filename)
    objects = []
    for obj in tree.findall('object'):
        obj_struct = {}
        obj_struct['name'] = obj.find('name').text
        obj_struct['pose'] = obj.find('pose').text
        obj_struct['truncated'] = int(obj.find('truncated').text)
        obj_struct['difficult'] = int(obj.find('difficult').text)
        bbox = obj.find('bndbox')
        obj_struct['bbox'] = [int(bbox.find('xmin').text) - 1,
                              int(bbox.find('ymin').text) - 1,
                              int(bbox.find('xmax').text) - 1,
                              int(bbox.find('ymax').text) - 1]
        objects.append(obj_struct)

    return objects


def get_output_dir(name, phase):
    """Return the directory where experimental artifacts are placed.
    If the directory does not exist, it is created.
    A canonical path is built using the name from an imdb and a network
    (if not None).
    """
    filedir = os.path.join(name, phase)
    if not os.path.exists(filedir):
        os.makedirs(filedir)
    return filedir


def get_voc_results_file_template(image_set, cls):
    # VOCdevkit/VOC2007/results/det_test_aeroplane.txt
    filename = 'det_' + image_set + '_%s.txt' % (cls)
    filedir = os.path.join(devkit_path, 'results')
    if not os.path.exists(filedir):
        os.makedirs(filedir)
    path = os.path.join(filedir, filename)
    return path


def write_voc_results_file(all_boxes, dataset):
    for cls_ind, cls in enumerate(labelmap):
        print('Writing {:s} VOC results file'.format(cls))
        filename = get_voc_results_file_template(set_type, cls) #/datasets/RDVOC_test_mini/VOC2007/results/det_test_target1.txt
        with open(filename, 'wt') as f:
            for im_ind, index in enumerate(dataset.ids):
                # print('im_ind:', im_ind, '\n', 'index:', index)
                dets = all_boxes[cls_ind+1][im_ind]
                # if dets == []:
                if isinstance(dets, list):
                    continue
                # the VOCdevkit expects 1-based indices
                for k in range(dets.shape[0]):
                    f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'. # imgid, score, xmin, ymin, xmax, ymax
                            format(os.path.join(index[0],'Annotations',index[1],'%s.xml'%index[2]), dets[k, -1],
                                dets[k, 0] + 1, dets[k, 1] + 1,
                                dets[k, 2] + 1, dets[k, 3] + 1))


def do_python_eval(gt_boxes, output_dir='output', use_07=True):
    cachedir = os.path.join(devkit_path, 'annotations_cache')
    aps = []
    # The PASCAL VOC metric changed in 2010
    use_07_metric = use_07
    print('VOC07 metric? ' + ('Yes' if use_07_metric else 'No'))
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    for i, cls in enumerate(labelmap):
        filename = get_voc_results_file_template(set_type, cls)
        rec, prec, ap = voc_eval(
           filename, gt_boxes, annopath, imgsetpath, cls, cachedir,
           ovthresh=0.5, use_07_metric=use_07_metric)
        aps += [ap]
        print('AP for {} = {:.4f}'.format(cls, ap))
        with open(os.path.join(output_dir, cls + '_pr.pkl'), 'wb') as f:
            pickle.dump({'rec': rec, 'prec': prec, 'ap': ap}, f)
    print('Mean AP = {:.4f}'.format(np.mean(aps)))
    print('~~~~~~~~')
    print('Results:')
    for ap in aps:
        print('{:.3f}'.format(ap))
    print('{:.3f}'.format(np.mean(aps)))
    print('~~~~~~~~')
    print('')
    print('--------------------------------------------------------------')
    print('Results computed with the **unofficial** Python eval code.')
    print('Results should be very close to the official MATLAB eval code.')
    print('--------------------------------------------------------------')


def voc_ap(rec, prec, use_07_metric=True):
    """
    ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:True).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def voc_eval(detpath,
             gt_boxes,
             annopath,
             imagesetfile,
             classname,
             cachedir,
             ovthresh=0.5,
             use_07_metric=True):
    """rec, prec, ap = voc_eval(detpath,
                           annopath,
                           imagesetfile,
                           classname,
                           [ovthresh],
                           [use_07_metric])
Top level function that does the PASCAL VOC evaluation.
detpath: Path to detections
   detpath.format(classname) should produce the detection results file.
annopath: Path to annotations
   annopath.format(imagename) should be the xml annotations file.
imagesetfile: Text file containing the list of images, one image per line.
classname: Category name (duh)
cachedir: Directory for caching the annotations
[ovthresh]: Overlap threshold (default = 0.5)
[use_07_metric]: Whether to use VOC07's 11 point AP computation
   (default True)
"""
    # first load gt
    if not os.path.isdir(cachedir):
        os.mkdir(cachedir)
    cachefile = os.path.join(cachedir, 'annots.pkl')
    # read list of images
    annonames = []
    for file in os.listdir(DataSetROOT + '/' + 'ImageSets/Main'):
        with open(imagesetfile % (file, set_type), 'r') as f:  # ImageSets/Main/test.txt
            lines = f.readlines()
        annonames = annonames + [annopath % (file, x.strip()) for x in lines]

    ## new add for removing .pkl first
    if not UsePickle:
        if os.path.isfile(cachefile):
            os.system("rm -rf {}".format(cachefile))
    if not os.path.isfile(cachefile):
        # load annots
        recs = {}
        for i, annoname in enumerate(annonames):
            recs[annoname] = parse_rec(annoname)
        # save
        print('Saving cached annotations to {:s}'.format(cachefile))
        with open(cachefile, 'wb') as f:
            pickle.dump(recs, f)
    else:
        # load
        with open(cachefile, 'rb') as f:
            recs = pickle.load(f)

    # extract gt objects for this class
    class_recs = {}
    npos = 0
    for annoname in annonames:
        R = [obj for obj in recs[annoname] if obj['name'] == classname]  # R is xml_info cressponding to classname
        bbox = np.array([x['bbox'] for x in R])
        difficult = np.array([x['difficult'] for x in R]).astype(np.bool)
        det = [False] * len(R)
        npos = npos + sum(~difficult)
        class_recs[annoname] = {'bbox': bbox,
                                'difficult': difficult,
                                'det': det}
    # read dets
    detfile = detpath.format(classname)  # det_test_target1.txt
    with open(detfile, 'r') as f:
        lines = f.readlines()
    if any(lines) == 1:
        splitlines = [x.strip().split(' ') for x in lines]
        image_ids = [x[0] for x in splitlines]
        # print("image_ids: ", image_ids)
        confidence = np.array([float(x[1]) for x in splitlines])
        # print("confidence", confidence)
        BB = np.array([[float(z) for z in x[2:]] for x in splitlines])

        # sort by confidence
        sorted_ind = np.argsort(-confidence)
        # print("sorted_ind: ", sorted_ind)
        sorted_scores = np.sort(-confidence)
        BB = BB[sorted_ind, :]
        image_ids = [image_ids[x] for x in sorted_ind]

        # go down dets and mark TPs and FPs
        nd = len(image_ids)
        tp = np.zeros(nd)
        fp = np.zeros(nd)
        for d in range(nd):
            R = class_recs[image_ids[d]]
            bb = BB[d, :].astype(float)  # sorted predict bb
            ovmax = -np.inf
            BBGT = R['bbox'].astype(float)  # grundtruth bb

            if args.visdom:
                img = dataset.pull_image(int(image_ids[d]))
                if BBGT.size > 0:
                    # print('BBGT[0]: ', BBGT[0])
                    plot(img, bb, BBGT[0], classname, classname)

            if BBGT.size > 0:
                # compute overlaps
                # intersection
                ixmin = np.maximum(BBGT[:, 0], bb[0])
                iymin = np.maximum(BBGT[:, 1], bb[1])
                ixmax = np.minimum(BBGT[:, 2], bb[2])
                iymax = np.minimum(BBGT[:, 3], bb[3])
                iw = np.maximum(ixmax - ixmin, 0.)
                ih = np.maximum(iymax - iymin, 0.)
                inters = iw * ih
                uni = ((bb[2] - bb[0]) * (bb[3] - bb[1]) +
                       (BBGT[:, 2] - BBGT[:, 0]) *
                       (BBGT[:, 3] - BBGT[:, 1]) - inters)
                overlaps = inters / uni
                ovmax = np.max(overlaps)
                jmax = np.argmax(overlaps)

            if ovmax > ovthresh:
                if not R['difficult'][jmax]:
                    if not R['det'][jmax]:
                        tp[d] = 1.
                        R['det'][jmax] = 1
                    else:
                        fp[d] = 1.
            else:
                fp[d] = 1.

        # compute precision recall
        fp = np.cumsum(fp)
        tp = np.cumsum(tp)
        rec = tp / float(npos)
        # avoid divide by zero in case the first detection matches a difficult
        # ground truth
        prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
        ap = voc_ap(rec, prec, use_07_metric)
    else:
        rec = -1.
        prec = -1.
        ap = -1.

    return rec, prec, ap


def test_net(save_folder, net, cuda, dataset, transform, BatchSize,
             im_size=300, thresh=0.05):
    num_images = len(dataset) * BatchSize
    all_boxes = [[[] for _ in range(num_images)]
                 for _ in range(len(labelmap)+1)]
    gt_boxes = [[[] for _ in range(num_images)]
                 for _ in range(len(labelmap) + 1)]

    data_loader = data.DataLoader(dataset, batch_size=1,  # 这里的batch_size必须为１，因为dataset中每次迭代已经改为为batch_size个
                                  num_workers=4,
                                  shuffle=False, pin_memory=True)
    batch_iterator = iter(data_loader)

    # timers
    _t = {'im_detect': Timer(), 'misc': Timer()}
    output_dir = get_output_dir('ssd300_120000', set_type)
    det_file = os.path.join(output_dir, 'detections.pkl')
    gt_file = os.path.join(output_dir, 'goundtruth.pkl')

    if not UsePickle or not os.path.exists(det_file):
        for idx in range(len(dataset)):
            images, targets = next(batch_iterator)
            # print('targets.size:', [targets[i].size() for i in range(len(targets))])
            if args.cuda:
                images = Variable(images).cuda()
            else:
                images = Variable(images)
            # 去掉最外面一维
            images = torch.squeeze(images, dim=0)

            _t['im_detect'].tic()
            detections = net(images).data
            detect_time = _t['im_detect'].toc(average=False)

            w, h = cfg['image_size']
            for i in range(detections.size(0)): # iter bz
                imgid = idx * BatchSize + i
                # skip j = 0, because it's the background class
                for j in range(1, detections.size(1)): # iter cls
                    dets = detections[i, j, :] #　N * 5 array
                    # print("gt(0.):", dets[:, 0].gt(0.).expand(5, dets.size(0)).t().size())
                    mask = dets[:, 0].gt(0.).expand(5, dets.size(0)).t() # .gt()：比较前者张量是否大于后者 expand：扩大为更高维　.t() means transpose
                    dets = torch.masked_select(dets, mask).view(-1, 5)
                    if dets.size(0) == 0:
                        continue
                    boxes = dets[:, 1:]
                    boxes[:, 0] *= w
                    boxes[:, 2] *= w
                    boxes[:, 1] *= h
                    boxes[:, 3] *= h
                    scores = dets[:, 0].cpu().numpy()
                    cls_dets = np.hstack((boxes.cpu().numpy(),
                                          scores[:, np.newaxis])).astype(np.float32,
                                                                         copy=False)
                    all_boxes[j][imgid] = cls_dets

                    # append gt
                    gt_boxes[j][imgid] = [target.numpy() for target in targets]

            print('im_detect: {:d}/{:d} {:.3f}s/batch'.format(imgid+1,
                                                            num_images, detect_time))

            with open(det_file, 'wb') as f:
                pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)
            with open(gt_file, 'wb') as f:
                pickle.dump(gt_boxes, f, pickle.HIGHEST_PROTOCOL)
    else:
        if os.path.exists(det_file):
            with open(det_file, 'rb') as f:
                all_boxes = pickle.load(f)
        if os.path.exists(gt_file):
            with open(gt_file, 'rb') as ff:
                gt_boxes = pickle.load(ff)
    print('Evaluating detections')
    evaluate_detections(all_boxes, gt_boxes, output_dir, dataset)


def evaluate_detections(box_list, gt_boxes, output_dir, dataset):
    write_voc_results_file(box_list, dataset)
    do_python_eval(gt_boxes, output_dir)


if __name__ == '__main__':
    # load net
    BatchSize = 10
    num_classes = len(labelmap) + 1                      # +1 for background
    net = build_ssd('test', 300, num_classes)            # initialize SSD
    net.load_state_dict(torch.load(args.trained_model))
    net.eval()
    print('Finished loading model!')
    # load data
    # dataset = RDVOCDetection(args.voc_root,
    #                          [('RDVOC_new'),('test')],
    #                          BaseTransform(300, dataset_mean),
    #                          RDVOCAnnotationTransform(),
    #                          "RDVOC_new", # dataset_name
    #                          seq_len = cfg['clstm_steps'],
    #                          batch_size = BatchSize
    #                          )
    dataset = SeqDataset(root=args.voc_root,
                         batch_size=BatchSize,
                         phase='test_split')

    if args.cuda:
        net = net.cuda()
        cudnn.benchmark = True
    # evaluation
    test_net(args.save_folder, net, args.cuda, dataset,
             BaseTransform(net.size, dataset_mean), BatchSize, 300,
             thresh=args.confidence_threshold)
