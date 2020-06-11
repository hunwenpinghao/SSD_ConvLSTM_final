# config.py
import os.path

# gets home dir cross platform
HOME = os.path.expanduser("~")

# for making bounding boxes pretty
COLORS = ((255, 0, 0, 128), (0, 255, 0, 128), (0, 0, 255, 128),
          (0, 255, 255, 128), (255, 0, 255, 128), (255, 255, 0, 128))

MEANS = (104, 117, 123)

# SSD300 CONFIGS
voc = {
    'num_classes': 21,
    'lr_steps': (80000, 100000, 120000),
    'max_iter': 1200,
    'feature_maps': [38, 19, 10, 5, 3, 1],
    'min_dim': 300,
    'steps': [8, 16, 32, 64, 100, 300],
    'min_sizes': [30, 60, 111, 162, 213, 264],
    'max_sizes': [60, 111, 162, 213, 264, 315],
    'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
    'variance': [0.1, 0.2],
    'clip': True,
    'name': 'VOC',
}

rdvoc = {
    'num_classes': 4,
    'lr_steps': (80000, 100000, 120000),
    'max_iter': 120000,
    'feature_maps': [38, 19, 10, 5, 3, 1, 19],
    'min_dim': 300,
    'steps': [8, 16, 32, 64, 100, 300, 16], # fic is 300*300, step=300/feature_map=14
    # refrence : https://blog.csdn.net/rainforestgreen/article/details/82762274
    # min_ratio:20, max_ratio=90 interval=70/5=14
    # ratio : 20 34 48 62 76 90
    # min_size : min_dim*ratio/100 for first ratio=10
    'min_sizes': [30, 60, 102, 144, 186, 228, 270],
    # max_size : min_dim*(ratio+step)/100 first :max_size=min_dim*20/100
    'max_sizes': [60, 102, 144, 186, 228, 270, 312],
    'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2], [2]],
    'variance': [0.1, 0.2],
    'clip': True,
    'name': 'RDVOC',
    'image_size' : [320, 64] # [w,h]
}

coco = {
    'num_classes': 201,
    'lr_steps': (280000, 360000, 400000),
    'max_iter': 400000,
    'feature_maps': [38, 19, 10, 5, 3, 1],
    'min_dim': 300,
    'steps': [8, 16, 32, 64, 100, 300],
    'min_sizes': [21, 45, 99, 153, 207, 261],
    'max_sizes': [45, 99, 153, 207, 261, 315],
    'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
    'variance': [0.1, 0.2],
    'clip': True,
    'name': 'COCO',
}
