# -*- coding:utf-8 -*-
# ! ./usr/bin/env python
"""
This file gen 3 files in the path VID, they are 'train'/'test'/'anno'.
the 'train' is the train dataset.
the 'test' is the test dataset.
the 'anno' is the labels of the train/test dataset.
"""
# ---------------------- gen VID which contains three file: 'train'/'test'/'anno' ---------

"""
import os
import xml.etree.ElementTree as ET

target1 = ['250000-350000-64-1-1000-gray_0',
           '350000-450000-64-gray',
           '4000-40000-64-gray',
           '780000-850000-64-gray_0',
           '550000-650000-64-gray',
           '635000-730000-64-gray_0',
           '65000-88000-64-gray',
           '49000-59000-64-gray',
           '580000-630000-64-gray',
           '975290-1132575-64-gray',
           '910000-916308-64-gray',
           ]
target2 = ['18455-33000-64-1-400-gray_0',
           '780000-850000-64-gray_1',
           '635000-730000-64-gray_1',
           ]
target3 = ['18455-33000-64-1-400-gray_1',
           '250000-350000-64-1-1000-gray_1',
           ]
SEQLENGH = 100

imgpath = './JPEGImages/'
newimgpath = './VID/{}'
for phase in ['train', 'test', 'anno']:
    if not os.path.exists(newimgpath.format(phase)): os.makedirs(newimgpath.format(phase))

txtpath = './targetbox/'
annopath = './Annotations/'

filelist = os.listdir(imgpath)
filelist.sort()

def change_one_xml(xml_path, filename):
    doc = ET.parse(xml_path)
    root = doc.getroot()
    sub1 = root.find('filename')       
    sub1.text = '{:06}.jpg'.format(filename)       # change filename   
    doc.write(xml_path)  


fnum_train = 0
fnum_val = 0
for file in filelist:
    if file in target1:
        target = 'target1'
    elif file in target2:
        target = 'target2'
    elif file in target3:
        target = 'target3'
    else:
        print('file error')
    # gen ground_truth.txt
    boxpath = os.path.join(txtpath, file)
    for txt in os.listdir(boxpath):
        txtdir = os.path.join(boxpath, txt)
        with open(txtdir, 'r') as f:
            lines = f.readlines()

    print(file)
    imglist = os.listdir(imgpath + file)
    imglist.sort()
    seqlen = len(imglist)
    num_train = int(seqlen * 0.7)
    num_val = seqlen - num_train
    print(seqlen, num_train, num_val)
    imgid = 0
    for img in imglist:
        imgdir = os.path.join(imgpath, file, img)
        xmldir = os.path.join(annopath, file, img.replace('jpg', 'xml'))
        # update new filename
        if imgid <= num_train and imgid % SEQLENGH == 0:
            newfile = os.path.join(newimgpath.format('train'), 'got10k_{}_{}'.format(target,fnum_train))
            if not os.path.exists(newfile): os.makedirs(newfile)
            # new gt text
            newtxt = os.path.join(newfile, 'groundtruth.txt')
            open(newtxt, 'w')
            # new annotations
            newanno = os.path.join(newimgpath.format('anno/train'), 'got10k_{}_{}'.format(target,fnum_train))
            if not os.path.exists(newanno): os.makedirs(newanno)
            fnum_train += 1
        elif imgid % SEQLENGH == 0:
            newfile = os.path.join(newimgpath.format('test'), 'got10k_{}_{}'.format(target,fnum_val))
            if not os.path.exists(newfile): os.makedirs(newfile)
            # new gt text
            newtxt = os.path.join(newfile, 'groundtruth.txt')
            open(newtxt, 'w')
            # new annotations
            newanno = os.path.join(newimgpath.format('anno/test'), 'got10k_{}_{}'.format(target,fnum_val))
            if not os.path.exists(newanno): os.makedirs(newanno)
            fnum_val += 1

        # copy img to newpath
        newimgfile = os.path.join(newfile, 'img')
        if not os.path.exists(newimgfile): os.makedirs(newimgfile)
        newimgdir = os.path.join(newfile, 'img', '{:06}.jpg'.format(int(imgid % SEQLENGH)))
        print(newimgdir)
        os.system('cp {} {}'.format(imgdir, newimgdir))

        # copy xml to newanno
        newxmldir = os.path.join(newanno, '{:06}.xml'.format(int(imgid % SEQLENGH)))
        print(newxmldir)
        os.system('cp {} {}'.format(xmldir, newxmldir))
        change_one_xml(newxmldir, int(imgid % SEQLENGH))

        # write box to gt
        with open(newtxt, 'a') as f:
            box = lines[imgid].strip().split(' ')
            # box = [float(b) for b in box]
            print(box)
            f.write(box[0] + ',' + box[1] + ',' + box[2] + ',' + box[3] + '\n')


        imgid += 1


"""

# ---------------------------------- gen train set  ---------------------------------------
# --------- The video will be splited into video slices of length 100 for training --------
import os
import json
from collections import OrderedDict
from os.path import join

ROOT = './VID/'

train_path = ROOT + 'train'
jsonpath = ROOT + 'train.json'

data = OrderedDict()

flist = sorted(os.listdir(train_path))
for file in flist:
	imgs = []
	rects = []
	info = OrderedDict()
	gtText = join(train_path, file, 'groundtruth.txt')

	# process rect
	with open(gtText, 'r') as f:
		lines = f.readlines()
	for i,line in enumerate(lines):
		img = '{:06}.jpg'.format(i)
		imgs.append(img)
		rect = line.strip().split(',')
		box = [rect[0], rect[1], rect[2], rect[3]]
		box = [int(float(i)) for i in box]
		box = [box[0], box[1], box[0]+box[2], box[1]+box[3]] # to xyxy
		rects.append(box)
	
	info['name'] = file
	info['image_files'] = imgs
	info['init_rect'] = rects[0]
	info['gt_rect'] = rects
	data['{}'.format(file)] = info
	print(file, data[file]['name'])

with open(jsonpath, 'w') as ff:
	json.dump(data, ff, indent=2)


# -------------------------------- gen split set for test -------------------------
# --------- The video will be splited into video slices of length 10 --------------
import os
import json
from os.path import join, exists
from collections import OrderedDict

ROOT = './VID/'

filepath = ROOT + 'test'
newfilepath = ROOT + 'test_split'

annopath = ROOT + 'anno'
newannopath = ROOT + 'Annotations'

SEQLEN = 10

filelist = os.listdir(filepath)
filelist.sort()
imgid = 0
for file in filelist:
	# parse gt
	gt = join(filepath, file, 'groundtruth.txt')
	with open(gt, 'r') as f:
		lines = f.readlines()

	imglist = os.listdir(join(filepath, file, 'img'))
	imglist.sort()
	i = 0
	for img in imglist:
		imgdir = join(filepath, file, 'img', img)
		annodir = join(annopath, 'test', file, img.replace('jpg', 'xml'))
		if imgid % SEQLEN == 0:
			newfile = join(newfilepath, 'video_{}_{}'.format(file.split('_')[1], int(imgid/SEQLEN)))

			# img
			newimgfile = join(newfile, 'img')
			if not exists(newimgfile): os.makedirs(newimgfile)

			# anno
			newannofile = newfile.replace(newfilepath, newannopath) # Annotations/file/
			if not exists(newannofile): os.makedirs(newannofile)

			# groundtruth.txt
			newgt = join(newfile, 'groundtruth.txt')
			with open(newgt, 'w') as f: f.close()

		# # cp img
		newimgdir = join(newfile, 'img', '{:06}.jpg'.format(imgid % SEQLEN))
		os.system('cp {} {}'.format(imgdir, newimgdir))

		# cp anno
		newannodir = join(newannofile, '{:06}.xml'.format(imgid % SEQLEN))
		os.system('cp {} {}'.format(annodir, newannodir))

		# write to groundtruth.txt
		with open(newgt, 'a') as f2:
			print(i, len(lines))
			box = lines[i].strip().split(',')
			f2.write(box[0] + ',' + box[1] + ',' + box[2] + ',' + box[3] + '\n')

		i += 1
		imgid += 1

# gen test_split json
test_path = ROOT + './test_split'
jsonpath = ROOT + 'test_split.json'

data = OrderedDict()

flist = sorted(os.listdir(test_path))
for file in flist:
	imgs = []
	rects = []
	info = OrderedDict()
	gtText = join(test_path, file, 'groundtruth.txt')

	# process rect
	with open(gtText, 'r') as f:
		lines = f.readlines()
	for i,line in enumerate(lines):
		img = '{:06}.jpg'.format(i)
		imgs.append(img)
		rect = line.strip().split(',')
		box = [rect[0], rect[1], rect[2], rect[3]]
		box = [int(float(i)) for i in box]
		box = [box[0], box[1], box[0]+box[2], box[1]+box[3]] # to xyxy
		rects.append(box)
	
	info['name'] = file
	info['image_files'] = imgs
	info['init_rect'] = rects[0]
	info['gt_rect'] = rects
	data['{}'.format(file)] = info
	print(file, data[file]['name'])

with open(jsonpath, 'w') as ff:
	json.dump(data, ff, indent=2)



# gen /ImageSets/Main/test.txt
imgpath = ROOT + './test_split'
txtpath = ROOT + './ImageSets/Main'
if not os.path.exists(txtpath): os.makedirs(txtpath)
testdir = join(txtpath, '{}', 'test.txt')

for file in os.listdir(imgpath):
	tdir = testdir.format(file)
	testfiledir = tdir[:tdir.rfind('/')]
	if not os.path.exists(testfiledir): os.makedirs(testfiledir)
	with open(testdir.format(file), 'w') as f: pass # new file

filelist = os.listdir(imgpath)
for file in filelist:
	imglist = os.listdir(join(imgpath, file, 'img'))
	imglist.sort()
	for img in imglist:
		print(file, img)
		imgname, ext = os.path.splitext(img)
		with open(testdir.format(file), 'a') as f:
			f.write(imgname + '\n')
        

        
