import config as cfg
import time
import os
import numpy as np
import pickle
import sys


if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET


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

def get_voc_results_file_template(image_set, cls):
    # VOCdevkit/VOC2007/results/det_test_aeroplane.txt
    filename = 'det_' + image_set + '_%s.txt' % (cls)
    filedir = os.path.join(cfg.output_dir, 'results')
    if not os.path.exists(filedir):
        os.makedirs(filedir)
    path = os.path.join(filedir, filename)
    return path

def voc_ap(rec, prec, use_07_metric=True):
    #根据传入的rec, prec, 求ap
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
# Y轴查准率p,X轴召回率r--
# 取11个点,如[r(0.0),p(0)],[r(0.1),p(1)],...,[r(1.0),p(10)],ap=(p(0)+p(1)+...+p(10))/11
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0: #召回率rec中大于阈值t的数量;等于0表示超过了最大召回率,对应的p设置为0
                p = 0
            else:
                p = np.max(prec[rec >= t]) #召回率大于t时精度的最大值
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
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1]) #计算PR曲线向下包围的面积
    return ap

def parse_rec(filename):
    #解析一个PASCAL VOC xml文件。返回值是key为
    #('name', 'pose', 'truncated', 'difficult', 'bbox(xmin, ymin, xmax, ymax)')构成的字典objects
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
        obj_struct['bbox'] = [int(bbox.find('ymin').text) - 1,
                              int(bbox.find('xmin').text) - 1,
                              int(bbox.find('ymax').text) - 1,
                              int(bbox.find('xmax').text) - 1]
        objects.append(obj_struct)

    return objects


def voc_eval(detpath,
             annopath,
             imagesetfile,
             classname,
             cachedir,
             ovthresh=0.5,
             use_07_metric=True):
    #根据测试结果路径、注释路径（gt）、类别名称等，输出特定类别的rec， prec, ap
    print('func:voc_eval start')
    if not os.path.isdir(cachedir):
        os.mkdir(cachedir)
    cachefile = os.path.join(cachedir, 'annots.pkl')
    # read list of images
    imagesetfile = "F:/xiejia/DataSet/VOC13/ImageSets/Main/test.txt"
    with open(imagesetfile, 'r') as f:
        lines = f.readlines()
    imagenames = [x.strip() for x in lines]
    if not os.path.isfile(cachefile):
        # load annots
        recs = {}
        for i, imagename in enumerate(imagenames):
            recs[imagename] = parse_rec(annopath % (imagename))
            # if i % 100 == 0:
                # print('Reading annotation for {:d}/{:d}'.format(
                #    i + 1, len(imagenames)))
        # save
        # print('Saving cached annotations to {:s}'.format(cachefile))
        with open(cachefile, 'wb') as f:
            pickle.dump(recs, f)
    else:
        # load
        with open(cachefile, 'rb') as f:
            recs = pickle.load(f)

    # extract gt objects for this class
    class_recs = {}
    npos = 0
    for imagename in imagenames:
        R = [obj for obj in recs[imagename] if obj['name'] == classname]
        bbox = np.array([x['bbox'] for x in R])
        difficult = np.array([x['difficult'] for x in R]).astype(np.bool)
        det = [False] * len(R)
        npos = npos + sum(~difficult)
        class_recs[imagename] = {'bbox': bbox,
                                 'difficult': difficult,
                                 'det': det}

    # read dets
    detfile = detpath.format(classname)
    print(detfile)
    print(classname)
    with open(detfile, 'r') as f:
        lines = f.readlines()
    #print(lines)
    if any(lines) == 1:
        splitlines = [x.strip().split(' ') for x in lines]
        image_ids = [x[0] for x in splitlines]
        confidence = np.array([float(x[1]) for x in splitlines])
        BB = np.array([[float(z) for z in x[2:]] for x in splitlines])
        #print('below is mAP in F-RCNN')
        # sort by confidence
        sorted_ind = np.argsort(-confidence)
        # sorted_scores = np.sort(-confidence)
        BB = BB[sorted_ind, :]
        image_ids = [image_ids[x] for x in sorted_ind]

        # go down dets and mark TPs and FPs
        nd = len(image_ids)
        tp = np.zeros(nd)
        fp = np.zeros(nd)
        print('origin nd,tp,fp:')
        print('nd={},tp={},fp={}'.format(nd,tp,fp))
        #print(str(tp.size()))
        for d in range(nd):
            R = class_recs[image_ids[d]]  #R表示当前帧图像上所有的GT bbox的信息
            bb = BB[d, :].astype(float)
            ovmax = -np.inf
            BBGT = R['bbox'].astype(float)
            if BBGT.size > 0:
                # compute overlaps
                # intersection
                iymin = np.maximum(BBGT[:, 0], bb[0])
                ixmin = np.maximum(BBGT[:, 1], bb[1])
                iymax = np.minimum(BBGT[:, 2], bb[2])
                ixmax = np.minimum(BBGT[:, 3], bb[3])
                iw = np.maximum(ixmax - ixmin, 0.)
                ih = np.maximum(iymax - iymin, 0.)
                inters = iw * ih
                uni = ((bb[2] - bb[0]) * (bb[3] - bb[1]) +
                       (BBGT[:, 2] - BBGT[:, 0]) *
                       (BBGT[:, 3] - BBGT[:, 1]) - inters)
                overlaps = inters / uni  #IOU
                ovmax = np.max(overlaps)
    # bb表示测试集中某一个检测出来的框的四个坐标，BBGT表示和bb同一图像上的所有检测框，取其中IOU最大的作为检测框的ground-true
                jmax = np.argmax(overlaps) #IOU取最大值时的先验框Index

            if ovmax > ovthresh:
                if not R['difficult'][jmax]:
                    if not R['det'][jmax]: # 判断是否被检测过(如果之前有置信度更高的bbox匹配上了这个BBGT,那么就表示检测过了)
                        tp[d] = 1. #预测为正，实际为正
                        R['det'][jmax] = 1
                    else:
                        fp[d] = 1. #预测为正，实际为负
            else:
                fp[d] = 1.

        # compute precision recall
        fp = np.cumsum(fp)
        tp = np.cumsum(tp)
        rec = tp / float(npos) #召回率
        print('below is computed fp,tp,rec')
        print(fp)
        print(tp)
        print(rec)
        # avoid divide by zero in case the first detection matches a difficult
        # ground truth
        prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps) # 精准率,查准率
        ap = voc_ap(rec, prec, use_07_metric)
        print('below is computed prec,ap')
        print(prec)
        print(ap)
    else:
        rec = -1.
        prec = -1.
        ap = -1.
    print('func:voc_eval end')
    return rec, prec, ap

def do_python_eval(output_dir='output', use_07=True):
    print('func:do_python_eval start')
    annopath = os.path.join(cfg.root, 'VOC13', 'Annotations', '%s.xml')
    imgsetpath = os.path.join(cfg.root, 'VOC13', 'ImageSets', 'Main', '{:s}.txt')

    cachedir = os.path.join(cfg.output_dir, 'annotations_cache')
    aps = []
    # The PASCAL VOC metric changed in 2010
    use_07_metric = use_07
    # print('VOC07 metric? ' + ('Yes' if use_07_metric else 'No'))
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    print('正确率Precision      P = TP/(TP+FP);')
    print('召回率Recall         R = TP/(TP+FN) = 1 - FN/T;')
    print('漏警率Missing Alarm  MA = FN/(TP + FN) = 1–TP/T = 1-R；')
    print('虚警率Flase Alarm    FA = FP/(TP + FP) = 1–P；')
    print('\n')
    for i, cls in enumerate(cfg.labelmap):
        filename = get_voc_results_file_template('test', cls)
        rec, prec, ap = voc_eval(
           filename, annopath, imgsetpath.format('test'), cls, cachedir,
           ovthresh=0.5, use_07_metric=use_07_metric)
        aps += [ap]

        # print('类别：'+cls)
        # print('平均正确率Average Precision for {} = {:.6f}'.format(cls, ap))
        # print('召回率Recall for {} = {:.6f}'.format(cls, rec[0]))
        # print('正确率Precision for {} = {:.6f}'.format(cls, prec[0]))
        # print('漏警率Missing Alarm for {} = {:.6f}'.format(cls, 1-rec[0]))
        # print('虚警率Flase Alarm for {} = {:.6f}'.format(cls, 1-prec[0]))
        # print('\n')
        print('类别：' + cls)
        # print('平均正确率AP = {:.6f}'.format(ap))
        # print('召回率R     = {:.6f}'.format(rec[0]))
        # print('正确率P     = {:.6f}'.format(prec[0]))
        # print('漏警率MA    = {:.6f}'.format(1 - rec[0]))
        # print('虚警率FA    = {:.6f}'.format(1 - prec[0]))
        # print('\n')
        print('rec,prec,ap:\n')
        print(rec)
        print(prec)
        print(ap)
        print('\n')

        with open(os.path.join(output_dir, cls + '_pr.pkl'), 'wb') as f:
            pickle.dump({'rec': rec, 'prec': prec, 'ap': ap}, f)
    print('\n')
    print('Mean AP = {:.6f}'.format(np.mean(aps)))
    return np.mean(aps)
    # print('~~~~~~~~')
    # print('Results:')
    # for ap in aps:
    #     print('{:.3f}'.format(ap))
    # print('{:.3f}'.format(np.mean(aps)))
    # print('~~~~~~~~')
    # print('')
    # print('--------------------------------------------------------------')
    # print('Results computed with the **unofficial** Python eval code.')
    # print('Results should be very close to the official MATLAB eval code.')
    # print('--------------------------------------------------------------')