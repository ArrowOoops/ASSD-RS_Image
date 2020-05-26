num_classes = 4
aspect_ratios = [[2], [2, 3], [2, 3], [2, 3], [2, 3], [2], [2]]
mbox = [4, 6, 6, 6, 6, 4, 4]
variance = [0.1, 0.2]
feature_maps = [65, 33, 17, 9, 5, 3, 1]

min_sizes = [  20.52,   51.3,   133.38,  215.46,  297.54,  379.62,  461.7 ]
max_sizes = [  51.3,   133.38,  215.46,  297.54,  379.62,  461.7,   543.78]

steps = [8, 16, 31, 57, 103, 171, 513]
top_k = 200

# detect settings
conf_thresh =  0.01
nms_thresh = 0.45

# Training settings
img_size = 513
batch_size = 7
epoch = 100
# lr_decay_epoch = 50
milestones = [120, 170, 220]

# data directory
# root = '/media/grace/Windows/ubuntu-backup/Datasets/PASCALVOC/VOCdevkit'
root = 'D:/BaiduNetdiskDownload/KY/2020/DataSet'


train_sets = [('13','train')]
test_sets = [('13', 'test')]

means = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
init_lr = 0.001
weight_decay = 0.0005

VOC_CLASSES = ('__background__', 'airplane', 'ship', 'storage_tank')

# dec evaluation
output_dir = 'output'

labelmap = ('airplane', 'ship', 'storage_tank')
