import torchvision.datasets as datasets
import torch
import argparse
import os
import glob
import numpy as np
import xml.etree.ElementTree as ET
import collections
from sklearn.cluster import KMeans


def parse_voc_xml(node):
    voc_dict = {}
    children = list(node)
    if children:
        def_dic = collections.defaultdict(list)
        for dc in map(parse_voc_xml, children):
            for ind, v in dc.items():
                def_dic[ind].append(v)
        if node.tag == 'annotation':
            def_dic['object'] = [def_dic['object']]
        voc_dict = {
            node.tag:
                {ind: v[0] if len(v) == 1 else v
                    for ind, v in def_dic.items()}
        }
    if node.text:
        text = node.text.strip()
        if not children:
            voc_dict[node.tag] = text
    return voc_dict


def coco_class_process(args):
    coco_det = datasets.CocoDetection(
        os.path.join(args.dataset_dir, 'train2017'),
        os.path.join(args.dataset_dir, 'annotations/instances_train2017.json'))

    num_classes = len(coco_det.coco.getCatIds())
    num_children = torch.as_tensor([2]*num_classes)  # 每个类别内的子类别个数

    classes = coco_det.coco.getCatIds()
    cwise_anns_ids = [coco_det.coco.getAnnIds(catIds=c) for c in classes]
    num_anns = len(coco_det.coco.getAnnIds())

    ann_cat_map = {}

    for idx, anns_ids in enumerate(cwise_anns_ids):
        anns = coco_det.coco.loadAnns(anns_ids)
        bboxes = [ann['bbox'] for ann in anns]
        _, _, w, h = tuple(zip(*bboxes))
        w, h = torch.as_tensor(w), torch.as_tensor(h)
        ratio = torch.min(w, h)/torch.max(w, h)

        # 子类别
        intra_classes = KMeans(n_clusters=2).fit_predict(ratio.view(-1, 1).numpy())
        generated_classes = intra_classes + num_children[:idx].sum().item() + 1  # intra class indices start from 1

        # 保存每个标签对应的子类别
        ann_cat_map.update({k: v for k, v in zip(anns_ids, generated_classes)})

    return {'convert_array': ann_cat_map, 'num_intra_classes': num_children}


def voc_class_process(args):
    categories_txt_paths = glob.glob(os.path.join(args.dataset_dir, 'VOCdevkit', args.dataset_name.upper(), 'ImageSets', 'Main', '*_train.txt'))
    ann_cat_map, secondary_index, category_ids = {}, {}, {}
    num_children = np.asarray([2]*len(categories_txt_paths))  # 每个类别内的子类别个数

    ann_id = 0  # 为每个标签创建索引
    for cid, categories_txt_path in enumerate(categories_txt_paths):
        category, _ = os.path.basename(categories_txt_path).split('_')
        lines = open(categories_txt_path, 'r').readlines()
        lines = [(line.split()[0], int(line.split()[1])) for line in lines]
        img_ids, validities = tuple(zip(*lines))
        validities = np.asarray(validities) != -1
        img_ids = np.asarray(img_ids)

        ann_paths = [os.path.join(args.dataset_dir, 'VOCdevkit', args.dataset_name.upper(), 'Annotations', '{}.xml').format(_id) for _id in img_ids[validities]]

        cwise_anns = []
        for ann_path in ann_paths:
            if os.path.basename(ann_path) not in secondary_index:
                secondary_index[os.path.basename(ann_path)] = {}
            voc_dict = parse_voc_xml(ET.parse(ann_path).getroot())
            for i, obj in enumerate(voc_dict['annotation']['object']):
                if obj['name'] == category:
                    secondary_index[os.path.basename(ann_path)][i] = ann_id  # 创建二级索引指向标签索引
                    obj['id'] = ann_id
                    w, h = int(obj['bndbox']['xmax'])-int(obj['bndbox']['xmin']), int(obj['bndbox']['ymax'])-int(obj['bndbox']['ymin'])
                    obj['ratio'] = min(w, h)/max(w, h)
                    ann_id += 1
                    cwise_anns.append(obj)

        ratios = np.asarray([ann['ratio'] for ann in cwise_anns])
        ann_ids = np.asarray([ann['id'] for ann in cwise_anns])
        # 子类别
        assert len(ratios) >= 2,'{:s} 类别的样本数目少于2，请检查数据集'.format(category)
        intra_classes = KMeans(n_clusters=2).fit_predict(ratios.reshape(-1, 1))
        generated_classes = intra_classes + num_children[:cid].sum() + 1  # intra class indices start from 1
        # 保存每个标签对应的子类别
        ann_cat_map.update({k: v for k, v in zip(ann_ids, generated_classes)})

        category_ids[category] = cid+1

    return {'convert_array': ann_cat_map, 'num_intra_classes': num_children, 'secondary_index': secondary_index, 'category_ids': category_ids}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset_dir', type=str, default='data/COCO2017', help='path to data directory')
    parser.add_argument('--dataset_name', type=str, default='coco2017', help='default: {:s}'.format('coco2017'))
    args = parser.parse_args()

    if args.dataset_name == 'coco2017':
        save_dict = coco_class_process(args)

    elif 'VOC' in args.dataset_name.upper():
        # voc_det = datasets.VOCDetection(
        #     args.dataset_dir, year='2007', image_set='train'
        # )
        save_dict = voc_class_process(args)

    torch.save(save_dict, os.path.join(args.dataset_dir, 'intra_classes.pth'))
