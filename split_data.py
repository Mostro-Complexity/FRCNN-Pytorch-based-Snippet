import os
import glob
import numpy as np
import xml.etree.ElementTree as ET
import collections
from dict2xml import dict2xml


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


def save_voc_xml(dict_in, xml_out):
    xml = dict2xml(dict_in, indent="  ")
    with open(xml_out, 'w', encoding='utf-8') as f:
        f.write(xml)


def sample_append(cat_dict, key, value):
    if key not in cat_dict:
        cat_dict[key] = {value}
    else:
        cat_dict[key].add(value)


def write_lines(path, array):
    with open(path, 'w', encoding='utf-8') as f:
        for line in sorted(list(array)):
            if isinstance(line, list) or isinstance(line, tuple):
                line = ' '.join([str(e) for e in line])
            f.write('{:s}\n'.format(line))


if __name__ == "__main__":
    anno_dir = 'data/VOC/VOCdevkit/VOC2007/Annotations'
    anno_paths = glob.glob(os.path.join(anno_dir, '*.xml'))
    anno_names = [os.path.splitext(os.path.basename(path))[0] for path in anno_paths]
    output_paths = 'data/VOC/VOCdevkit/VOC2007/ImageSets/Main'
    train_val_ratio = 0.7

    os.makedirs(output_paths, exist_ok=True)
    categories_splits, train_set, val_set = {}, set(), set()

    # 按类别整合并修改图片名称
    for name, path in zip(anno_names, anno_paths):
        voc_dict = parse_voc_xml(ET.parse(path).getroot())
        categories = np.asarray([obj['name'] for obj in voc_dict['annotation']['object']])
        appendix = os.path.splitext(voc_dict['annotation']['filename'])[1]
        voc_dict['annotation']['filename'] = name + appendix
        save_voc_xml(voc_dict, path)
        for c in categories:
            sample_append(categories_splits, c, name)

    # 划分
    for cat, names in categories_splits.items():
        names = np.asarray(list(names))
        n_train = int(len(names)*train_val_ratio)

        done_split_train = np.isin(names, list(train_set))
        done_split_val = np.isin(names, list(val_set))

        n_train -= done_split_train.sum()
        names_remain = names[np.logical_and(~done_split_train, ~done_split_val)]
        train_set.update(names_remain[:n_train])
        val_set.update(names_remain[n_train:])

    write_lines(os.path.join(output_paths, 'train.txt'), train_set)
    write_lines(os.path.join(output_paths, 'val.txt'), val_set)

    for cat, names in categories_splits.items():
        flags = ~np.isin(list(names), list(train_set))
        flags = -flags.astype(np.int8)
        write_lines(os.path.join(output_paths, '{:s}_train.txt'.format(cat)), zip(names, flags))

        flags = ~np.isin(list(names), list(val_set))
        flags = -flags.astype(np.int8)
        write_lines(os.path.join(output_paths, '{:s}_val.txt'.format(cat)), zip(names, flags))
