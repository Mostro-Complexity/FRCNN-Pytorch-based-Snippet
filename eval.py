import transforms as T
import torch
import torch.utils.tensorboard
import os
import argparse
import torchvision.datasets as datasets
import time
import utils
import json
from torchvision.models.detection.faster_rcnn import FasterRCNN
from transforms import ClassConvert
from train import load_classes, resnet_fpn_backbone


@torch.no_grad()
def evaluate_on_coco(model, data_loader, device, args):
    from pycocotools.cocoeval import COCOeval

    # n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    # torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")

    results = []
    for images, targets in metric_logger.log_every(data_loader, 100, 'Test:'):
        images = list(img.to(device) for img in images)

        torch.cuda.synchronize()
        model_time = time.time()
        outputs = model(images)

        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        model_time = time.time() - model_time

        for output, target in zip(outputs, targets):
            if len(target) == 0:
                continue
            output = ClassConvert.reverse(output, args.num_intra_classes, args.category_ids)
            results.extend(generate_coco_predictions(
                target[0]['image_id'],  # has to be a scalar
                output['boxes'].tolist(),
                output['labels'].tolist(),
                output['scores'].tolist()
            ))
        metric_logger.update(model_time=model_time)

    json.dump(results, open('results.json', 'w'))
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    coco_gt = data_loader.dataset.coco
    coco_dt = coco_gt.loadRes('results.json')
    cocoEval = COCOeval(coco_gt, coco_dt, 'bbox')
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

    show_coco_pr_curves(cocoEval)


@torch.no_grad()
def evaluate_on_voc(model, data_loader, device, args):
    # n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    # torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")

    cwise_predictions = [[] for _ in range(len(args.category_ids))]

    for images, targets in metric_logger.log_every(data_loader, 100, 'Test:'):
        images = list(img.to(device) for img in images)

        torch.cuda.synchronize()
        model_time = time.time()
        outputs = model(images)

        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        model_time = time.time() - model_time

        for output, target in zip(outputs, targets):
            if len(target) == 0:
                continue
            
            if args.num_intra_classes is not None:
                output = ClassConvert.reverse(output, args.num_intra_classes)
            generate_voc_predictions(
                cwise_predictions,
                target['annotation']['filename'][:6],
                output['boxes'].tolist(),
                output['labels'].tolist(),
                output['scores'].tolist()
            )
        metric_logger.update(model_time=model_time)

    os.makedirs('voc_gt', exist_ok=True)
    for cid, predictions in enumerate(cwise_predictions):
        with open('voc_gt/{:s}.txt'.format(args.category_ids[cid]), 'w') as f:
            f.writelines(['{:s} {:f} {:f} {:f} {:f} {:f}\n'.format(*p) for p in predictions])
            f.close()
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    mAP = []
    # 计算每个类别的AP
    for cid in range(len(args.category_ids)):
        class_name = args.category_ids[cid] 
        rec, prec, ap = utils.voc_eval(
            'voc_gt/{}.txt',
            'data/VOC/VOCdevkit/VOC2007/Annotations/{}.xml',
            'data/VOC/VOCdevkit/VOC2007/ImageSets/Main/val.txt',
            class_name, './'
        )
        print("{} :\t {} ".format(class_name, ap))
        mAP.append(ap)

    # 输出总的mAP
    print("mAP :\t {}".format(float(sum(mAP)/len(mAP))))


def show_coco_pr_curves(coco_eval):
    import matplotlib.pyplot as plt
    import numpy as np
    pr_array1 = coco_eval.eval['precision'][0, :, 0, 0, 2]
    pr_array2 = coco_eval.eval['precision'][2, :, 0, 0, 2]
    pr_array3 = coco_eval.eval['precision'][4, :, 0, 0, 2]
    x = np.arange(0, 1.01, 0.01)
    plt.xlabel('recall')
    plt.ylabel('precision')
    plt.xlim(0, 1.0)
    plt.ylim(0, 1.01)
    plt.grid()

    plt.plot(x, pr_array1, 'b-', label='IoU=0.5')
    plt.plot(x, pr_array2, 'c-', label='IoU=0.6')
    plt.plot(x, pr_array3, 'y-', label='IoU=0.7')

    plt.legend(loc='lower left')
    plt.show()


def generate_voc_predictions(cwise_predictions, image_id, bboxes, classes, probs):
    for bbox, cls, prob in zip(bboxes, classes, probs):
        line_data = [image_id, prob]
        line_data.extend(bbox)
        cwise_predictions[cls-1].append(line_data)


def generate_coco_predictions(image_id, bboxes, classes, probs):
    results = [
        {
            'image_id': int(image_id),  # COCO evaluation requires `image_id` to be type `int`
            'category_id': cls,
            'bbox': [   # format [left, top, width, height] is expected
                bbox[0],
                bbox[1],
                bbox[2] - bbox[0],
                bbox[3] - bbox[1]
            ],
            'score': prob
        }
        for bbox, cls, prob in zip(bboxes, classes, probs)
    ]

    return results


def load_coco_data(args):
    # define coco dataset
    dataset = datasets.CocoDetection(
        os.path.join(args.dataset_dir, 'val2017'),
        os.path.join(args.dataset_dir, 'annotations', 'instances_val2017.json'),
        transforms=T.Compose([T.ToTensor()])
    )

    # load classes (including background)
    if args.intra_class:
        convert_map, args.num_intra_classes = load_classes(os.path.join(args.dataset_dir, 'intra_classes.pth'))
        args.num_categories = args.num_intra_classes.sum().item() + 1
    else:
        convert_map, args.num_intra_classes = dataset.coco.getCatIds(), None
        args.num_categories = len(convert_map) + 1

    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, num_workers=args.workers, shuffle=False,
        collate_fn=collate_fn_coco)

    args.category_ids = data_loader.dataset.coco.getCatIds()

    return data_loader


def load_voc_data(args):
    # define voc dataset
    dataset = datasets.VOCDetection(
        args.dataset_dir,
        year='2007',
        image_set='val',
        transforms=T.Compose([T.ToTensor()])
    )

    # load classes (including background)
    if args.intra_class:
        convert_map, args.num_intra_classes, _, category_ids = load_classes(os.path.join(args.dataset_dir, 'intra_classes.pth'))
        args.num_categories = sum(args.num_intra_classes) + 1
    else:
        _, _, _, category_ids = load_classes(os.path.join(args.dataset_dir, 'intra_classes.pth'))
        args.num_intra_classes = None
        args.num_categories = len(category_ids) + 1

    args.category_ids = list(category_ids.keys())

    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, num_workers=args.workers, shuffle=False,
        collate_fn=collate_fn_coco)

    return data_loader


def collate_fn_coco(batch):
    return tuple(zip(*batch))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--backbone_path', type=str, default='./backbone/resnext101_32x8d-8ba56ff5.pth', help='name of backbone model')
    parser.add_argument('-d', '--dataset_dir', type=str, default='./data', help='path to data directory')
    parser.add_argument('-m', '--model_path', type=str, default='./checkpoint', help='path to outputs directory')
    parser.add_argument('--image_min_side', type=int, default=600, help='default: {:d}'.format(600))
    parser.add_argument('--image_max_side', type=int, default=1000, help='default: {:d}'.format(1000))
    parser.add_argument('--batch_size', type=int, default=8, help='default: {:g}'.format(8))
    parser.add_argument('--num_steps_to_display', type=int, default=20, help='default: {:d}'.format(20))
    parser.add_argument('--workers', type=int, default=4, help='default: {:d}'.format(4))
    parser.add_argument('--intra_class', action='store_true')
    parser.add_argument('--dataset_name', type=str, default='coco2017', help='default: {:s}'.format('coco2017'))
    args = parser.parse_args()

    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    if args.dataset_name == 'coco2017':
        data_loader = load_coco_data(args)
    elif 'VOC' in args.dataset_name.upper():
        data_loader = load_voc_data(args)

    # get the model using our helper function
    backbone_name = os.path.basename(args.backbone_path).split('-')[0]
    if os.path.exists(args.backbone_path):
        backbone = resnet_fpn_backbone(
            backbone_name,
            pretrained=True,
            backbone_path=args.backbone_path
        )
    else:
        backbone = resnet_fpn_backbone(
            backbone_name,
            pretrained=False
        )
    # number of categories includes background
    model = FasterRCNN(backbone, args.num_categories, min_size=args.image_min_side, max_size=args.image_max_side)
    # move model to the right device
    model.to(device)

    checkpoint_path = os.path.join(args.model_path)
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['state_dict'])
        print('checkpoint loaded.')
    else:
        print('{:s} does not exist.'.format(checkpoint_path))

    # evaluate on the test dataset
    if args.dataset_name == 'coco2017':
        evaluate_on_coco(model, data_loader, device, args)
    elif 'VOC' in args.dataset_name.upper():
        evaluate_on_voc(model, data_loader, device, args)
