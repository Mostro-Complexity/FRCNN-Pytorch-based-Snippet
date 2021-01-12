import transforms as T
import torch
import torch.utils.tensorboard
import os
import argparse
import torchvision.datasets as datasets
import time
import utils
import json
from pycocotools.cocoeval import COCOeval
from torchvision.ops import misc as misc_nn_ops
from torchvision.models import resnet
from torchvision.models.detection.backbone_utils import BackboneWithFPN
from torchvision.models.detection.faster_rcnn import FasterRCNN
from transforms import ClassConvert


@torch.no_grad()
def evaluate(model, data_loader, device, num_intra_classes=None):

    # n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    # torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    cat_ids = data_loader.dataset.coco.getCatIds()

    results = []
    for images, targets in metric_logger.log_every(data_loader, 100, header):
        images = list(img.to(device) for img in images)

        torch.cuda.synchronize()
        model_time = time.time()
        outputs = model(images)

        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        model_time = time.time() - model_time

        for output, target in zip(outputs, targets):
            if len(target) == 0:
                continue
            output = ClassConvert.reverse(output, num_intra_classes, cat_ids)
            results.extend(generate_outputs(
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

    show_pr_curves(cocoEval)

# def evaluate(self, image_ids: List[str], bboxes: List[List[float]], classes: List[int], probs: List[float]) -> Tuple[float, str]:
#     generate_ground_truth(, image_ids, bboxes, classes, probs)

#     annType = 'bbox'
#     path_to_coco_dir = os.path.join(self._path_to_data_dir, 'COCO')
#     path_to_annotations_dir = os.path.join(path_to_coco_dir, 'annotations')
#     path_to_annotation = os.path.join(path_to_annotations_dir, 'instances_val2017.json')

#     cocoDt = generate_ground_truth()

#     cocoEval = COCOeval(cocoGt, cocoDt, annType)
#     cocoEval.evaluate()
#     cocoEval.accumulate()

#     original_stdout = sys.stdout
#     string_stdout = StringIO()
#     sys.stdout = string_stdout
#     cocoEval.summarize()
#     sys.stdout = original_stdout

#     mean_ap = cocoEval.stats[0].item()  # stats[0] records AP@[0.5:0.95]
#     detail = string_stdout.getvalue()

#     self._show_pr_curves(cocoEval)

#     return mean_ap, detail


def show_pr_curves(coco_eval):
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


def generate_outputs(image_id, bboxes, classes, probs):
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


def resnet_fpn_backbone(backbone_name, pretrained, backbone_path=None):
    if backbone_path is not None and os.path.exists(backbone_path) and pretrained:
        backbone = resnet.__dict__[backbone_name](
            pretrained=False,
            norm_layer=misc_nn_ops.FrozenBatchNorm2d)
        backbone.load_state_dict(torch.load(backbone_path))
    else:
        backbone = resnet.__dict__[backbone_name](
            pretrained=pretrained,
            norm_layer=misc_nn_ops.FrozenBatchNorm2d)

    # freeze layers
    for name, parameter in backbone.named_parameters():
        if 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
            parameter.requires_grad_(False)

    return_layers = {'layer1': '0', 'layer2': '1', 'layer3': '2', 'layer4': '3'}

    in_channels_stage2 = backbone.inplanes // 8
    in_channels_list = [
        in_channels_stage2,
        in_channels_stage2 * 2,
        in_channels_stage2 * 4,
        in_channels_stage2 * 8,
    ]
    out_channels = 256
    return BackboneWithFPN(backbone, return_layers, in_channels_list, out_channels)


def load_classes(path):
    info_dict = torch.load(path)
    instance_convert_map = info_dict['convert_array']
    num_intra_classes = info_dict['num_intra_classes']
    return instance_convert_map, num_intra_classes


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
    args = parser.parse_args()

    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # define coco dataset
    coco_det = datasets.CocoDetection(
        os.path.join(args.dataset_dir, 'val2017'),
        os.path.join(args.dataset_dir, 'annotations', 'instances_val2017.json'),
        transforms=T.Compose([T.ToTensor()])
    )

    # load classes (including background)
    if args.intra_class:
        convert_map, num_intra_classes = load_classes(os.path.join(args.dataset_dir, 'intra_classes.pth'))
        num_categories = num_intra_classes.sum().item() + 1
    else:
        convert_map, num_intra_classes = coco_det.coco.getCatIds(), None
        num_categories = len(convert_map) + 1

    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        coco_det, batch_size=args.batch_size, num_workers=args.workers, shuffle=False,
        collate_fn=collate_fn_coco)

    # data_loader_test = torch.utils.data.DataLoader(
    #     dataset_test, batch_size=2, shuffle=False,  # num_workers=4,
    #     collate_fn=utils.collate_fn)

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
    model = FasterRCNN(backbone, num_categories, min_size=args.image_min_side, max_size=args.image_max_side)
    # move model to the right device
    model.to(device)

    checkpoint_path = os.path.join(args.model_path)
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['state_dict'])
        print('checkpoint loaded.')
    else:
        print('{:s} does not exist.'.format(checkpoint_path))

    model.eval()
    # evaluate on the test dataset
    evaluate(model, data_loader, num_intra_classes=num_intra_classes, device=device)
