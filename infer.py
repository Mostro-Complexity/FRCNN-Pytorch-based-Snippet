import argparse
import os

import numpy as np
import torch
import glob
from PIL import Image, ImageDraw
from torchvision.models.detection.faster_rcnn import FasterRCNN
from torchvision.transforms import functional as TF
import torchvision.datasets as datasets

from train import resnet_fpn_backbone, load_classes
from transforms import ClassConvert
import json


class Detector(object):
    def __init__(self, args):
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.num_classes = args.num_classes
        self.checkpoint = torch.load(args.model_path, map_location=self.device)
        self.num_intra_classes = args.num_intra_classes
        backbone_name = os.path.basename(args.backbone_path).split('-')[0]
        backbone = resnet_fpn_backbone(
            backbone_name,
            pretrained=False
        )
        self.categories = args.categories
        self.model = FasterRCNN(backbone, args.num_classes + 1, min_size=args.image_min_side, max_size=args.image_max_side)

        # number of categories includes background
        self.model.load_state_dict(self.checkpoint['state_dict'])
        self.model.to(self.device)

        self._index = 1

    def detect(self, PIL_image, nms_thres=0.9):
        global_scores, global_boxes, global_labels = [], [], []
        self.model.eval()
        img_input = TF.to_tensor(PIL_image)
        with torch.no_grad():
            '''
            prediction形如：
            [{'boxes': tensor([[1492.6672,  238.4670, 1765.5385,  315.0320],
            [ 887.1390,  256.8106, 1154.6687,  330.2953]], device='cuda:0'),
            'labels': tensor([1, 1], device='cuda:0'),
            'scores': tensor([1.0000, 1.0000], device='cuda:0')}]
            '''
            prediction = self.model([img_input.to(self.device)])

            for pred in prediction:
                scores = pred['scores']
                boxes = pred['boxes']
                labels = pred['labels']
                selected = scores > nms_thres

                scores = scores[selected]
                boxes = boxes[selected]
                labels = labels[selected]

                global_scores.append(scores)
                global_boxes.append(boxes)
                global_labels.append(labels)

            global_scores = torch.cat(global_scores)
            global_boxes = torch.cat(global_boxes)
            global_labels = torch.cat(global_labels)

        return global_scores, global_boxes, global_labels

    def from_global_image(self, global_image, nms_thres=0.9):
        scores, boxes, labels = self.detect(global_image, nms_thres)
        draw = ImageDraw.Draw(global_image)

        for score, box, label in zip(scores, boxes, labels):
            xmin = round(box[0].item())
            ymin = round(box[1].item())
            xmax = round(box[2].item())
            ymax = round(box[3].item())
            category_id = ClassConvert.reverse(label.item(), self.num_intra_classes)
            category = self.categories[category_id-1]
            category_name = category['name']
            draw.rectangle(((xmin, ymin), (xmax, ymax)), outline='blue')
            draw.text((xmin, ymin), text=f'{category_name:s} {score.item():.3f}', fill='blue')

        return global_image

    def save_as_labelme(self, output_dir, global_image, sub_region, nms_thres=0.9):
        image = global_image.crop(sub_region)
        scores, boxes, labels = self.detect(image, nms_thres)

        image_path = os.path.join(output_dir, "{:d}_real.jpg".format(self._index))
        content = {
            "version": "4.5.6",
            "flags": {},
            "shapes": [],
            "imagePath": os.path.basename(image_path),
            "imageData": None,
            "imageHeight": image.height,
            "imageWidth": image.width
        }
        for i in range(labels.size(0)):
            box = boxes[i]
            label = labels[i].item()
            xmin = round(box[0].item())
            ymin = round(box[1].item())
            xmax = round(box[2].item())
            ymax = round(box[3].item())
            content["shapes"].append(
                {
                    "label": self.labels_cvt[label],
                    "points": [
                        [
                            xmin,
                            ymin
                        ],
                        [
                            xmax,
                            ymax
                        ]
                    ],
                    "group_id": None,
                    "shape_type": "rectangle",
                    "flags": {}
                }
            )
        json.dump(content, open(image_path.replace('.jpg', '.json'), 'w'))
        image.save(image_path)
        self._index += 1

    def load_labels(self, path):
        with open(path, 'r') as f:
            lines = f.readlines()
        self.labels_cvt = {int(l.split(' ')[0]): l.split(' ')[1].rstrip() for l in lines}


def _infer_from_image():
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--backbone_path', type=str, default='./backbone/resnext101_32x8d-8ba56ff5.pth', help='name of backbone model')
    parser.add_argument('-m', '--model_path', type=str, default='./checkpoint', help='path to outputs directory')
    parser.add_argument('-i', '--input_dir', type=str, required=True, help='inputs directory')
    parser.add_argument('-o', '--output_dir', type=str, required=True, help='outputs directory')
    parser.add_argument('-d', '--dataset_dir', type=str, default='./data', help='path to data directory')
    parser.add_argument('--image_min_side', type=int, default=600, help='default: {:d}'.format(600))
    parser.add_argument('--image_max_side', type=int, default=1000, help='default: {:d}'.format(1000))
    args = parser.parse_args()

    # load classes (including background)
    _, num_intra_classes = load_classes('data/intra_classes.pth')
    args.num_classes = num_intra_classes.sum().item()
    args.num_intra_classes = num_intra_classes

    # define coco dataset
    coco_det = datasets.CocoDetection(
        os.path.join(args.dataset_dir, 'train2017'),
        os.path.join(args.dataset_dir, 'annotations', 'instances_train2017.json')
    )
    args.categories = coco_det.coco.loadCats(coco_det.coco.getCatIds())

    detector = Detector(args)

    for path in glob.glob(os.path.join(args.input_dir, '*.jpg')):
        global_image = Image.open(path).convert("RGB")
        global_image = detector.from_global_image(global_image, nms_thres=0.8)
        global_image.save(os.path.join(args.output_dir, os.path.basename(path)))


def _infer_from_video():
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--backbone_path', type=str, default='./backbone/resnext101_32x8d-8ba56ff5.pth', help='name of backbone model')
    parser.add_argument('-d', '--dataset_dir', type=str, default='./medicine_data', help='path to data directory')
    parser.add_argument('-m', '--model_path', type=str, default='./checkpoint', help='path to outputs directory')
    parser.add_argument('-i', '--input_path', type=str, required=True, help='path to outputs directory')
    parser.add_argument('--image_min_side', type=int, default=600, help='default: {:d}'.format(600))
    parser.add_argument('--image_max_side', type=int, default=1000, help='default: {:d}'.format(1000))
    parser.add_argument('--num_classes', type=int, default=4, help='default: {:d}'.format(75))
    args = parser.parse_args()

    import cv2
    sub_region = [800, 150, 1300, 920]
    detector = Detector(args)
    cap = cv2.VideoCapture(args.input_path)
    while(cap.isOpened()):
        _, frame = cap.read()
        global_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        global_image = detector.from_global_image(global_image, sub_region, nms_thres=0.6)
        global_image = global_image.resize(tuple(int(i*0.8) for i in global_image.size))
        cv2.imshow('image', cv2.cvtColor(np.array(global_image), cv2.COLOR_BGR2RGB))
        k = cv2.waitKey(20)
        # q键退出
        if k & 0xff == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


def _regen_labeled_images():
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--backbone_path', type=str, default='./backbone/resnext101_32x8d-8ba56ff5.pth', help='name of backbone model')
    parser.add_argument('-d', '--dataset_dir', type=str, default='./medicine_data', help='path to data directory')
    parser.add_argument('-m', '--model_path', type=str, default='./checkpoint', help='path to outputs directory')
    parser.add_argument('-i', '--input_path', type=str, required=True, help='path to inputs')
    parser.add_argument('-o', '--output_dir', type=str, default='./regenerated_data', help='path to outputs directory')
    parser.add_argument('--image_min_side', type=int, default=600, help='default: {:d}'.format(600))
    parser.add_argument('--image_max_side', type=int, default=1000, help='default: {:d}'.format(1000))
    parser.add_argument('--num_classes', type=int, default=4, help='default: {:d}'.format(75))
    args = parser.parse_args()

    import cv2
    SUB_REGION = [800, 150, 1300, 920]
    SAMPLE_INTERVAL = 300
    detector = Detector(args)
    cap = cv2.VideoCapture(args.input_path)
    counter = 0
    while cap.isOpened():
        valid, frame = cap.read()
        if not valid:
            break
        if counter % SAMPLE_INTERVAL == 0:
            global_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            detector.save_as_labelme(args.output_dir, global_image, SUB_REGION, nms_thres=0.6)
        counter += 1

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    _infer_from_image()
    # _infer_from_video()
    # _regen_labeled_images()
