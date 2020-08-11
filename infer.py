from utils import showbbox
from PIL import Image, ImageDraw
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
import argparse
from torchvision.transforms import functional as TF
import numpy as np


class Detector(object):
    def __init__(self, model_path, num_classes):
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.num_classes = num_classes
        self.checkpoint = torch.load(model_path)

        self.model = fasterrcnn_resnet50_fpn(
            pretrained=False,
            progress=True,
            num_classes=self.num_classes,
            pretrained_backbone=False
        )  # 或get_object_detection_model(num_classes)
        self.model.load_state_dict(self.checkpoint['state_dict'])
        self.model.to(self.device)

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

    def from_global_image(self, global_image, sub_region, nms_thres=0.9):
        image = global_image.crop(sub_region)
        scores, boxes, labels = self.detect(image, nms_thres)
        draw = ImageDraw.Draw(image)

        for box in boxes:
            xmin = round(box[0].item())
            ymin = round(box[1].item())
            xmax = round(box[2].item())
            ymax = round(box[3].item())

            draw.rectangle(((xmin, ymin), (xmax, ymax)), outline='blue')

        global_image.paste(image, sub_region)
        return global_image


def _infer_from_image():
    num_classes = 12

    detector = Detector(r'checkpoint/checkpoint-epoch23.pth', num_classes)

    global_image = Image.open('medicine_data/JPEGImages/2_gaussian.jpg').convert("RGB")
    global_image = detector.from_global_image(global_image, [800, 150, 1300, 920])
    global_image.show()


def _infer_from_video():
    import cv2
    num_classes = 12
    detector = Detector(r'checkpoint/checkpoint-epoch23.pth', num_classes)
    cap = cv2.VideoCapture('medicine_data/Videos/震元药店收银台_震元药店收银台_20200612094650_20200612123733_.avi')
    while(cap.isOpened()):
        _, frame = cap.read()
        global_image = Image.fromarray(frame).convert("RGB")
        global_image = detector.from_global_image(global_image, [800, 150, 1300, 920])
        global_image = global_image.resize(tuple(int(i*0.8) for i in global_image.size))
        cv2.imshow('image', np.array(global_image))
        k = cv2.waitKey(20)
        # q键退出
        if k & 0xff == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # _infer_from_image()
    _infer_from_video()
