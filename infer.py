from utils import showbbox
from PIL import Image
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FasterRCNN
from train import resnet_fpn_backbone
if __name__ == "__main__":
    num_classes = 2
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    checkpoint = torch.load(r'checkpoint/checkpoint-epoch50.pth')
    model = fasterrcnn_resnet50_fpn(
        pretrained=False,
        progress=True,
        num_classes=num_classes,
        pretrained_backbone=False
    )  # 或get_object_detection_model(num_classes)
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)

    image = Image.open('medicine_data/JPEGImages/2_gaussian.jpg').convert("RGB")
    showbbox(model, image, device)
