import utils
import transforms
import torch
import torchvision
import os
from engine import train_one_epoch
# from engine import train_one_epoch, evaluate
from torchvision.ops import misc as misc_nn_ops
from torchvision.models import resnet
from torchvision.models.detection.backbone_utils import BackboneWithFPN
from torchvision.models.detection.faster_rcnn import FasterRCNN
from dataset import MarkDataset


def resnet_fpn_backbone(backbone_name, pretrained, backbone_path):
    if os.path.exists(backbone_path) and pretrained:
        backbone = resnet.__dict__[backbone_name](
            pretrained=False,
            norm_layer=misc_nn_ops.FrozenBatchNorm2d)
        backbone.load_state_dict(torch.load(backbone_path))
    elif pretrained:
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


def get_transform(train):
    transformers = []
    # converts the image, a PIL image, into a PyTorch Tensor
    transformers.append(transforms.ToTensor())
    if train:
        # during training, randomly flip the training images
        # and ground-truth for data augmentation
        # 50%的概率水平翻转
        transformers.append(transforms.RandomHorizontalFlip(0.5))
        transformers.append(transforms.Crop([800, 150, 1300, 920]))

    return transforms.Compose(transformers)


if __name__ == "__main__":
    root = r'medicine_data'

    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # use our dataset and defined transformations
    dataset = MarkDataset(root, get_transform(train=True))
    dataset_test = MarkDataset(root, get_transform(train=False))

    # including background
    num_classes = len(dataset.labels_cvtmap)

    # split the dataset in train and test set
    # 我的数据集一共有492张图，差不多训练验证4:1
    # indices = torch.randperm(len(dataset)).tolist()
    # dataset = torch.utils.data.Subset(dataset, indices[:-100])
    # dataset_test = torch.utils.data.Subset(dataset_test, indices[-100:])

    # define training and validation data loaders
    # 在jupyter notebook里训练模型时num_workers参数只能为0，不然会报错，这里就把它注释掉了
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=4, shuffle=True, num_workers=0,
        collate_fn=utils.collate_fn)

    # data_loader_test = torch.utils.data.DataLoader(
    #     dataset_test, batch_size=2, shuffle=False,  # num_workers=4,
    #     collate_fn=utils.collate_fn)

    # get the model using our helper function
    backbone_path = 'backbone/resnet50-19c8e357.pth'
    if os.path.exists(backbone_path):
        backbone = resnet_fpn_backbone(
            'resnet50',
            pretrained=True,
            backbone_path=backbone_path
        )
        model = FasterRCNN(backbone, num_classes)
    else:
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
            pretrained=False,
            progress=True,
            num_classes=num_classes,
            pretrained_backbone=True
        )  # 或get_object_detection_model(num_classes)
    # move model to the right device
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]

    # SGD
    optimizer = torch.optim.SGD(params, lr=0.003,
                                momentum=0.9, weight_decay=0.0005)

    # and a learning rate scheduler
    # cos学习率
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=1, T_mult=2)

    # let's train it for   epochs
    num_epochs = 100

    for epoch in range(num_epochs):
        model.train()
        # train for one epoch, printing every 10 iterations
        # engine.py的train_one_epoch函数将images和targets都.to(device)了
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=50)

        # update the learning rate
        lr_scheduler.step()

        # evaluate on the test dataset
        # evaluate(model, data_loader_test, device=device)

        checkpoint = {
            'epoch': epoch,
            'optimizer': optimizer,
            'state_dict': model.state_dict()
        }

        torch.save(checkpoint, "checkpoint/checkpoint-epoch{:d}.pth".format(epoch))
        print('')
        print('==================================================')
        print('')

    print("That's it!")
