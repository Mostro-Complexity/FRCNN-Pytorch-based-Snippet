import utils
import transforms
import torch
import torchvision
import os
import argparse
from engine import train_one_epoch
# from engine import train_one_epoch, evaluate
from torchvision.ops import misc as misc_nn_ops
from torchvision.models import resnet
from torchvision.models.detection.backbone_utils import BackboneWithFPN
from torchvision.models.detection.faster_rcnn import FasterRCNN
from dataset import MarkDataset


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


def get_transform(train):
    transformers = []
    # converts the image, a PIL image, into a PyTorch Tensor
    transformers.append(transforms.ToTensor())
    if train:
        # during training, randomly flip the training images
        # and ground-truth for data augmentation
        # 50%的概率水平翻转
        transformers.append(transforms.RandomHorizontalFlip(0.5))

    return transforms.Compose(transformers)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--backbone_path', type=str, default='./backbone/resnext101_32x8d-8ba56ff5.pth', help='name of backbone model')
    parser.add_argument('-d', '--dataset_dir', type=str, default='./croped_medicine_data', help='path to data directory')
    parser.add_argument('-c', '--checkpoints_dir', type=str, default='./checkpoint', help='path to outputs directory')
    parser.add_argument('--image_min_side', type=int, default=600, help='default: {:d}'.format(600))
    parser.add_argument('--image_max_side', type=int, default=1000, help='default: {:d}'.format(1000))
    parser.add_argument('--batch_size', type=int, default=8, help='default: {:g}'.format(8))
    parser.add_argument('--learning_rate', type=float, default=0.003, help='default: {:g}'.format(0.003))
    parser.add_argument('--momentum', type=float, default=0.9, help='default: {:g}'.format(0.9))
    parser.add_argument('--weight_decay', type=float, default=0.0005, help='default: {:g}'.format(0.0005))
    parser.add_argument('--num_steps_to_display', type=int, default=20, help='default: {:d}'.format(20))
    parser.add_argument('--num_epochs_to_snapshot', type=int, default=1, help='default: {:d}'.format(1))
    parser.add_argument('--epochs', type=int, default=100, help='default: {:d}'.format(100))
    parser.add_argument('--current_epoch', type=int, default=1, help='default: {:d}'.format(1))
    parser.add_argument('--workers', type=int, default=4, help='default: {:d}'.format(4))
    args = parser.parse_args()

    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # use our dataset and defined transformations
    dataset = MarkDataset(args.dataset_dir, get_transform(train=True))
    dataset_test = MarkDataset(args.dataset_dir, get_transform(train=False))

    # including background
    num_classes = len(dataset.labels_cvtmap)

    # split the dataset in train and test set
    # 我的数据集一共有492张图，差不多训练验证4:1
    # indices = torch.randperm(len(dataset)).tolist()
    # dataset = torch.utils.data.Subset(dataset, indices[:-100])
    # dataset_test = torch.utils.data.Subset(dataset_test, indices[-100:])

    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
        collate_fn=utils.collate_fn)

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
    model = FasterRCNN(backbone, num_classes, min_size=args.image_min_side, max_size=args.image_max_side)
    # move model to the right device
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]

    # SGD
    optimizer = torch.optim.SGD(params, lr=args.learning_rate,
                                momentum=args.momentum, weight_decay=args.weight_decay)

    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=1, T_mult=2)

    checkpoint_path = os.path.join(args.checkpoints_dir, 'checkpoint-epoch{:d}.pth'.format(args.current_epoch-1))
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        optimizer.load_state_dict(checkpoint['optimizer'])
        model.load_state_dict(checkpoint['state_dict'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        print('checkpoint loaded.')
    else:
        print('{:s} does not exist.'.format(checkpoint_path))

    for epoch in range(args.current_epoch, args.epochs+1):
        model.train()
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=50)

        # update the learning rate
        lr_scheduler.step()

        # evaluate on the test dataset
        # evaluate(model, data_loader_test, device=device)

        if epoch % args.num_epochs_to_snapshot == 0:
            checkpoint = {
                'epoch': epoch,
                'optimizer': optimizer.state_dict(),
                'state_dict': model.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict()
            }
            torch.save(checkpoint, "checkpoint/checkpoint-epoch{:d}.pth".format(epoch))

    print("That's it!")
