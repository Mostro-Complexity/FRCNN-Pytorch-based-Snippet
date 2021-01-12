import transforms as T
import torch
import torch.utils.tensorboard
import os
import argparse
import torchvision.datasets as datasets
import glob
from engine import train_one_epoch
from lr_scheduler import WarmUpMultiStepLR
from torchvision.ops import misc as misc_nn_ops
from torchvision.models import resnet
from torchvision.models.detection.backbone_utils import BackboneWithFPN
from torchvision.models.detection.faster_rcnn import FasterRCNN


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
    image, target = tuple(zip(*batch))
    for t in target:
        t['boxes'] = t['bbox'] if 'bbox' in t else torch.empty(0, 4)
        t['labels'] = t['category_id'] if 'category_id' in t else torch.empty(0, dtype=torch.int64)
    return image, target


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--backbone_path', type=str, default='./backbone/resnext101_32x8d-8ba56ff5.pth', help='name of backbone model')
    parser.add_argument('-d', '--dataset_dir', type=str, default='./data', help='path to data directory')
    parser.add_argument('-c', '--checkpoints_dir', type=str, default='./checkpoint', help='path to outputs directory')
    parser.add_argument('--image_min_side', type=int, default=600, help='default: {:d}'.format(600))
    parser.add_argument('--image_max_side', type=int, default=1000, help='default: {:d}'.format(1000))
    parser.add_argument('--batch_size', type=int, default=8, help='default: {:g}'.format(8))
    parser.add_argument('--learning_rate', type=float, default=0.003, help='default: {:g}'.format(0.003))
    parser.add_argument('--momentum', type=float, default=0.9, help='default: {:g}'.format(0.9))
    parser.add_argument('--weight_decay', type=float, default=0.0005, help='default: {:g}'.format(0.0005))
    parser.add_argument('--num_steps_to_display', type=int, default=20, help='default: {:d}'.format(20))
    parser.add_argument('--num_epochs_to_snapshot', type=int, default=1, help='default: {:d}'.format(1))
    parser.add_argument('--num_checkpoints_to_reserve', type=int, default=10, help='default: {:d}'.format(10))
    parser.add_argument('--epochs', type=int, default=100, help='default: {:d}'.format(100))
    parser.add_argument('--current_epoch', type=int, default=1, help='default: {:d}'.format(1))
    parser.add_argument('--workers', type=int, default=4, help='default: {:d}'.format(4))
    parser.add_argument('--intra_class', action='store_true')
    parser.add_argument('--dataset_name', type=str, default='coco2017', help='default: {:s}'.format('coco2017'))
    args = parser.parse_args()

    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # load classes (including background)
    if args.intra_class:
        instance_convert_map, num_intra_classes = load_classes(os.path.join(args.dataset_dir, 'intra_classes.pth'))
        class_convert = T.ClassConvert(instance_convert_map, num_intra_classes)  # number of categories includes background
        num_categories = num_intra_classes.sum().item() + 1
    elif args.dataset_name == 'coco2017':
        convert_map = [
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
            11, 13, 14, 15, 16, 17, 18, 19, 20,
            21, 22, 23, 24, 25, 27, 28, 31, 32,
            33, 34, 35, 36, 37, 38, 39, 40, 41,
            42, 43, 44, 46, 47, 48, 49, 50, 51,
            52, 53, 54, 55, 56, 57, 58, 59, 60,
            61, 62, 63, 64, 65, 67, 70, 72, 73,
            74, 75, 76, 77, 78, 79, 80, 81, 82,
            84, 85, 86, 87, 88, 89, 90
        ]
        convert_map = {c: i+1 for i, c in enumerate(convert_map)}
        class_convert = T.ClassConvert(convert_map)
        num_categories = len(convert_map) + 1

    # define coco dataset
    coco_det = datasets.CocoDetection(
        os.path.join(args.dataset_dir, 'train2017'),
        os.path.join(args.dataset_dir, 'annotations', 'instances_train2017.json'),
        transforms=T.Compose([
            T.AnnotationCollate(),
            T.BoxesFormatConvert(),
            class_convert,
            T.ToTensor(),
            T.RandomHorizontalFlip(0.5)
        ])
    )

    # define coco sampler
    sampler = torch.utils.data.RandomSampler(coco_det)
    batch_sampler = torch.utils.data.BatchSampler(sampler, args.batch_size, drop_last=True)

    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        coco_det, batch_sampler=batch_sampler, num_workers=args.workers,
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

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]

    # SGD
    optimizer = torch.optim.SGD(params, lr=args.learning_rate,
                                momentum=args.momentum, weight_decay=args.weight_decay)

    # and a learning rate scheduler
    lr_scheduler = WarmUpMultiStepLR(optimizer, milestones=[6, 8], gamma=0.1,
                                     factor=0.3333, num_iters=5)
    # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=1, T_mult=2)
    summary_writer = torch.utils.tensorboard.SummaryWriter('logs')

    checkpoint_path = os.path.join(args.checkpoints_dir, 'checkpoint-epoch{:04d}.pth'.format(args.current_epoch-1))
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
        train_one_epoch(
            model, optimizer, data_loader, device, epoch,
            print_freq=args.num_steps_to_display,
            summary_writer=summary_writer
        )

        # update the learning rate
        lr_scheduler.step()

        # evaluate on the test dataset
        # evaluate(model, data_loader_test, device=device)

        if epoch % args.num_epochs_to_snapshot == 0:
            os.makedirs('checkpoint', exist_ok=True)
            ckpts_path = glob.glob('checkpoint/checkpoint-epoch*.pth')
            if len(ckpts_path) > args.num_checkpoints_to_reserve:
                for path in ckpts_path[:len(ckpts_path)-args.num_checkpoints_to_reserve]:
                    os.remove(path)
            checkpoint = {
                'epoch': epoch,
                'optimizer': optimizer.state_dict(),
                'state_dict': model.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict()
            }
            torch.save(checkpoint, "checkpoint/checkpoint-epoch{:04d}.pth".format(epoch))

    print("That's it!")
