import torchvision.datasets as datasets
import torch
from sklearn.cluster import KMeans

coco_det = datasets.CocoDetection(
    'data/train2017',
    'data/annotations/instances_train2017.json')

num_classes = len(coco_det.coco.getCatIds())
num_children = torch.as_tensor([2]*num_classes)

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
    intra_classes = KMeans(n_clusters=2).fit_predict(ratio.view(-1, 1).numpy())
    generated_classes = intra_classes + num_children[:idx].sum().item() + 1  # intra class indices start from 1

    ann_cat_map.update({k: v for k, v in zip(anns_ids, generated_classes)})
    # ann_cat_map[anns_ids] = torch.as_tensor(generated_classes, dtype=torch.int64)

torch.save({'convert_array': ann_cat_map, 'num_intra_classes': num_children}, 'data/intra_classes.pth')
