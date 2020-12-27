import random
import torch
from torch._C import dtype

from torchvision.transforms import functional as F


def _flip_coco_person_keypoints(kps, width):
    flip_inds = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
    flipped_data = kps[:, flip_inds]
    flipped_data[..., 0] = width - flipped_data[..., 0]
    # Maintain COCO convention that if visibility == 0, then x, y = 0
    inds = flipped_data[..., 2] == 0
    flipped_data[inds] = 0
    return flipped_data


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class RandomHorizontalFlip(object):
    def __init__(self, prob):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            height, width = image.shape[-2:]
            image = image.flip(-1)
            if "bbox" in target:
                bbox = target["bbox"]
                # assert torch.all(bbox[:, 2:] > bbox[:, :2])
                bbox[:, [0, 2]] = width - bbox[:, [2, 0]]
                target["bbox"] = bbox

            if "masks" in target:
                target["masks"] = target["masks"].flip(-1)
            if "keypoints" in target:
                keypoints = target["keypoints"]
                keypoints = _flip_coco_person_keypoints(keypoints, width)
                target["keypoints"] = keypoints
        return image, target


class ToTensor(object):
    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target


class Crop(object):
    def __init__(self, bound):
        self.bound = bound  # xmin, ymin, xmax, ymax

    def __call__(self, image, target):
        boxes = target['boxes']  # xmin, ymin, xmax, ymax
        boxes[:, 0] = torch.clamp_min(boxes[:, 0], self.bound[0]) - self.bound[0]
        boxes[:, 1] = torch.clamp_min(boxes[:, 1], self.bound[1]) - self.bound[1]
        boxes[:, 2] = torch.clamp(boxes[:, 2], self.bound[0], self.bound[2]) - self.bound[0]
        boxes[:, 3] = torch.clamp(boxes[:, 3], self.bound[1], self.bound[3]) - self.bound[1]
        # boxes += 1

        target['area'] = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        image = image[:, self.bound[1]:self.bound[3], self.bound[0]:self.bound[2]]
        assert torch.all(target['area'] > 0)
        return image, target


class AnnotationCollate(object):
    def __call__(self, image, target):
        collated_target = {}
        if len(target) != 0:
            keys = target[0].keys()
            for key in keys:
                _list = [t[key] for t in target if not t['iscrowd']]
                assert len(_list) != 0

                try:
                    collated_target[key] = torch.as_tensor(_list)
                except ValueError:
                    collated_target[key] = _list
                except Exception as e:
                    raise e

        return image, collated_target


class BoxesFormatConvert(object):
    def __call__(self, image, target):
        try:
            bbox = target["bbox"]
            bbox = bbox[torch.all(bbox[:, 2:] > 1e-6, dim=-1)]  # filter illegal labels
            bbox[:, 2:] += bbox[:, :2]
            target["bbox"] = bbox
        except KeyError:
            pass

        return image, target


class ClassConvert(object):
    def __init__(self, cvt_map, num_intra_list):
        self.cvt_map = cvt_map
        self.num_intra_list = num_intra_list

    def __call__(self, image, target):
        try:
            category_id = [self.cvt_map[i.item()] for i in target['id']]
            target['category_id'] = torch.as_tensor(category_id, dtype=torch.int64)
        except KeyError:
            pass
        except Exception as e:
            raise e
        return image, target

    def reverse(self):
        pass
