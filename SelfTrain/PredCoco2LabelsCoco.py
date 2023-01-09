import os
import json
import argparse
import numpy as np

from tqdm import tqdm
from pycocotools import mask

import annotations as an

def check_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)

def load_file(path):
    with open(path) as f:
        lines = f.readlines()
    data = []
    for l in lines:
        data += [json.loads(l)]
    return data


def build_mask(indices,anns,shape,threshold=0.3):
    resulting_mask = np.zeros(shape,dtype=np.uint8)
    count = 1
    categories = {0: 0}
    for i in indices:
        if anns[i]['score']>threshold:
            categories[count] = anns[i]['category_id']
            mask_decoded = mask.decode(anns[i]['segmentation'])
            resulting_mask = np.where(mask_decoded==1,count,resulting_mask)
            count += 1

    # print("N indices: ",len(indices))
    # print("Max value: ",resulting_mask.max())
    return resulting_mask,categories


def fusion_annotations(old_anns, path_save_anns, new_annotations, new_images_info):
    # for file in json_annotation['images']:
    #     shutil.copy(os.path.join(path_original_images, file['file_name']),
    #                 os.path.join(path_save_imgs, file['file_name']))

    # old_anns['images'].extend(new_images_info)
    # old_anns['annotations'].extend(new_annotations)
    json_annotation = old_anns.copy()
    # json_annotation['images'] = old_anns['images'].copy()
    json_annotation['images'] += new_images_info
    # json_annotation['annotations'] = old_anns['annotations'].copy()
    json_annotation['annotations'] += new_annotations
    print("Old vs new anns: {:d}/{:d} ".format(len(old_anns['annotations']), len(json_annotation['annotations'])))
    print("Old vs new imgs: {:d}/{:d} ".format(len(old_anns['images']), len(json_annotation['images'])))  # DELETE
    with open(path_save_anns, 'w+') as f:
        # this would place the entire output on one line
        # use json.dump(lista_items, f, indent=4) to "pretty-print" with four spaces per indent
        json.dump(json_annotation, f)


def get_bbox_annotations(anns_bbox:int,indices:int, image_id:int, ann_id:int, threshold = 0.3):
    annotations = []
    for i in indices:
        if anns_bbox[i]['score']>threshold:
            ann = {'image_id':image_id}
            bbox = anns_bbox[i]['bbox']
            ann['bbox'] = bbox
            ann['segmentation'] = []
            ann['iscrowd'] = 0
            ann['category_id'] = anns_bbox[i]['category_id']
            ann['id'] = ann_id
            ann['area'] = bbox[2]*bbox[3]

            ann_id += 1
            annotations += [ann]
    return annotations


def main(path_anns_st:str, path_anns_train:str, path_bb: str, path_segm: str, opt):
    """
    Function to create a new json file that combines predictions
    Parameters
    ----------
    path_anns_st: Path to json file with un-labeled images
    path_anns_train: Path to original json annotations
    path_bb: Path to BB predictions (.json)
    path_segm: Path to segments predictions (.json)
    opt

    Returns
    -------

    """
    threshold = opt.thrs
    shape_img = opt.shape
    # ext = opt.ext
    use_mask = opt.use_mask



    unlabeled_dataset = load_file(path_anns_st)[0]
    originals_anns = load_file(path_anns_train)[0]
    images = unlabeled_dataset['images']
    indices = [[] for _ in images]
    if use_mask:
        anns_segm = load_file(path_segm)[0]
        for i, ann in enumerate(anns_segm):
            indices[ann['image_id']] += [i]
    else:
        anns_bbox = load_file(path_bb)[0]
        for i, ann in enumerate(anns_bbox):
            indices[ann['image_id']] += [i]

    # n_predictions = len(anns_bbox)
    # print("Predictions and images: {:d}/{:d}".format(n_predictions, n_images))

    new_annotations = []
    new_images_info = []
    ann_id = len(originals_anns['annotations'])
    n_images = len(originals_anns['images'])

    shape = [images[0]['height'],images[0]['width']]
    img_dummy = np.empty(shape,dtype=np.uint8)

    for i, img_info in enumerate(tqdm(images)):
        img_name = img_info['file_name']
        if use_mask:
            mask_sample, categories = build_mask(indices[i], anns_segm, shape_img, threshold)
            new_anns = an.get_new_annotation(mask_sample, i + n_images, ann_id, categories)
            # ann_id = new_anns[-1]['id'] + 1
        else:
            new_anns = get_bbox_annotations(anns_bbox,indices[i], i + n_images, ann_id)
        ann_id += len(new_anns)
        # if i == 2:  # DELETE
        #     print("Found")
        # if len(new_anns)==0: # DELETE
        #     print("ERROR INVALID CONDITION: no background image detected for image {:d}".format(i))
        an.add_new_annotation(img_dummy, new_anns, i + n_images, img_name, new_images_info, new_annotations)
    path_save_new_anns = opt.path_save_anns
    fusion_annotations(originals_anns, path_save_anns=path_save_new_anns, new_annotations=new_annotations,
                       new_images_info=new_images_info)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='swin_predictions2labels.py',
                                     description=" Method to convert predictions into labels. Labels are in the  COCO"
                                                 "Format")
    parser.add_argument('--path-dataset', type=str, help='path of the dataset')
    parser.add_argument('--path-images', type=str, help='path to the images')
    parser.add_argument('--path-anns-train', default='/home/dataset/cherry_dataset/train_self/instances_train.json',
                        type=str, help='path to original annotations file.')
    parser.add_argument('--path-anns-st', default='/home/dataset/cherry_dataset/train_self/instances_self_train.json',
                        type=str, help='path to self train annotations (its supossed to be empty in annotations).')

    parser.add_argument('--path-bb', default='/home/dataset/cherry_dataset/train_self/predictions_swin_bb.jso',
                        type=str, help='path to BB json file.')
    parser.add_argument('--path-segm', default='/home/dataset/cherry_dataset/train_self/predictions_swin_segm.json',
                        type=str, help='path to segm. json file.')
    parser.add_argument('--thrs', default=0.3, type=float, help='Minimum confidence to be considered.')
    parser.add_argument('--shape', nargs='+', type=int, default=[1328, 1328],
                        help='shape image ')
    opt = parser.parse_args()
    np.random.seed(3993)

    # opt.path_dataset = '/home/luis/2021/COCO/cherry_dataset3/train_self'
    # opt.path_images = '/home/luis/2021/COCO/cherry_dataset3/train_self/images'
    # opt.path_anns_train = '/home/luis/2021/COCO/cherry_dataset3/train/instances_train.json'
    # opt.path_anns_st = '/home/luis/2021/COCO/cherry_dataset3/train_self/instances_test.json'
    # opt.path_save_anns = '/home/luis/2021/Cherry/instances_test_faster_ripeness.json'

    opt.path_dataset = '/home/luis/2021/COCO/cherry_dataset3/test'
    opt.path_images = '/home/luis/2021/COCO/cherry_dataset3/test/images'
    opt.path_anns_train = '/home/luis/2021/COCO/cherry_dataset3/train/instances_train.json'
    opt.path_anns_st = '/home/luis/2021/COCO/cherry_dataset3/train_self/instances_test_combined.json'
    opt.path_save_anns = '/home/luis/2021/COCO/cherry_dataset3/train_self/instances_self_train_combined_swin.json'

    # opt.path_bb = '/home/luis/2021/COCO/cherry_dataset3/train_self/predictions_swin_bb.json'
    # opt.path_segm = '/home/luis/2021/COCO/cherry_dataset3/train_self/predictions_swin_segm.json'

    opt.path_bb = '/home/luis/2022/mmdetection/mask_results.bbox.json'
    opt.path_segm = '/home/luis/2022/mmdetection/mask_results.segm.json'

    opt.shape = [1328,1328]
    opt.use_mask = True
    assert len(opt.shape) == 2, "ERROR invalid dimensions for shape item {}".format(opt.shape)
    main(opt.path_anns_st, opt.path_anns_train, opt.path_bb, opt.path_segm, opt)
