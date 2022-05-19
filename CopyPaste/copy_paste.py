import os
import cv2
import argparse
import numpy as np
from tqdm import tqdm
from pycocotools.coco import COCO
import CopyPaste.annotations as an 

def check_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)


class distribution_gaussian:
    def __init__(self, parameters_gaussian):
        if isinstance(parameters_gaussian, list):
            parameters_gaussian = np.array(parameters_gaussian)

        parameters_gaussian = parameters_gaussian.reshape([-1, 2])
        self.mean = parameters_gaussian[:, 0]
        self.std = parameters_gaussian[:, 1]

    def __call__(self, value=1.0):
        result = np.random.normal(self.mean * value, np.abs(value) * self.std)
        # result = np.clip(result, self.mean*value - np.abs(value)*self.std * 2.5, self.mean*value  + np.abs(value)*self.std * 2.5)  # Clipping just to
        # maintain values in certain range
        return result


class distribution_uniform:
    def __init__(self, parameters_uniform: list):
        if isinstance(parameters_uniform, list):
            parameters_uniform = np.array(parameters_uniform)
        parameters_uniform = parameters_uniform.reshape([-1, 2])
        self.lower = parameters_uniform[:, 0]
        self.upper = parameters_uniform[:, 1]

    def __call__(self, value: float):
        return (np.random.uniform(self.lower * value, self.upper * value))


def crop_item(image, mask, shape):
    image = image[:shape[0], :shape[1], :]
    mask = mask[:shape[0], :shape[1]]
    return image, mask


def padd_item(image, mask, shape, shape2, padding):
    image2 = image.copy()
    mask2 = mask.copy()
    image = np.empty(shape, dtype=np.uint8)
    image.fill(padding[0])
    mask = np.empty([shape[0], shape[1]], dtype=np.uint8)
    mask.fill(0)

    row1 = np.random.randint(shape[0] - shape2[0])
    row2 = shape2[0] + row1
    if shape[1] < shape2[1]:
        col1 = np.random.randint(shape[1] - shape2[1])
        col2 = shape2[1] + col1
    else:
        col1 = 0
        col2 = shape2[1]
    image[row1:row2, col1:col2, :] = image2
    mask[row1:row2, col1:col2] = mask2
    return image, mask


def is_connected(mask, row, col):
    """
    Function to check if there is another value in next to the mask, to eliminate sporadic disconnected values
    Parameters
    ----------
    mask
    row
    col

    Returns
    -------

    """

    # value = mask[row - 1, col] > 0 or mask[row, col - 1] > 0 or mask[row + 1, col] > 0 or mask[row, col + 1] > 0
    value = mask[row - 1, col] > 0
    value = value or mask[row, col - 1]
    value = mask[row + 1, col] or value
    value = value or mask[row, col + 1] > 0
    return value


def update_mask(mask, row, col, margin=4):
    if mask[row, col] > 0:
        row1 = max(row - margin, 0)
        col1 = max(col - margin, 0)
        row2 = min(row + margin + 1, mask.shape[0])
        col2 = min(col + margin + 1, mask.shape[1])
        n_samples = (row2 - row1) * (col2 - col1)
        samples_threshold = round(n_samples * 0.25)
        if is_connected(mask, row, col):
            vals, counts = np.unique(mask[row1:row2, col1:col2], return_counts=True)
            update_mask_location(row, col, counts, mask, samples_threshold, vals)


def update_mask_location(row, col, counts, mask, samples_threshold, vals):
    if vals[0] == 0:
        indexes = np.argsort(counts[1:])
        if counts[1 + indexes[-1]] >= samples_threshold:
            mask[row, col] = vals[indexes[-1] + 1]
        else:
            mask[row, col] = 0
    else:
        indexes = np.argsort(counts)
        if counts[indexes[-1]] >= samples_threshold:
            mask[row, col] = vals[indexes[-1]]
        else:
            mask[row, col] = 0


def update_mask_border(mask, row_range, col_range, row, col):
    vals, counts = np.unique(mask[row_range[0]:row_range[1], col_range[0]:col_range[1]], return_counts=True)
    samples_threshold = round(np.sum(counts) * 0.2)
    update_mask_location(row, col, counts, mask, samples_threshold, vals)


def fix_mask(mask):
    """
    Function to group values according to cluster proximity.
    Parameters
    ----------
    mask

    Returns
    -------

    """
    shape = mask.shape
    new_mask = mask.copy()

    indices1 = np.repeat(np.arange(1, shape[1] - 1).reshape([1, -1]), shape[0] - 2, axis=0)  # Get no borders indices
    indices2 = np.arange(shape[1], shape[1] * (shape[0] - 1), shape[1]).reshape([-1, 1])  # Get no borders indices

    indices = indices1 + indices2
    indices = indices.reshape([-1])  # Get no borders indices
    np.random.shuffle(indices)
    for i, pos in enumerate(indices):
        row = pos // shape[1]
        col = pos % shape[1]
        update_mask(new_mask, row, col)
    margin_border = 3
    col1 = 0
    col2 = shape[1] - 1

    for i in range(shape[0]):
        if new_mask[i, col1] > 0:
            range_col1 = [col1, margin_border]
            range_row1 = [max(0, i - margin_border), min(shape[0], i + margin_border)]
            update_mask_border(new_mask, range_row1, range_col1, i, col1)
        if new_mask[i, col2] > 0:
            range_col2 = [col2 - margin_border, shape[1]]
            range_row2 = [max(0, i - margin_border), min(shape[0], i + margin_border)]
            update_mask_border(new_mask, range_row2, range_col2, i, col2)
    row1 = 0
    row2 = shape[0] - 1

    for i in range(shape[1]):
        if new_mask[row1, i] > 0:
            range_row1 = [row1, margin_border]
            range_col1 = [max(0, i - margin_border), min(shape[1], i + margin_border)]
            update_mask_border(new_mask, range_row1, range_col1, row1, i)
        if new_mask[row2, i] > 0:
            range_row2 = [row2 - margin_border, shape[0]]
            range_col2 = [max(0, i - margin_border), min(shape[1], i + margin_border)]
            update_mask_border(new_mask, range_row2, range_col2, row2, i)

    return new_mask


def get_shape_cuts(pos: tuple, shape_base: tuple, shape_cut: tuple):
    # shape_cut = np.array(shape_cut)
    p1 = [pos[0] - shape_cut[0] / 2, pos[1] - shape_cut[1] / 2]  # y1,x1
    p2 = [pos[0] + shape_cut[0] / 2, pos[1] + shape_cut[1] / 2]
    positions_base = np.array([p1, p2], dtype=np.int32)
    positions_base[:, 0] = np.clip(positions_base[:, 0], 0, shape_base[0])
    positions_base[:, 1] = np.clip(positions_base[:, 1], 0, shape_base[1])

    positions_cut = np.empty([2, 2], dtype=np.int32)
    difference_dimensions = positions_base[1, :] - positions_base[0, :]

    if difference_dimensions[0] < shape_cut[0]:
        positions_cut[0, 0] = np.random.randint(0, shape_cut[0] - difference_dimensions[0])
    else:
        positions_cut[0, 0] = 0
    if difference_dimensions[1] < shape_cut[1]:
        positions_cut[0, 1] = np.random.randint(0, shape_cut[1] - difference_dimensions[1])
    else:
        positions_cut[0, 1] = 0

    positions_cut[1, :] = positions_cut[0, :] + difference_dimensions
    return positions_base, positions_cut


class dataset_manager:
    def __init__(self, path_dataset_images: str, path_annotations: str, distribution='gaussian',
                 param_scale=[1.0, 0.4], param_translate=[[0.5, 0.5], [0.4, 0.4]],
                 grey_color=(114, 114, 114), invalid_cats=[],
                 min_size=9):

        self.coco = COCO(path_annotations)
        self.catIds = self.coco.getCatIds()
        self.imgIds = self.coco.getImgIds()
        self.invalid_categories = invalid_cats
        # images
        self.path_images = path_dataset_images
        check_folder(self.path_images)
        self.images = [self.coco.imgs[i]['file_name'] for i in range(len(self.coco.imgs))]

        self._path_imgs = [os.path.join(self.path_images, file) for file in self.images]

        #######  TRANSFORMATION PARAMETERS  #######
        assert (distribution in ['gaussian', 'uniform']), "Unvalid distribution: {}".format(distribution)
        if distribution == 'gaussian':
            self.position_generator = distribution_gaussian(param_translate)
            self.new_scale_generator = distribution_gaussian(param_scale)

        else:
            self.position_generator = distribution_uniform(param_translate)
            self.new_scale_generator = distribution_uniform(param_scale)

        self.padding = grey_color
        self.min_size = min_size

    def build_mask(self, idx: int):
        """
        Method to create the mask of a corresponding image.
        Parameters
        ----------
        idx

        Returns
        -------

        """
        img = self.coco.imgs[idx]
        annIds = self.coco.getAnnIds(imgIds=idx)
        anns_img = self.coco.loadAnns(annIds)
        mask_annotations = np.zeros([img['height'], img['width']], dtype=np.uint8)
        for i in range(len(anns_img)):
            mask = self.coco.annToMask(anns_img[i])
            if not (anns_img[i]['category_id'] in self.invalid_categories):
                mask_annotations = np.where(mask == 1, i + 1, mask_annotations).astype(dtype=np.uint8)

        return mask_annotations, anns_img

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index: int):
        image = cv2.imread(self._path_imgs[index])
        mask, anns_mask = self.build_mask(index)
        return image, mask, anns_mask

    def create_copy_pasted_masks(self, range_samples: list, modify: bool, img_id: int, ann_id: int):
        """
        Method to create an image and its corresponding mask and annotations
        Parameters
        ----------
        range_samples
        modify: If True the base image is rescaled
        img_id: ID to identify the new image
        ann_id: Annotation ID of last annotation

        Returns
        -------

        """

        random_index = np.random.randint(len(self.images))
        image_base, mask_base, anns_base = self[random_index]

        if modify:
            image_base, mask_base = self.transform(image_base, mask_base)

        samples_in_image = len(anns_base)  #
        n_samples = np.random.randint(range_samples[0], range_samples[1])
        n_samples_valid = min(255 - samples_in_image, n_samples)  # max number of instances = 256

        # if n_samples_valid == 0:  # max number of instances is 255 # UNCOMMENT
        #     return self.create_copy_pasted_masks(n_samples, modify, img_id, ann_id)
        categories = {}
        for i, ann in enumerate(anns_base):
            categories[i + 1] = ann['category_id']

        new_image, new_mask = image_base.copy(), mask_base.copy()

        for i in range(n_samples_valid):
            # create new image
            new_image, new_mask, categories = self.add_figure(new_image, new_mask, samples_in_image + 1 + i,
                                                              categories, modify)

        new_mask = fix_mask(new_mask)

        new_anns = an.get_new_annotation(new_mask, img_id, ann_id, categories)
        return new_image, mask_base, new_anns

    def transform(self, image: np.ndarray, mask: np.ndarray):
        """
        Method to apply transformations to a mask and image. Mainly rescale and translation

        Parameters
        ----------
        image
        mask

        Returns
        -------

        """
        shape = np.array(image.shape)

        factor = float(self.new_scale_generator(1)[0])
        image = cv2.resize(image, None, fx=factor, fy=factor, interpolation=cv2.INTER_AREA)

        mask = cv2.resize(mask, None, fx=factor, fy=factor, interpolation=cv2.INTER_AREA)
        shape2 = image.shape
        if shape[0] < shape2[0]:
            image, mask = crop_item(image, mask, shape)

        elif shape[0] > shape2[0]:
            image, mask = padd_item(image, mask, shape, shape2, self.padding)
        return image, mask

    def add_figure(self, image: np.ndarray, mask: np.ndarray, intensity: int, categories: dict, modify: bool):
        random_index = np.random.randint(len(self.images))
        image_copy, mask_copy, anns_copy = self[random_index]  # sample to copy from

        n_segments = len(anns_copy)
        random_segment = np.random.randint(n_segments)
        position = self.position_generator(np.array(image.shape)[:2]) # (y,x)
        # get range from the random segment.
        range_segment, random_segment = self.get_range_segment(mask_copy, anns_copy, random_segment)
        #  range_segment is organized as ((x1,x2),(y1,y2))
        if range_segment[0][0] < self.min_size or range_segment[0][1] < self.min_size:
            return self.add_figure(image, mask, intensity, categories, modify)

        categories[intensity] = anns_copy[random_segment]['category_id']
        image, mask = self.paste_segment(image, mask, image_copy, mask_copy, random_segment + 1, intensity,
                                         pos_paste=position, range_copy=range_segment, modify=modify)
        return image, mask, categories

    def get_range_segment(self, mask: np.ndarray, annotations: list, segment_number: int):
        if segment_number < 1:
            return ((0, 0), (0, 0)), 1

        category_id = annotations[segment_number]['category_id']
        if category_id in self.invalid_categories:
            return self.get_range_segment(mask, annotations, segment_number - 1)

        bbox = annotations[segment_number]['bbox']
        xmin = np.min(bbox[0])
        xmax = np.max(bbox[0] + bbox[2])
        ymin = np.min(bbox[1])
        ymax = np.max(bbox[1] + bbox[3])

        delta = round((mask.shape[0] + mask.shape[1]) / 2 * 0.01)
        return ((round(max(ymin - delta, 0)), round(min(ymax + delta, mask.shape[0]))),
                (round(max(xmin - delta, 0)), round(min(xmax + delta, mask.shape[1])))), segment_number

    def paste_segment(self, image_base: np.ndarray, mask_base: np.ndarray, image_donor: np.ndarray,
                      mask_donor: np.ndarray,
                      segment_value: int, intensity: int, pos_paste: tuple, range_copy: tuple, modify=False):
        """
        Method to paste a given segment in the image base. The base mask is updated as well.
        Parameters
        ----------
        image_base: Image to copy to
        mask_base: Mask to copy to
        image_donor: Image to copy from
        mask_donor: Mask to copy from
        segment_value: ID of segment in mask (value between 0-255)
        intensity: Color to paste in mask
        pos_paste: Location to paste ([y,x])
        range_copy: Range of paste
        modify: If True the mask is rescaled

        Returns
        -------

        """

        shape1 = image_base.shape
        if modify:
            image_donor, mask_donor = self.transform(image_donor[range_copy[0][0]:range_copy[0][1],
                                                     range_copy[1][0]:range_copy[1][1], :],
                                                     mask_donor[range_copy[0][0]:range_copy[0][1],
                                                     range_copy[1][0]:range_copy[1][1]])
        mask_donor = fix_mask(mask_donor)
        shape2 = image_donor.shape

        # get dimensions of a bounding box that cover a random segment in original image
        rect_base, rect_donor = get_shape_cuts(pos_paste, shape1, shape2)

        if (rect_donor[1, 0] > self.min_size) and (rect_donor[1, 1] > self.min_size):
            # paste from the image donor to a
            mini_base = image_base[rect_base[0, 0]:rect_base[1, 0], rect_base[0, 1]:rect_base[1, 1], :]

            image_base[rect_base[0, 0]:rect_base[1, 0], rect_base[0, 1]:rect_base[1, 1], :] = np.where(
                mask_donor[rect_donor[0, 0]:rect_donor[1, 0], rect_donor[0, 1]:rect_donor[1, 1], None] == segment_value,
                image_donor[rect_donor[0, 0]:rect_donor[1, 0], rect_donor[0, 1]:rect_donor[1, 1], :],
                mini_base)

            mask_base[rect_base[0, 0]:rect_base[1, 0], rect_base[0, 1]:rect_base[1, 1]] = np.where(
                mask_donor[rect_donor[0, 0]:rect_donor[1, 0], rect_donor[0, 1]:rect_donor[1, 1]] == segment_value,
                intensity, mask_base[rect_base[0, 0]:rect_base[1, 0],
                           rect_base[0, 1]:rect_base[1, 1]])

        return image_base, mask_base

    def get_last_ids(self):
        last_img_id = len(self.images) - 1
        annIds = self.coco.getAnnIds(imgIds=[last_img_id])
        anns_img = self.coco.loadAnns(annIds)
        last_ann_id = anns_img[-1]['id']
        return last_img_id, last_ann_id


def save_mask(mask: np.ndarray, dataset_path: str, name_file: str):
    cv2.imwrite(os.path.join(dataset_path, 'masks', name_file), mask)


def setup_new_dataset(path_to_save: str):
    check_folder(path_to_save)
    path_to_save_images = os.path.join(path_to_save, 'images')
    check_folder(path_to_save_images)
    return path_to_save_images


def main(path_dataset_img: str, path_anns: str, path_to_save: str, ratio_dataset: float,
         range_samples: list, distribution: str, param_distribution: list, param_scale: list,
         modify=True, name_anns='instances_train.json', opt=None):
    ignore_categories = opt.ignore_cat
    min_size = opt.min_size
    check_folder(path_to_save)
    assert range_samples[0] < range_samples[1], "ERROR, INVALID RANGE SAMPLES {:d} > {:d}".format(range_samples[0],
                                                                                                  range_samples[1])
    manager = dataset_manager(path_dataset_img, path_anns, distribution, param_translate=param_distribution,
                              param_scale=param_scale, invalid_cats=ignore_categories, min_size=min_size)

    n_images = round(len(manager) * ratio_dataset)
    path_to_save_images = setup_new_dataset(path_to_save)
    annotations = []
    images = []
    last_id_img, last_id_ann = manager.get_last_ids()
    path_to_save_anns = os.path.join(path_to_save, name_anns)
    last_id_img += 1
    for i in tqdm(range(n_images)):
        image_id = i + last_id_img
        new_img, new_mask, new_annotation = manager.create_copy_pasted_masks(range_samples, modify,
                                                                             image_id, last_id_ann + 1)

        name_file = "img_gen_{:04d}.png".format(i)

        if len(new_annotation)>0:
            last_id_ann = new_annotation[-1]['id']

        an.add_new_annotation(new_img, new_annotation, image_id, name_file, images, annotations)
        cv2.imwrite(os.path.join(path_to_save_images, name_file), new_img)

    an.save_new_dataset(path_dataset_img, images, annotations, path_anns, path_to_save_images, path_to_save_anns)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='copy_paste_augmentation.py',
                                     description='This is an implementation of copy-paste augmentation based on '
                                                 'https://arxiv.org/abs/2012.07177v2. The dataset must be organized'
                                                 'as: '
                                                 '-dataset'
                                                 '  -images'
                                                 '      -image1.png'
                                                 '      -image2.png'
                                                 '      ...'
                                                 '  -masks'
                                                 '      -image1.png'
                                                 '      -image2.png'
                                                 '      ...'
                                                 'Each mask is a PNG image that define each object with a unique color '
                                                 'defined as and integer 0<=n<256 in a scale of gray.'
                                                 ''
                                                 'Each image new image is constructed from a base image and several '
                                                 'instances.')

    parser.add_argument('--path-dataset', type=str, help='path of the dataset')
    parser.add_argument('--path-annotations', default='/home/dataset/minneapple/annotations/instances_train.json',
                        type=str, help='path of annotation file (.json)')
    parser.add_argument('--path-to-save', type=str, default='predictions',
                        help='Path to save files, in a similar arrange as that of the original dataset.')
    parser.add_argument('--ratio-dataset', type=float, default=2.00,
                        help='Percentage of images to create, relative to the original dataset size.')
    parser.add_argument('--n-samples', nargs='+', type=int, default=[3, 21],
                        help='Number samples to take from other images')
    parser.add_argument('--distribution', type=str, default='uniform', choices=['gaussian', 'uniform'],
                        help='Method to distribute copy pasted images around the image.')

    parser.add_argument('--param-distribution', nargs='+', type=float, default=[0.1, 0.9],
                        help='Parameters that define the distribution random function. In the case of the uniform'
                             'distribution are the upper and lower locations relative to the image size (e.g. [0.2,0.8]). On the '
                             'other hand, for a gaussian distribution are the values of mean and variance,'
                             'also relative to the image size. e.g.:[0.5,0.25] or [0.5, 0.25, 0.5,0.3]. This one is'
                             'for a 2 dims gaussian distribution, with a different x axis distribution than the y axis.'
                             'mean_x= 0.5 and std_x = 0.25. ')

    parser.add_argument('--param-rescale', nargs='+', type=float, default=[0.9, 1.1],
                        help='Parameters that define the distribution random function. In the case of the uniform'
                             'distribution are the upper and lower locations relative to the image size '
                             '(e.g. [0.9,1.1]). On the other hand, for a gaussian distribution are the values of mean'
                             ' and variance, also relative to the image size. e.g.:[1,0.25] or [1.0, 0.25, 1.0,0.3]. '
                             'This one is for a 2 dims gaussian distribution, with a different x axis distribution '
                             'than the y axis. In this case mean_y= 1 and std_y = 0.25. and mean_x= 1 and std_x = 0.3')

    parser.add_argument('--min-size', type=float, default=15,
                        help='Min width and height dimension of BB')

    parser.add_argument('--modify-base', action='store_true', help='Modify the size of the image to copy from.')
    parser.add_argument('--ignore-cat', nargs='+', type=float, default=[0],
                        help='Class categories to not copy-paste. By default ignores only class 0/background.')

    opt = parser.parse_args()
    np.random.seed(903)
    opt.distribution = 'gaussian'
    if len(opt.param_distribution) == 4:
        opt.param_distribution = np.array([opt.param_distribution[:2], opt.param_distribution[2:]])
    if len(opt.param_distribution) == 2:
        opt.param_distribution = np.array(opt.param_distribution)
    else:
        raise "ERROR: Invalid size for param-distribution. Size must be either 2 or 4 not {:d}".format(len(
            opt.param_distribution))

    opt.param_rescale = [1.05, 0.05]  # mean and variance in a gaussian distribution
    if len(opt.param_rescale) == 4:
        opt.param_rescale = np.array([opt.param_rescale[:2], opt.param_rescale[2:]])
    if len(opt.param_rescale) == 2:
        opt.param_rescale = np.array(opt.param_rescale)
    else:
        raise "ERROR: Invalid size for param-distribution. Size must be either 2 or 4 not {:d}".format(len(
            opt.param_rescale))

    main(opt.path_dataset, opt.path_annotations, opt.path_to_save, opt.ratio_dataset, opt.n_samples, opt.distribution,
         param_distribution=opt.param_distribution, param_scale=opt.param_rescale, modify=opt.modify_base, opt=opt)
