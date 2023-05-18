import os
import cv2
import argparse
import numpy as np
from tqdm import tqdm
from numpy import ndarray
from pycocotools.coco import COCO

import annotations as an


def normalize_annotations_indices(annotations, annotation_id):
    n_anns = len(annotations)
    updated_anns = []
    for i in range(n_anns):
        updated_anns += [annotations[i].copy()]
        updated_anns[-1]['id'] = annotation_id
        annotation_id += 1
    return updated_anns


class Range_value:
    def __init__(self, Min_range: int, Max_range: int):
        """
        Class to represent a range of valid numbers. A range can be disjoint, as such can be represented as sub ranges
        [[x1,x2],[x3,x4],...]. Where xi<xj for i<j.
        Parameters
        ----------
        Min_range: Minimun value
        Max_range: Maximum value
        """
        self.ranges = [[Min_range, Max_range]]

    def update_valid_ranges(self, invalid_min, invalid_max):

        new_ranges = []
        for i, mid_range in enumerate(self.ranges):
            if invalid_max < mid_range[0]:
                self.position_range_before(new_ranges, self.ranges[i:])
                break
            elif invalid_min > mid_range[1]:
                self.position_range_after(new_ranges, mid_range)
            elif invalid_min <= mid_range[0]:
                if invalid_max < mid_range[1]:
                    self.position_range_overlap_before(new_ranges, mid_range, invalid_max)
            elif invalid_max < mid_range[1]:

                self.position_range_overlap_interior(new_ranges, mid_range, invalid_min, invalid_max)
            else:
                self.position_range_overlap_after(new_ranges, mid_range, invalid_min)

        self.ranges = new_ranges

    def position_range_before(self, new_ranges, rest_ranges):
        for range_value in rest_ranges:
            new_ranges += [range_value.copy()]

    def position_range_after(self, new_ranges, mid_range):
        new_ranges += [mid_range.copy()]
        # return new_ranges

    def position_range_overlap_before(self, new_ranges, mid_range, invalid_max):
        new_ranges += [[invalid_max + 1, mid_range[1]]]

    def position_range_overlap_interior(self, new_ranges, mid_range, invalid_min, invalid_max):
        new_ranges += [[mid_range[0], invalid_min - 1]]
        new_ranges += [[invalid_max + 1, mid_range[1]]]

    def position_range_overlap_after(self, new_ranges, mid_range, invalid_min):
        new_ranges += [[mid_range[0], invalid_min - 1]]

    def get_valid_range(self):
        return self.ranges.copy()


def get_cut_location(range_horizontal, range_vertical):
    # index = np.random.randint(len(range_horizontal))
    col = np.random.choice(range_horizontal)
    # index = np.random.randint(len(range_horizontal))
    # row = range_vertical[index]
    row = np.random.choice(range_vertical)
    return row, col


def check_folder(folder_save_to):
    if not os.path.isdir(folder_save_to):
        os.mkdir(folder_save_to)


def xywh2xyxy_pixels(bounding_box: np.ndarray):
    """
    Function to get format a bounding box from a [x1,y1,w,h] pixel format,
    to a [x1,y1,x2,y2] pixel format.

    Parameters
    ----------
    bounding_box: original bounding_box

    Returns
        Resulting bounding box
    -------

    """
    x_pos1 = bounding_box[:, 0]
    x_pos2 = bounding_box[:, 0] + bounding_box[:, 2]
    y_pos1 = bounding_box[:, 1]
    y_pos2 = bounding_box[:, 1] + bounding_box[:, 2]
    x_pos1, x_pos2, y_pos1, y_pos2 = np.around(x_pos1), np.around(x_pos2), np.around(y_pos1), np.around(y_pos2)
    return x_pos1, x_pos2, y_pos1, y_pos2


def xyxy2xywh_positions(bounding_box: np.ndarray, image_shape: np.ndarray):
    """
    Function to pass a bounding box writen as x1,y1,x2,y2 columns (relative coordinates) to xc,yc,w,h (pixel cordinates)
    Parameters
    ----------
    bounding_box
    image_shape

    Returns
    -------

    """
    labels = np.empty(shape=bounding_box.shape, dtype=np.float32)
    # image_shape = np.flip(image_shape[:2])
    labels[:, 0] = bounding_box[:, 0]
    labels[:, 1] = ((bounding_box[:, 1] + bounding_box[:, 3]) / 2) / image_shape[1]
    labels[:, 2] = ((bounding_box[:, 2] + bounding_box[:, 4]) / 2) / image_shape[0]
    labels[:, 3] = (bounding_box[:, 3] - bounding_box[:, 1]) / image_shape[1]
    labels[:, 4] = (bounding_box[:, 4] - bounding_box[:, 2]) / image_shape[0]
    return labels


def top_right_annotator(image_shape, donator_shape):
    return np.array([0, 0])


def bottom_right_annotator(image_shape, donator_shape):
    return np.array([0, image_shape[1] - donator_shape[1]])


def bottom_right_annotator(image_shape, donator_shape):
    return np.array([image_shape[0] - donator_shape[0], 0])


def bottom_left_annotator(image_shape, donator_shape):
    return np.array([image_shape[0] - donator_shape[0], image_shape[1] - donator_shape[1]])


def modify_segments(segments, offset):
    n_segments = len(segments)
    for i in range(n_segments):
        points = len(segments[i]) // 2
        for j in range(points):
            segments[i][2 * j] += offset[0]
            segments[i][2 * j + 1] += offset[1]
    return segments


def update_annotation(ann: dict, image_shape: np.ndarray, donator_shape: np.ndarray, image_id: int, corner: str):
    if corner == "TR":
        offfset = top_right_annotator(image_shape, donator_shape)
    elif corner == "BR":
        offfset = bottom_right_annotator(image_shape, donator_shape)
    elif corner == "TL":
        offfset = top_right_annotator(image_shape, donator_shape)
    else:
        offfset = bottom_left_annotator(image_shape, donator_shape)

    ann['segmentation'] = modify_segments(ann['segmentation'], offfset)
    ann['bbox'][0] += offfset[0]
    ann['bbox'][1] += offfset[1]
    ann['image_id'] = image_id
    return ann


class mosaic_generator:
    def __init__(self, path_dataset, path_annotations, path_save, size_image=[1024, 1024], mode_random='uniform',
                 ranges_size=15, invalid_cats=[0]):

        self.path_anns = path_annotations
        self.coco = COCO(path_annotations)
        self.catIds = self.coco.getCatIds()
        self.imgIds = self.coco.getImgIds()
        self.invalid_categories = invalid_cats
        self.path_images = os.path.join(path_dataset, 'images')

        self.images = [self.coco.imgs[i]['file_name'] for i in range(len(self.coco.imgs))]
        self._path_imgs = [os.path.join(self.path_images, file) for file in self.images]
        self.n_images = len(self.images)

        self.path_save = path_save
        self.path_save_images = os.path.join(path_save, 'images')
        self.path_save_anns = os.path.join(path_save, 'instances_train.json')
        self.setup_new_dataset()

        if mode_random == 'uniform':
            self.get_mosaic_cut = self.generator_separator_random_uniform
        elif mode_random == 'gaussian':
            self.get_mosaic_cut = self.generator_separator_random_gaussian
        else:
            print("[ERROR] Invalid random mode {:s}. \n Choosing uniform random distribution instead: ".format(
                mode_random))
            size_image = [1024, 1024]
            self.get_mosaic_cut = self.generator_separator_random_uniform
        self.ranges_size = ranges_size
        self.sizes = np.array(size_image)
        # self.verbose = False

    def get_bounding_box(self, idx: int):
        annIds = self.coco.getAnnIds(imgIds=idx)
        anns_img = self.coco.loadAnns(annIds)
        val_ids = []
        bb = []
        for ann in anns_img:
            if not (ann['category_id'] in self.invalid_categories):
                bb += [ann['bbox']]
                val_ids += [ann]
        bb = np.array(bb).reshape([-1, 4])
        return bb, val_ids

    def __getitem__(self, idx):
        img = cv2.imread(self._path_imgs[idx])
        bb, anns = self.get_bounding_box(idx)
        return img, bb, anns

    def setup_new_dataset(self):
        check_folder(self.path_save)
        check_folder(self.path_save_images)

    def generator_separator_random_uniform(self, x1, x2):
        return round(np.random.uniform(x1, x2))

    def generator_separator_random_gaussian(self, x1, x2):
        mean = (x2 + x1) / 2
        std = (x2 - x1) / 4
        return np.random.normal(mean, std * std)

    def create_dataset(self, ratio_images, ext):
        new_dataset_size = int(self.n_images * ratio_images)

        size_dataset_valid_images = self.n_images
        valid_images_indexes = [i for i in range(size_dataset_valid_images)]

        np.random.shuffle(valid_images_indexes)
        new_images = []
        new_annotations = []

        # check_locations = [38]  # DELETE
        for i in tqdm(range(new_dataset_size)):  # DELETE
            # if i in check_locations:  # DELETE
            #     print("Index to check : ", i)
            #     check_locations.pop(0)
            #     self.verbose = True
            # else:
            #     self.verbose = False
            img, ann_img = self.get_new_image(valid_images_indexes, i + self.n_images)
            name = "mosaic_img_{:06d}.{:s}".format(i, ext)
            an.add_new_annotation(img, ann_img, i + self.n_images, name, new_images, new_annotations)
            self.save_image_mosaic(img, name)
        new_annotations = normalize_annotations_indices(new_annotations, annotation_id=len(self.coco.anns))
        an.save_new_dataset(self.path_images, new_images, new_annotations, self.path_anns, self.path_save_images,
                            self.path_save_anns)

    def get_new_image(self, valid_images_indexes: list, new_image_index: int):
        """
        Method to produce a mosaic image with its corresponding bounding boxes (at least 1 BB).

        Parameters
        ----------
        valid_images_indexes
        new_image_index

        Returns
        -------

        """
        ### Ranges for each crop are range 1: Top left, 2: Bottom left; 3: Top rigth, 4: Bottom rigth
        index_image1 = np.random.choice(valid_images_indexes)
        range_horizontal1, range_vertical1 = self.get_ranges_top_left(index_image1)

        while len(range_horizontal1) == 0:
            valid_images_indexes.remove(index_image1)
            index_image1 = np.random.choice(valid_images_indexes)
            range_horizontal1, range_vertical1 = self.get_ranges_top_left(index_image1)

        range_horizontal2, range_vertical2, index_image2 = self.get_ranges_bottom_left(range_horizontal1,
                                                                                       range_vertical1)
        range_horizontal3, range_vertical3, index_image3 = self.get_ranges_top_right(range_horizontal2)
        range_horizontal4, range_vertical4, index_image4 = self.get_ranges_bottom_right(range_horizontal3,
                                                                                        range_vertical3)

        indexes_found = [index_image1, index_image2, index_image3, index_image4]
        # if self.verbose: # DELETE
        #     print("Indices used: ", indexes_found)
        #     names = [self.images[idx] for idx in indexes_found]
        #     print("image names: ",names)
        img = np.empty([self.sizes[0], self.sizes[1], 3], dtype=np.uint8)

        cut_locations = get_cut_location(range_horizontal4, range_vertical4)

        new_annotations = []
        self.paste_image(img, indexes_found[3], cut_locations, new_annotations, new_image_index, corner="BR")

        self.paste_image(img, indexes_found[2], cut_locations, new_annotations, new_image_index, corner="TR")

        row = np.random.choice(range_vertical2)
        cut_locations = (row, cut_locations[1])

        self.paste_image(img, indexes_found[1], cut_locations, new_annotations, new_image_index, corner="BL")

        self.paste_image(img, indexes_found[0], cut_locations, new_annotations, new_image_index, corner="TL")

        return img, new_annotations

    def get_ranges_top_left(self, index):
        """
        Method to get horizontal and vertical valid ranges to cut the image without cutting through an annotation

        Parameters
        ----------
        index: int
        image index (loader based)

        Returns
        -------
        ranges: tuple
            ranges of valid locations.
        """

        img, bounding_boxes, annotations = self[index]

        shape = img.shape
        limits_height_lower, limits_height_upper = self.sizes[0] // 8, self.sizes[0] // 8 * 7
        limits_width_lower, limits_width_upper = self.sizes[1] // 8, self.sizes[1] // 8 * 7

        limits_height_upper = min(shape[0], limits_height_upper)
        limits_width_upper = min(shape[1], limits_width_upper)

        random_pos_i = self.get_mosaic_cut(limits_height_lower, limits_height_upper)
        random_pos_j = self.get_mosaic_cut(limits_width_lower, limits_width_upper)

        if len(bounding_boxes) > 0:

            x_pos1, x_pos2, y_pos1, y_pos2 = xywh2xyxy_pixels(bounding_boxes)

            count = 0
            while self.is_invalid_pos((0, random_pos_j), (0, random_pos_i), x_pos1, x_pos2, y_pos1, y_pos2, [-1, -1]):
                random_pos_i = self.get_mosaic_cut(limits_height_lower, limits_height_upper)
                random_pos_j = self.get_mosaic_cut(limits_width_lower, limits_width_upper)
                count += 1
                if count > 30:
                    return [], []

            range_y = self.get_ranges_y_from_point(random_pos_i, random_pos_j, self.ranges_size, x_pos1, x_pos2, y_pos1,
                                                   y_pos2, corner='TL')
            range_x = self.get_ranges_x_from_point(random_pos_i, random_pos_j, self.ranges_size, x_pos1, x_pos2, y_pos1,
                                                   y_pos2,
                                                   corner='TL')
        else:
            range_x = np.arange(random_pos_j, random_pos_j + self.ranges_size)
            range_y = np.arange(random_pos_i, random_pos_i + self.ranges_size)
        return range_x, range_y

    def get_ranges_bottom_left(self, range_horizontal, range_vertical):
        """
        Method to get ranges valid for columns and ros to cut a given image.
        Returns valid ranges plus the index of the corresponding image.
        Parameters
        ----------
        range_horizontal : ndarray
        range_vertical: ndarray

        Returns
        -------
        ranges: tuple
            ranges of valid locations.
        """
        not_found_yet = True
        random_index = 0

        while not_found_yet:
            random_img, random_anns, random_index, x_pos1, x_pos2, y_pos1, y_pos2 = self.get_random_coordinates()
            image_shape = random_img.shape

            range_vertical_local = image_shape[0] - self.sizes[0] + range_vertical

            if range_vertical_local[0] > 0:
                range_y = self.extract_valid_range_y_from_range(range_vertical_local, range_horizontal[-1],
                                                                x_pos1, x_pos2, y_pos1, y_pos2, corner="BL")

                if len(range_y) > 0:
                    range_x = self.extract_valid_range_x_from_range(range_horizontal, range_y[0], x_pos1, x_pos2,
                                                                    y_pos1, y_pos2, corner="BL")

                    if len(range_x) > 0:
                        not_found_yet = False

        range_y = self.sizes[0] - image_shape[0] + range_y
        return range_x, range_y, random_index

    def get_ranges_top_right(self, range_horizontal):
        not_found_yet = True
        random_index = 0

        while not_found_yet:

            random_img, random_anns, random_index, x_pos1, x_pos2, y_pos1, y_pos2 = self.get_random_coordinates()

            image_shape = random_img.shape
            range_horizontal_local = image_shape[1] - self.sizes[1] + range_horizontal

            limits_height_lower, limits_height_upper = self.sizes[0] // 8, self.sizes[0] // 8 * 7
            limits_height_upper = min(image_shape[0], limits_height_upper)

            random_row = self.get_mosaic_cut(limits_height_lower, limits_height_upper)

            count = 0
            while self.is_invalid_pos((0, range_horizontal_local[0]), (0, random_row), x_pos1, x_pos2, y_pos1,
                                      y_pos2, [0, -1]) and count < 30:  # repeat until limit = 30 is reach or no
                # bounding box is cropped.
                random_row = self.get_mosaic_cut(limits_height_lower, limits_height_upper)
                count += 1

            if count < 30:
                # distance = 15
                range_y = self.get_ranges_y_from_point(random_row, range_horizontal_local[0], self.ranges_size,
                                                       x_pos1, x_pos2, y_pos1, y_pos2, corner="TR")

                if len(range_y) > 0:
                    range_x = self.extract_valid_range_x_from_range(range_horizontal_local, random_row,
                                                                    x_pos1, x_pos2, y_pos1, y_pos2, corner="TR")

                    if len(range_x) > 0:
                        not_found_yet = False
        range_x = self.sizes[1] - image_shape[1] + range_x
        return range_x, range_y, random_index

    def get_random_coordinates(self):
        random_index = np.random.randint(self.n_images)
        random_img, random_bounding_box, random_annotations = self[random_index]
        x_pos1, x_pos2, y_pos1, y_pos2 = xywh2xyxy_pixels(random_bounding_box)
        return random_img, random_annotations, random_index, x_pos1, x_pos2, y_pos1, y_pos2

    def get_ranges_bottom_right(self, range_horizontal, range_vertical):
        not_found_yet = True
        random_index = 0

        while not_found_yet:

            random_img, random_anns, random_index, x_pos1, x_pos2, y_pos1, y_pos2 = self.get_random_coordinates()
            image_shape = random_img.shape
            range_vertical_local = image_shape[0] - self.sizes[0] + range_vertical
            range_horizontal_local = image_shape[1] - self.sizes[1] + range_horizontal

            if range_horizontal_local[0] > 0 and range_vertical_local[0] > 0:
                range_y = self.extract_valid_range_y_from_range(range_vertical_local, range_horizontal_local[0],
                                                                x_pos1, x_pos2, y_pos1, y_pos2, corner="BR")

                if len(range_y) > 0:
                    range_x = self.extract_valid_range_x_from_range(range_horizontal_local, range_y[0], x_pos1,
                                                                    x_pos2,
                                                                    y_pos1,
                                                                    y_pos2,
                                                                    corner="BR")

                    if len(range_x) > 0:
                        not_found_yet = False

        range_y = self.sizes[0] - image_shape[0] + range_y
        range_x = self.sizes[1] - image_shape[1] + range_x

        return range_x, range_y, random_index

    def get_ranges_y_from_point(self, row, col, size_range, x_pos1, x_pos2, y_pos1, y_pos2, corner='TL'):
        """
        Method to get valid y range for a set of bounding boxes defined by [x_pos1, x_pos2, y_pos1, y_pos2].
        Parameters
        ----------
        range_x : list
        row : int
        min_y : int
        max_y : int
        x_pos1 : ndarray
            array of size [n]
        x_pos2 : ndarray
            array of size [n]
        y_pos1 : ndarray
            array of size [n]
        y_pos2 : ndarray
            array of size [n]

        Returns
        -------
        list
        """
        if corner == 'TL':
            increase = -1
        elif corner == "TR":
            increase = -1
        elif corner == 'BL':
            increase = 1
        else:
            increase = 1

        function_horizontal, function_vertical, index_to_point = self.get_corner_info(corner)

        next_row = row + increase  # Check this validates immediately
        count = 0
        while not self.is_invalid_pos(function_horizontal(col), function_vertical(next_row), x_pos1, x_pos2, y_pos1,
                                      y_pos2, index_to_point) and count < size_range:
            next_row += increase
            count += 1
        next_row -= increase
        range_y = sorted([row, next_row])
        range_y = np.arange(range_y[0], range_y[1] + 1)
        return range_y

    def get_ranges_x_from_point(self, row, col, size_range, x_pos1, x_pos2, y_pos1, y_pos2, corner='TL'):
        """
        Method to get valid x range given a valid y range for a set of bounding boxes defined by [x_pos1, x_pos2, y_pos1, y_pos2].
        Parameters
        ----------
        range_y : list
        col : int
        min_x : int
        max_x : int
        x_pos1 : ndarray
            array of size [n]
        x_pos2 : ndarray
            array of size [n]
        y_pos1 : ndarray
            array of size [n]
        y_pos2 : ndarray
            array of size [n]

        Returns
        -------

        """
        if corner == 'TL':
            increase = -1
        elif corner == "TR":
            increase = 1

        elif corner == 'BL':
            increase = -1
        else:
            increase = 1

        function_horizontal, function_vertical, index_to_point = self.get_corner_info(corner)

        next_col = col + increase
        count = 0
        while not self.is_invalid_pos(function_horizontal(next_col), function_vertical(row), x_pos1, x_pos2, y_pos1,
                                      y_pos2, index_to_point) and count < size_range:
            next_col += increase
            count += 1

        next_col -= increase * 2
        range_x = sorted([col, next_col])
        range_x = np.arange(range_x[0], range_x[1] + 1)
        return range_x

    def is_invalid_pos(self, range_x: tuple, range_y: tuple, x_pos1: np.ndarray, x_pos2: np.ndarray, y_pos1: np.ndarray,
                       y_pos2: np.ndarray, index_desired: list):
        """
        Method to check if a bounding box defined by range_x and range_y cut through a label/bounding box, defined
        by the arrays x_pos1,x_pos2,y_pos1,y_pos2

        Parameters
        ----------
        range_x:
        range_y
        x_pos1
        x_pos2
        y_pos1
        y_pos2
        index_desired

        Returns
        -------

        """
        range_valid_x = Range_value(range_x[0], range_x[1])
        range_valid_y = Range_value(range_y[0], range_y[1])
        validity = True
        for i in range(len(x_pos1)):
            range_valid_x.update_valid_ranges(x_pos1[i], x_pos2[i])
            range_valid_y.update_valid_ranges(y_pos1[i], y_pos2[i])
            if len(range_valid_y.ranges) == 0 or len(range_valid_x.ranges) == 0:
                return True
            elif range_valid_x.ranges[index_desired[0]][index_desired[0]] != range_x[index_desired[0]] or \
                    range_valid_y.ranges[index_desired[1]][index_desired[1]] != range_y[index_desired[1]]:
                validity = False
                break
        return not validity

    def extract_valid_range_x_from_range(self, range_horizontal, row, x_pos1, x_pos2, y_pos1, y_pos2,
                                         corner='TL'):

        function_horizontal, function_vertical, index_to_point = self.get_corner_info(corner)
        valid_x = []
        connected = False
        for j in range_horizontal:
            if not self.is_invalid_pos(function_horizontal(j), function_vertical(row), x_pos1, x_pos2, y_pos1, y_pos2,
                                       index_to_point):
                valid_x += [j]
                connected = True
            elif connected:
                break
        return np.array(valid_x)

    def extract_valid_range_y_from_range(self, range_vertical, col, x_pos1, x_pos2, y_pos1,
                                         y_pos2, corner='TL'):
        function_horizontal, function_vertical, index_to_point = self.get_corner_info(corner)

        valid_y = []
        connected = False
        for i in range_vertical:

            if not self.is_invalid_pos(function_horizontal(col), function_vertical(i), x_pos1, x_pos2,
                                       y_pos1, y_pos2, index_to_point):
                valid_y += [i]
                connected = True
            elif connected:
                break
        return np.array(valid_y)

    def get_corner_info(self, corner):
        if corner == 'TL':
            function_horizontal = self.left
            function_vertical = self.upper
            index_to_point = [-1, -1]
        elif corner == "TR":
            function_horizontal = self.right
            function_vertical = self.upper
            index_to_point = [0, -1]
        elif corner == 'BL':
            function_horizontal = self.left
            function_vertical = self.lower
            index_to_point = [-1, 0]
        else:
            function_horizontal = self.right
            function_vertical = self.lower
            index_to_point = [0, 0]
        return function_horizontal, function_vertical, index_to_point

    def left(self, col):
        return (0, col)

    def right(self, col):
        return (col, self.sizes[1])

    def upper(self, row):
        return (0, row)

    def lower(self, row):
        return (row, self.sizes[0])

    def paste_image(self, img: np.ndarray, index_donor: int, position: tuple, new_annotations: list,
                    index_new_image: int, corner="TL"):
        """
        Method that paste a cut of a given image (defined by index_image) to a given corner,
        Parameters
        ----------

        img: tensor to paste image (HxWxC)
        index_donor: index of image donor
        position: Cut location
        new_annotations: json annotations to save to
        index_new_image: Index of the newly created image
        corner: Corner name
        Returns
        -------

        """
        img_donator, bb_donor, old_annotations = self[index_donor]
        bb_donor[:, [2, 3]] += bb_donor[:, [0, 1]]

        shape_donator = img_donator.shape

        if "L" in corner:
            ranges_x = self.left(position[1])  # absolute ranges correspond to resulting image coordinates
            range_x_local = np.array(ranges_x)  # local ranges correspond to donor image coordinates.
            if "T" in corner:
                ranges_y = self.upper(position[0])
                range_y_local = np.array(ranges_y)
            else:
                ranges_y = self.lower(position[0])
                range_y_local = shape_donator[0] - self.sizes[0] + np.array(ranges_y)
        else:
            ranges_x = self.right(position[1])
            range_x_local = shape_donator[1] - self.sizes[1] + np.array(ranges_x)
            if "T" in corner:
                ranges_y = self.upper(position[0])
                range_y_local = np.array(ranges_y)
            else:
                ranges_y = self.lower(position[0])
                range_y_local = shape_donator[0] - self.sizes[0] + np.array(ranges_y)

        img[ranges_y[0]:(ranges_y[1]), ranges_x[0]:ranges_x[1], :] = img_donator[range_y_local[0]:range_y_local[1],
                                                                     range_x_local[0]:range_x_local[1], :]
        if len(old_annotations) > 0:
            valid_annotations = self.get_valid_annotations(range_x_local, range_y_local, bb_donor)
            self.add_annotations(new_annotations, old_annotations, valid_annotations, index_new_image, self.sizes,
                                 np.array(shape_donator), corner)

    def save_image_mosaic(self, img, name):

        img_file = os.path.join(self.path_save_images, name)
        cv2.imwrite(img_file, img)

    def add_annotations(self, annotations: list, old_annotations: list, valid_positions: np.ndarray,
                        index_img: int, image_shape: np.ndarray, donator_shape: np.ndarray, corner) -> None:
        """
        Method to update the annotations file
        Parameters
        ----------
        donator_shape: shape of donating image
        image_shape: Shape of newly constructed image
        index_img: index of newly created image
        annotations: annotations of new images
        old_annotations: annotations of image
        valid_positions: valid object found in cut

        Returns
        -------

        """
        for i, ann in enumerate(old_annotations):
            if valid_positions[i]:
                # if not(ann['category_id'] in self.invalid_categories):
                annotations += [update_annotation(ann.copy(), image_shape, donator_shape, index_img, corner=corner)]

    def save_labels_yolo(self, labels_file, annotations: np.ndarray):
        """
        Method to save labels to a given given labels_file name

        Parameters
        ----------
        labels_file: path to file
        annotations: array of bounding boxes

        Returns
        -------

        """
        file1 = open(labels_file, "w")
        for bb in annotations:
            file1.writelines(
                "{:d} {:.10f} {:.10f} {:.10f} {:.10f} \n".format(int(bb[0]), bb[1],
                                                                 bb[2], bb[3], bb[4]))
        file1.close()  # to change file access modes

    def get_valid_annotations(self, range_x, range_y, bounding_boxes):
        lower_x = (bounding_boxes[:, 0] >= range_x[0] - 1).reshape([-1, 1])
        lower_y = (bounding_boxes[:, 1] >= range_y[0] - 1).reshape([-1, 1])
        upper_x = (bounding_boxes[:, 2] <= range_x[1] + 1).reshape([-1, 1])
        upper_y = (bounding_boxes[:, 3] <= range_y[1] + 1).reshape([-1, 1])

        conditions = np.concatenate([lower_x, lower_y, upper_x, upper_y], axis=1).reshape([-1, 4])
        valid = np.min(conditions, axis=1)
        return valid


def create_dataset_mosaic(path_dataset, path_annotations, path_to_save, ratio_images=0.5, ext='.jpg',
                          size=[1024, 1024]):
    """
    Parameters
    ----------
    path_dataset: Path to original dataset
    path_annotations: Path to json file (COCO format)
    path_to_save: Path to save new json file and imaages
    ratio_images: ratio between (images in new dataset):(images original dataset)
    ext: Extension of images (EX: .png)
    size: Default image size.

    Returns
    -------
    """
    mosaic = mosaic_generator(path_dataset, path_annotations=path_annotations, path_save=path_to_save, size_image=size)
    mosaic.create_dataset(ratio_images, ext)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path-dataset', type=str, default='', help='path to original dataset')
    parser.add_argument('--path-dataset-saving', type=str, default='', help='path to dataset to save to')
    parser.add_argument('--img-size', nargs='+', type=int, default=[1024, 1024], help='train,test sizes')
    parser.add_argument('--ratio', type=float, default=2.0, help='Ratio of size of new dataset in comparison'
                                                                 'to the original')
    parser.add_argument('--ext', type=str, default='png', choices=['png', 'jpg', 'JPG'],
                        help='Extension to which save the files.')
    parser.add_argument('--mask', action='store_true',
                        help='Save mask files as well.')

    opt = parser.parse_args()
    np.random.seed(61116)
    create_dataset_mosaic(opt.path_dataset, opt.path_dataset_annotations, opt.path_dataset_saving,
                          opt.ratio, ext=opt.ext, size=opt.img_size)
