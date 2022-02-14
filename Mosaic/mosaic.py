import argparse
import cv2
import os
from tqdm import tqdm
from numpy import ndarray
import numpy as np
import Mosaic_DA.dataset_yolo as dy


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


def xywh2xyxy_pixels(bounding_box:np.ndarray, image_shape:tuple):
    """
    Function to get format a bounding box from a [xc,yc,w,h] relative format (xc and yx are the center coordinates),
    to a [x1,y1,x2,y2] absolute pixel format.

    Parameters
    ----------
    bounding_box: original bounding_box
    image_shape: Tuple of shape. EX:(1280,720,3)

    Returns
        Resulting bounding box
    -------

    """
    # image_shape = np.flip(image_shape[:,0])
    x_pos1 = (bounding_box[:, 1] - bounding_box[:, 3] / 2) * image_shape[1]
    x_pos2 = (bounding_box[:, 1] + bounding_box[:, 3] / 2) * image_shape[1]
    y_pos1 = (bounding_box[:, 2] - bounding_box[:, 4] / 2) * image_shape[0]
    y_pos2 = (bounding_box[:, 2] + bounding_box[:, 4] / 2) * image_shape[0]
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


class mosaic_generator:
    def __init__(self, path_dataset, path_to_save, size_image=[1024, 1024], mode_random='uniform',ranges_size = 15):
        self.loader = dy.dataset_yolo(path_dataset)
        self.path_to_save = path_to_save
        self.path_labels = os.path.join(self.path_to_save, 'labels')
        self.path_images = os.path.join(self.path_to_save, 'images')
        self.setup_new_dataset()

        if mode_random == 'uniform':
            self.get_mosaic_cut = self.generator_separator_random_uniform
        elif mode_random == 'gaussian':
            self.get_mosaic_cut = self.generator_separator_random_gaussian
        else:
            print("[ERROR] Invalid random mode {:s}. \n Chosing uniform random distribution instead: ".format(
                mode_random))
            size_image = [1024, 1024]
            self.get_mosaic_cut = self.generator_separator_random_uniform
        self.ranges_size = ranges_size
        self.sizes = np.array(size_image)
        # self.params_random = params

    def setup_new_dataset(self):
        if not os.path.isdir(self.path_to_save):
            os.mkdir(self.path_to_save)

        if not os.path.isdir(self.path_labels):
            os.mkdir(self.path_labels)
            os.mkdir(self.path_images)

    def generator_separator_random_uniform(self, x1, x2):
        return round(np.random.uniform(x1, x2))

    def generator_separator_random_gaussian(self, x1, x2):
        mean = (x2 + x1) / 2
        std = (x2 - x1) / 4
        return np.random.normal(mean, std * std)

    def create_dataset(self, ratio_images, ext):
        dataset_size = len(self.loader)
        new_dataset_size = int(dataset_size * ratio_images)
        size_dataset_valid_images = len(self.loader)
        valid_images_indexes = [i for i in range(size_dataset_valid_images)]
        np.random.shuffle(valid_images_indexes)
        for i in tqdm(range(new_dataset_size)):
            img, annotations = self.get_new_image(valid_images_indexes, i)
            self.save_image_mosaic(img, annotations, i, ext)

    def get_new_image(self, valid_images_indexes: list, image_index: int):
        """
        Method to produce a mosaic image with its corresponding bounding boxes (at least 1 BB).

        Parameters
        ----------
        valid_images_indexes
        image_index

        Returns
        -------

        """
        ### Ranges for each crop are range 1: Top left, 2: Bottom left; 3: Top rigth, 4: Bottom rigth
        size_dataset_valid_images = len(valid_images_indexes)
        index_image1 = valid_images_indexes[image_index % size_dataset_valid_images]
        range_horizontal1, range_vertical1 = self.get_ranges_top_left(index_image1)

        while len(range_horizontal1) == 0:
            valid_images_indexes.pop(image_index % size_dataset_valid_images)
            size_dataset_valid_images -= 1
            index_image1 = valid_images_indexes[image_index % size_dataset_valid_images]
            range_horizontal1, range_vertical1 = self.get_ranges_top_left(index_image1)

        range_horizontal2, range_vertical2, index_image2 = self.get_ranges_bottom_left(range_horizontal1,
                                                                                       range_vertical1)
        range_horizontal3, range_vertical3, index_image3 = self.get_ranges_top_right(range_horizontal2)
        range_horizontal4, range_vertical4, index_image4 = self.get_ranges_bottom_right(range_horizontal3,
                                                                                        range_vertical3)

        indexes_found = [index_image1, index_image2, index_image3, index_image4]
        img = np.empty([self.sizes[0], self.sizes[1], 3], dtype=np.uint8)

        cut_locations = get_cut_location(range_horizontal4, range_vertical4)
        annotations = self.paste_image(img, indexes_found[3],
                                       cut_locations, [], corner="BR")

        annotations = self.paste_image(img, indexes_found[2],
                                       cut_locations, annotations, corner="TR")

        row = np.random.choice(range_vertical2)
        cut_locations = (row, cut_locations[1])

        annotations = self.paste_image(img, indexes_found[1],
                                       cut_locations, annotations, corner="BL")

        annotations = self.paste_image(img, indexes_found[0],
                                       cut_locations, annotations, corner="TL")

        labels = np.array(annotations, dtype=int)
        if len(labels) > 0:
            labels = xyxy2xywh_positions(labels, self.sizes)
            return img, labels
        else:
            return self.get_new_image(valid_images_indexes, image_index)

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
        img, bounding_box = self.loader[index]
        shape = img.shape

        x_pos1, x_pos2, y_pos1, y_pos2 = xywh2xyxy_pixels(bounding_box, shape)

        limits_height_lower, limits_height_upper = self.sizes[0] // 8, self.sizes[0] // 8 * 7
        limits_width_lower, limits_width_upper = self.sizes[1] // 8, self.sizes[1] // 8 * 7

        limits_height_upper = min(shape[0], limits_height_upper)
        limits_width_upper = min(shape[1], limits_width_upper)

        random_pos_i = self.get_mosaic_cut(limits_height_lower, limits_height_upper)
        random_pos_j = self.get_mosaic_cut(limits_width_lower, limits_width_upper)

        count = 0
        while self.is_invalid_pos((0, random_pos_j), (0, random_pos_i), x_pos1, x_pos2, y_pos1, y_pos2, [-1, -1]):
            random_pos_i = self.get_mosaic_cut(limits_height_lower, limits_height_upper)
            random_pos_j = self.get_mosaic_cut(limits_width_lower, limits_width_upper)
            count += 1
            if count > 30:
                return [], []

        # range_x = [0, random_pos_j]
        # distance = 15
        range_y = self.get_ranges_y_from_point(random_pos_i, random_pos_j, self.ranges_size, x_pos1, x_pos2, y_pos1,
                                               y_pos2, corner='TL')
        range_x = self.get_ranges_x_from_point(random_pos_i, random_pos_j, self.ranges_size, x_pos1, x_pos2, y_pos1,
                                               y_pos2,
                                               corner='TL')
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
            random_index = np.random.randint(len(self.loader))
            random_img, random_labels = self.loader[random_index]
            image_shape = random_img.shape
            x_pos1, x_pos2, y_pos1, y_pos2 = xywh2xyxy_pixels(random_labels, image_shape)

            range_vertical_local = image_shape[0] - self.sizes[0] + range_vertical

            if range_vertical_local[0] > 0:
                # max_col = min(size_range[-1] + 1, image_shape[1])
                # range_horizontal_local = np.arange(size_range[0], max_col)
                # max_row = min(range_vertical_local[-1] + 1,image_shape[0])
                # range_vertical_local = np.arange(range_vertical_local[0], max_row)
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

            random_index = np.random.randint(len(self.loader))
            random_img, random_labels = self.loader[random_index]
            image_shape = random_img.shape
            x_pos1, x_pos2, y_pos1, y_pos2 = xywh2xyxy_pixels(random_labels, image_shape)

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

    def get_ranges_bottom_right(self, range_horizontal, range_vertical):
        not_found_yet = True
        random_index = 0

        while not_found_yet:

            random_index = np.random.randint(len(self.loader))

            random_img, random_labels = self.loader[random_index]
            image_shape = random_img.shape
            x_pos1, x_pos2, y_pos1, y_pos2 = xywh2xyxy_pixels(random_labels, image_shape)

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
                       y_pos2: np.ndarray, index_desired: int):
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

    def paste_image(self, img, index_image, position, annotations, corner="TL"):
        """
        Method that paste a cut of a given image (defined by index_image) to a given corner,
        Parameters
        ----------
        img
        index_image
        position
        annotations
        corner

        Returns
        -------

        """
        img_donator, bb = self.loader[index_image]
        shape_donator = img_donator.shape
        x_pos1, x_pos2, y_pos1, y_pos2 = xywh2xyxy_pixels(bb, img_donator.shape)

        bounding_boxes = np.empty([len(bb), 5], dtype=int);
        bounding_boxes[:, 0] = bb[:, 0];
        bounding_boxes[:, 1] = x_pos1;
        bounding_boxes[:, 2] = y_pos1;
        bounding_boxes[:, 3] = x_pos2;
        bounding_boxes[:, 4] = y_pos2;
        if "L" in corner:
            ranges_x = self.left(position[1])
            range_x_local = np.array(ranges_x)
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

        annotations = self.add_annotations(annotations, bounding_boxes, position, corner)
        return annotations

    def save_image_mosaic(self, img, annotations, index=0, ext='jpg'):
        name = "mosaic_img_{:07d}".format(index)
        labels_file = os.path.join(self.path_labels, "{:s}.txt".format(name))
        img_file = os.path.join(self.path_images, "{:s}.{:s}".format(name, ext))
        cv2.imwrite(img_file, img, )
        self.save_labels_yolo(labels_file, annotations)

    def add_annotations(self, annotations, bounding_box, position, corner):

        classes, x_pos1, y_pos1, x_pos2, y_pos2 = bounding_box[:, 0], bounding_box[:, 1], bounding_box[:, 2], \
                                                  bounding_box[:, 3], bounding_box[:, 4]
        if corner == 'TL':
            for i, (x1, x2, y1, y2) in enumerate(zip(x_pos1, x_pos2, y_pos1, y_pos2)):
                if x2 < position[1]:
                    if y2 < position[0]:
                        # w = x2 - x1
                        # h = y2 - y1
                        # xc = x1 + w / 2.0
                        # yc = y1 + h / 2.0
                        # annotations += [[xc, yc, w, h]]
                        annotations += [[classes[i], x1, y1, x2, y2]]

        elif corner == "TR":
            for i, (x1, x2, y1, y2) in enumerate(zip(x_pos1, x_pos2, y_pos1, y_pos2)):
                if x1 > position[1]:
                    if y2 < position[0]:
                        annotations += [[classes[i], x1, y1, x2, y2]]

        elif corner == 'BL':
            for i, (x1, x2, y1, y2) in enumerate(zip(x_pos1, x_pos2, y_pos1, y_pos2)):
                if x2 < position[1]:
                    if y1 > position[0]:
                        annotations += [[classes[i], x1, y1, x2, y2]]


        else:
            for i, (x1, x2, y1, y2) in enumerate(zip(x_pos1, x_pos2, y_pos1, y_pos2)):
                if x1 > position[1]:
                    if y1 > position[0]:
                        annotations += [[classes[i], x1, y1, x2, y2]]
        return annotations

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


def create_dataset_mosaic(path_dataset, path_to_save, ratio_images=0.5, ext='.jpg', size=[1024, 1024]):
    """
    Function to create a dataset
    Parameters
    ----------

    size: Size images to generate
    path_dataset: Path to the folder with sub folders images and labels, to load from
    path_to_save: Path to folder to save to, same format.
    ratio_images: Fraction of images to present
    ext: extension name (jpg, png)
    Returns
    -------

    """
    mosaic = mosaic_generator(path_dataset, path_to_save,size_image=size)
    mosaic.create_dataset(ratio_images, ext)





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path-dataset', type=str, default='', help='path to original dataset')
    parser.add_argument('--path-dataset-saving', type=str, default='', help='path to dataset to save to')
    parser.add_argument('--img-size', nargs='+', type=int, default=[1024, 1024], help='train,test sizes')
    parser.add_argument('--ratio', type=float, default=2.0, help='Ratio of size of new dataset in comparison'
                                                                 'to the original')
    parser.add_argument('--ext', type=str, default='png',choices=['png','jpg'], help='Extension to which save the files.')

    opt = parser.parse_args()
    np.random.seed(3)
    # opt.path_dataset = '/home/luis/datasets/wider/wider_val'  # 12880
    # opt.path_dataset_saving = '/home/luis/datasets/wider/wider_val2'
    # opt.path_dataset_saving = '/home/luis/datasets/wider/mosaic_wider_face'

    opt.path_dataset = '/home/luis/datasets/minneapple/train'  # 12880
    opt.path_dataset_saving = '/home/luis/datasets/minneapple/train2'
    opt.img_size = [1280, 720]
    create_dataset_mosaic(opt.path_dataset, opt.path_dataset_saving, opt.ratio, ext=opt.ext,size=opt.img_size)

