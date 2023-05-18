import os
import numpy as np
# import matplotlib.image as mpimg
import cv2
import shutil


def check_folder(folder_save_to):
    if not os.path.isdir(folder_save_to):
        os.mkdir(folder_save_to)


def random_crop(img, default_size):
    img_shape = img.shape
    if img_shape[0] > default_size:
        if img_shape[1] > default_size:
            end_height = np.random.randint(default_size, img_shape[0])
            init_height = end_height - default_size
            end_width = np.random.randint(default_size, img_shape[1])
            init_width = end_width - default_size
            return img[init_height:end_height, init_width:end_width, :]
        else:
            end_height = np.random.randint(default_size, img_shape[0] + 1)
            init_height = end_height - default_size
            return img[init_height:end_height, :, :]

    # print("Shape image: ", img_shape)  # DELETE
    elif img_shape[0] == default_size:
        end_width = default_size
    else:
        end_width = np.random.randint(default_size, img_shape[1])
    init_width = end_width - default_size
    return img[:, init_width:end_width, :]


def rescale_image(image, new_shape):
    image = cv2.resize(image, (new_shape[1], new_shape[0]), interpolation=cv2.INTER_LINEAR)
    return image


def main(path_dataset, path_folder_save, default_size=1024):
    check_folder(path_folder_save)
    images = os.listdir(path_dataset)
    for image_name in images:
        path_img = os.path.join(path_dataset, image_name)
        image = get_image_pre_process(path_img,default_size)
        if image is None:
            continue

        cv2.imwrite(os.path.join(path_folder_save, image_name), image)
        os.remove(path_img)
        # break


def get_image_pre_process(path_img, default_size):
    image = cv2.imread(path_img, cv2.COLOR_BGR2RGB)
    if image is None:
        os.remove(path_img)
        return None

    elif len(image.shape) == 3:
        size_image = np.array(image.shape[:2])

    else:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        size_image = np.array(image.shape[:2])

    if min(size_image) >= default_size:
        image = random_crop(image, default_size)

    elif max(size_image) < default_size:
        new_shape = np.array(size_image / max(size_image) * default_size, dtype=np.int32)
        image = rescale_image(image, new_shape)

    else:
        new_shape = np.array(size_image / min(size_image) * default_size, dtype=np.int32)
        # print(new_shape)
        image = rescale_image(image, new_shape)
        image = random_crop(image, default_size)
    return image

def main2(path_dataset, path_folder_save, default_size=1024, ext='.jpg'):
    check_folder(path_folder_save)

    folders = os.listdir(path_dataset)
    for i, folder in enumerate(folders):
        if i > 100:
            break
        folder_name = folder.replace(' ', '_')
        path_folder = os.path.join(path_dataset, folder)

        images = os.listdir(path_folder)
        counter = 0
        for image_name in images:

            path_img = os.path.join(path_folder, image_name)
            image = get_image_pre_process(path_img,default_size)
            if image is None:
                continue
            image_name = "{:s}_{:03d}{:s}".format(folder_name, counter,ext)
            cv2.imwrite(os.path.join(path_folder_save, image_name), image)
            os.remove(path_img)
            counter += 1
        if len(os.listdir(path_folder)) == 0:
            os.rmdir(path_folder)
        # break


def get_new_name(path_to_save: str, file_name: str):
    parts_name = file_name.split("_")
    name = ''
    for part in parts_name[2:]:
        name += part + '_'
    name = os.path.join(path_to_save, name)
    return name


def main3(path_dataset: str, path_to_save: str):
    files = os.listdir(path_dataset)

    for file in files:
        path_file = os.path.join(path_dataset, file)
        new_path = get_new_name(path_to_save, file)
        shutil.copy(path_file, new_path)
        # break
        # os.remove(path_file)


if __name__ == '__main__':
    path_dataset = '/home/luis/fiftyone/coco-2017/train/data'
    # path_dataset = '/home/luis/2021/Data_augmentation/Self_training/downloads'
    # path_dataset = '/home/luis/2021/Data_augmentation/Self_training/downloaded_images'
    save_folder = '/home/luis/2021/Data_augmentation/Self_training/coco'
    default_size = 1024
    np.random.seed(33339)
    main(path_dataset, save_folder, default_size)

