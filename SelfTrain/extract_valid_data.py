import os
import numpy as np
import shutil


def check_folder(folder_save_to):
    if not os.path.isdir(folder_save_to):
        os.mkdir(folder_save_to)


def get_valid_predictions_yolor(predictions, threshold):
    """
    Function to take
    Parameters
    ----------
    predictions

    Returns
    -------

    """
    valid = predictions[predictions[:, 5] > threshold, :]
    valid_copy = valid.copy()
    valid_copy[:,2:6] = valid_copy[:, 1:5]
    valid_copy[:,1] = valid[:, 5]
    return valid_copy

def get_valid_predictions_yolov4(predictions, threshold):
    """
    Function to take
    Parameters
    ----------
    predictions

    Returns
    -------

    """
    valid = predictions[predictions[:, 1] > threshold, :]
    return valid


def move_image(pred_name: str, path_images: str, folder_to_save_images: str, ext=".jpg", delete=True):
    name_img = pred_name.replace(".txt", ext)
    img_path = os.path.join(path_images, name_img)
    shutil.copy(img_path, os.path.join(folder_to_save_images, name_img))
    if delete:
        os.remove(img_path)


def move_prediction(prediction_name, valid_predictions, folder_to_save_images, delete=True):
    file_path = os.path.join(folder_to_save_images, prediction_name)
    file1 = open(file_path, "w")
    for bb in valid_predictions:
        file1.writelines("{:d} {:.10f} {:.10f} {:.10f} {:.10f}\n".format(0, bb[2], bb[3], bb[4], bb[5]))
    # \n is placed to indicate EOL (End of Line)
    file1.close()  # to change str_input access modes
    if delete:
        os.remove(file_path)


def main(path_images: str, path_predictions: str, folder_save_to: str, delete_files: str, threshold=0.2,
         extension='.jpg',yolor=False):
    files_pred = os.listdir(path_predictions)
    check_folder(folder_save_to)
    folder_to_save_images = os.path.join(folder_save_to, 'images')
    folder_to_save_labels = os.path.join(folder_save_to, 'labels')
    check_folder(folder_to_save_images)
    check_folder(folder_to_save_labels)
    if yolor:
        get_predictions = get_valid_predictions_yolor
    else:
        get_predictions = get_valid_predictions_yolov4

    for pred_name in files_pred:
        path_file = os.path.join(path_predictions, pred_name)
        # pred = np.loadtxt(path_file, dtype=np.float32, delimiter=' ')
        # pred = np.loadtxt(path_file, dtype=np.float32)
        pred = np.loadtxt(path_file).reshape([-1, 6])
        valid_predictions = get_predictions(pred, threshold)
        if len(valid_predictions) > 0:
            move_image(pred_name, path_images, folder_to_save_images, ext=extension, delete=delete_files)
            move_prediction(pred_name, valid_predictions, folder_to_save_labels, delete=delete_files)


if __name__ == '__main__':
    path_images = '/home/luis/datasets/minneapple/train4/images'
    # path_images = '/home/luis/2021/Data_augmentation/Self_training/wider_self_train/images'
    path_predictions = '/home/luis/2021/yolov4_concept_testing/output_val' # yolov4
    # path_predictions = '/home/luis/2021/yolor/runs/test/exp/labels' # yolor
    folder_save_to = '/home/luis/datasets/minneapple/train3'
    # folder_save_to = '/home/luis/datasets/wider/wider_train3'

    threshold = 0.36
    delete = False
    yolor = False
    ext = '.png'
    main(path_images, path_predictions, folder_save_to, threshold=threshold, delete_files=delete,extension=ext,yolor=yolor)
