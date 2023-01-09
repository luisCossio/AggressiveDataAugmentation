import os
import argparse
import numpy as np
from tqdm import tqdm


def check_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)


def save_empty_file(path_file_empty):
    # str_file = ''
    open(path_file_empty, 'w').close()

def save_label(save_detections, path_save):
    str_file = ''
    for bb in save_detections:
        str_file += "{:d} {:.6f} {:.6f} {:.6f} {:.6f}\n".format(int(bb[0]), bb[1], bb[2], bb[3], bb[4])
    file1 = open(path_save, "w")
    # \n is placed to indicate EOL (End of Line)
    file1.write(str_file)
    file1.close()  # to change



def main(path_images: str, path_predictions: str,opt):
    ext = opt.ext
    threshold = opt.threshold
    path_save_labels = opt.path_labels
    images = os.listdir(path_images)
    check_folder(path_save_labels)
    check_folder(path_save_labels)
    predictions = os.listdir(path_predictions)
    n_predictions = len(predictions)
    n_images = len(images)
    print("Predictions and images: {:d}/{:d}".format(n_predictions,n_images))
    for i, img_name in tqdm(enumerate(images)):
        txt_file_name = img_name.replace(ext, '.txt')
        path_prediction = os.path.join(path_predictions, txt_file_name)
        path_save = os.path.join(path_save_labels, txt_file_name)
        if os.path.isfile(path_prediction):
            prediction = np.loadtxt(path_prediction)

            if len(prediction) > 0:
                prediction = prediction.reshape([-1,6])
                scores = prediction[:, 5]
                prediction = prediction[scores > threshold, :]
                if len(prediction) > 0:
                    # save_detections = np.empty([len(prediction),5],dtype=np.float)
                    save_detections = prediction[:, :5]
                    save_label(save_detections,path_save)
                else:
                    save_empty_file(path_save)
            else:
                save_empty_file(path_save)
        else:
            save_empty_file(path_save)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='PredYolo2LabelsCOCO.py',
                                     description=" Method to convert predictions into labels. Labels are in the "
                                                 " format YOLO:"
                                                 "class xc yc w h"
                                                 "EX: "
                                                 "0 0.44954819 0.04442771 0.03840361 0.04894578 "
                                                 ".... "
                                                 "Predictions are in the format: "
                                                 ""
                                                 "EX"
                                                 "0 0.0969896 0.669921 0.0268508 0.0398234 0.54834"
                                                 "0 0.00319776 0.798066 0.00632562 0.0166016 0.0184326"
                                                 "0 0.00335534 0.798882 0.0064518 0.0166016 0.00211716"
                                                 "0 0.227651 0.995117 0.0164717 0.00976562 0.00137329"
                                                 "0 0.0104445 0.609193 0.0208891 0.0400391 0.0012598")
    parser.add_argument('--path-dataset', type=str, help='path of the dataset',default='')
    parser.add_argument('--path-images', type=str, help='path to the images',default='')
    parser.add_argument('--path-predictions', type=str, help='path to folder with .txt predictions',default='')
    parser.add_argument('--path-labels', type=str, help='path to save .txt pseudo-labels',default='')
    parser.add_argument('--ext', type=str, help='extension of images',default='.JPG')
    parser.add_argument('--threshold', type=float, help='threshold to filter predictions',default=0.3)


    opt = parser.parse_args()


    if opt.path_dataset != '':
        if opt.path_images == '':
            opt.path_images = os.path.join(opt.path_dataset,'images')
        if opt.path_predictions == '':
            opt.path_predictions = os.path.join(opt.path_dataset,'pred')
        if opt.path_labels == '':
            opt.path_labels = os.path.join(opt.path_dataset,'labels')
    elif opt.path_images != '' and opt.path_predictions != '' and opt.path_labels != '':
        pass
    else:
        raise argparse.ArgumentError(opt.path_dataset, "Either argument path_dataset needs to be identified or the "
                                                       "rest need to be specified")
    main(opt.path_images, opt.path_predictions,opt)
