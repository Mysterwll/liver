import os
import numpy as np
import SimpleITK as sitk
import json
import cv2
from tqdm import tqdm

'''
This file is used for preprocessing the private dataset of this study,
and is not compatible when using this code to process your dataset.
This code is for reference only.
'''


def pre_process_source(file_path, target_depth, json_path=None):
    depth = len(os.listdir(file_path))
    if json_path is not None:
        # get ROI_ids
        Slice_Uids, data_dict = get_ROI(json_path=json_path)
        # get ROI
        temp_image = process_ROI2sitkImage(file_path, Slice_Uids, data_dict, depth)
    else:
        # without ROI
        temp_image = process_SRC2sitkImage(file_path)
    # window_choose
    temp_image = window_choose(temp_image)
    # resample
    return resampleSize(temp_image, target_depth)


def process_SRC2sitkImage(file_path):
    return read_series(file_path)


def process_ROI2sitkImage(file_path, Slice_Uids, data_dict, depth):
    # read & reset source_dcm_arrey
    deep = -1
    ROI_arrey = np.full((depth, 512, 512), -1024)
    for item in os.listdir(file_path):
        deep += 1
        image = sitk.ReadImage(file_path + '/' + item)
        source_arrey = sitk.GetArrayFromImage(image)
        slice_uid = image.GetMetaData("0008|0018")
        if slice_uid in Slice_Uids:
            # Find the uid corresponding to the slice
            for item in data_dict['ai_annos'][0]['groups'][0]['imgs']:
                if item['instanceUid'] == slice_uid:
                    contour = []
                    for position in item['paths'][0]:
                        # print((position['x'], position['y']))
                        contour.append((int(position['x']), int(position['y'])))
                    mask = np.full((512, 512), 0, dtype=np.uint8)
                    contour = np.array([contour])
                    cv2.drawContours(mask, [contour], -1, 255, thickness=-1)
                    mask = np.transpose(np.where(mask))
                    for point in mask:
                        x = point[0]
                        y = point[1]
                        ROI_arrey[deep][x][y] = source_arrey[0][x][y]
    output = sitk.GetImageFromArray(ROI_arrey)
    image = read_series(file_path)
    output.CopyInformation(image)
    return output


def resampleSize(sitkImage, depth):
    euler3d = sitk.Euler3DTransform()
    xsize, ysize, zsize = sitkImage.GetSize()
    xspacing, yspacing, zspacing = sitkImage.GetSpacing()
    new_spacing_z = zspacing / (depth / float(zsize))
    # new_spacing_x = xspacing/(256/float(xsize))
    # new_spacing_y = yspacing/(256/float(ysize))
    origin = sitkImage.GetOrigin()
    direction = sitkImage.GetDirection()
    # based on new spacing calculate new size
    newsize = (xsize, ysize, depth)
    newspace = (xspacing, yspacing, new_spacing_z)
    # newsize = (256, 256, depth)
    # newspace = (new_spacing_x, new_spacing_y, new_spacing_z)
    sitkImage = sitk.Resample(sitkImage, newsize, euler3d, sitk.sitkNearestNeighbor, origin, newspace, direction)
    return sitkImage


def read_series(file_path):
    reader = sitk.ImageSeriesReader()
    dcm_series = reader.GetGDCMSeriesFileNames(file_path)
    reader.SetFileNames(dcm_series)
    img = reader.Execute()
    return img


def get_ROI(json_path):
    Slice_Uids = []
    with open(json_path, encoding="UTF-8") as f:
        data_dict = json.load(f)
    for item in data_dict['ai_annos'][0]['groups'][0]['imgs']:
        Slice_Uids.append(item['instanceUid'])
    return Slice_Uids, data_dict


def window_choose(sitk_image):
    intensityWindowingFilter = sitk.IntensityWindowingImageFilter()
    # set window -> (-70~130) for liver
    intensityWindowingFilter.SetWindowMaximum(130)
    intensityWindowingFilter.SetWindowMinimum(-70)
    intensityWindowingFilter.SetOutputMaximum(255)
    intensityWindowingFilter.SetOutputMinimum(0)
    sitk_image = intensityWindowingFilter.Execute(sitk_image)
    return sitk_image


# debug
if __name__ == '__main__':
    fold_path = "../raw_data/src_dicom"
    save_path = "../data/img"
    summery = open("../data/summery.txt", 'r')
    summery.readline()
    names = []
    for item in summery:
        names.append(item.split()[1])
    print(names)
    print(len(names))
    file_list = os.listdir(fold_path)
    for item in tqdm(file_list, desc="Processing files", unit="file"):
        if item in names:
            temp = pre_process_source(fold_path + '/' + item, 64)
            sitk.WriteImage(temp, save_path + '/' + item + ".nii.gz")
        else:
            print("ignore: " + item)
            continue



