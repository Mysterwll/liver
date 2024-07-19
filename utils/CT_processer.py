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


def get_dicom_series_shape(directory):
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(directory)
    reader.SetFileNames(dicom_names)
    try:
        image = reader.Execute()
    except:
        print("error in" + directory)
        return [64, 512, 512]
    return image.GetSize()

def process_ROI2sitkImage(file_path, Slice_Uids, data_dict, depth):
    shape = get_dicom_series_shape(file_path)
    _2d_shape = shape[:2]
    shape = (shape[2], shape[0], shape[1])
    if _2d_shape != (512, 512):
        print("warning: file's shape" + str(_2d_shape))
    # read & reset source_dcm_arrey
    deep = -1
    ROI_arrey = np.full(shape, -1024)
    for item in os.listdir(file_path):
        if not item.endswith('.dcm'):
            continue
        deep += 1
        try:
            image = sitk.ReadImage(file_path + '/' + item)
        except:
            print("Read error in" + file_path + '/' + item)
        source_arrey = sitk.GetArrayFromImage(image)
        img = np.array(source_arrey[0])
        img = img.astype('int8')
        cv2.imshow("source", img)
        slice_uid = image.GetMetaData("0008|0018")
        if slice_uid in Slice_Uids:
            # Find the uid corresponding to the slice
            for item in data_dict['ai_annos'][0]['groups'][0]['imgs']:
                if item['instanceUid'] == slice_uid:
                    contour = []
                    for position in item['paths'][0]:
                        # print((position['x'], position['y']))
                        contour.append((int(position['x']), int(position['y'])))
                    mask = np.full(_2d_shape, 0, dtype=np.uint8)
                    contour = np.array([contour])
                    cv2.drawContours(mask, [contour], -1, 255, thickness=-1)
                    cv2.imshow("mask", mask)
                    cv2.waitKey(-1)
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


def pre_prosses_format(fold_path="D:/2024/code2024/raw_data/src_dicom", target_depth="../data/new_img",
                       json_fold="D:/2024/code2024/raw_data/ROIliver"):
    assert os.path.exists(fold_path) and os.path.exists(json_fold), "please ensure paths exist"
    if not os.path.exists(target_depth):
        os.mkdir(target_depth)
    # 确保CT预处理流程在文本信息之后，以指定需要处理的数据
    summery = open("../data/summery_new.txt", 'r')
    summery.readline()
    temp_ids = []
    json_files = os.listdir(json_fold)
    json_dict = {}
    for file in json_files:
        _split = file.index('(')
        uid = file[:_split]
        json_dict.update({uid: json_fold + '/' + file})
    # 根据数据处理的规则，summery中指定ROI的为患者uid，指定CT的为srcid
    for item in summery:
        temp_ids.append([item.split()[0], item.split()[1]])

    for uid, srcid in tqdm(temp_ids, desc="Processing files", unit="file"):
        source_fold = fold_path + '/' + srcid
        # Step1. 依据标签进行ROI的分割，这里偷懒就用以前的方法了
        # 提取ROI信息
        _slice_ids, roi_data = get_ROI(json_path=json_dict[uid])
        depth = len(os.listdir(source_fold))
        temp_image = process_ROI2sitkImage(source_fold, _slice_ids, roi_data, depth)
        # Step2. 窗选
        temp_image = window_choose(temp_image)
        # Step3. 体素重采样，Average voxel size: [0.75005407 0.75005407 4.90774803]
        # Mean shape of DICOM series: [509.13103448 510.12413793  73.25172414]

        target_spacing = [0.75, 0.75, 5.00]
        target_shape = (512, 512, 64)
        resample = sitk.ResampleImageFilter()
        resample.SetOutputSpacing(target_spacing)
        resample.SetSize(target_shape)
        resample.SetOutputDirection(temp_image.GetDirection())
        resample.SetOutputOrigin(temp_image.GetOrigin())
        resample.SetInterpolator(sitk.sitkLinear)

        resampled_image = resample.Execute(temp_image)
        sitk.WriteImage(resampled_image, target_depth + '/' + "test" + '.nii.gz')


# debug
if __name__ == '__main__':
    pre_prosses_format()
