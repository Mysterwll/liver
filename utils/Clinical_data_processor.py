import os
import win32com
from tqdm import tqdm
from win32com import client
import pandas as pd

'''
This file is used for preprocessing the private dataset of this study,
and is not compatible when using this code to process your dataset.
This code is for reference only.
'''
SOURCE_PATH = "D:/STUDY/2024/code2024/raw_data"
TARGET_PATH = "D:/STUDY/2024/code2024/data"


def overwrite_data(cli_data, fat_data, radio_data, summery):
    # set title
    title = ("uid srcid Age Gender HBV HCV Alcoholic PBC AIH NAFLD DILI Outcome Sarcopenia grip_strength Upper_arm "
             "lower_leg triceps_skinfold_thickness upper_arm_muscle "
             "nutritional_risk nutritional_score height weight BMI stride WBC RBC HB "
             "PLT neutrophils lymphocytes ALT AST ALP GGT "
             "protein albumin Total_Bilirubin Direct_Bilirubin bileacid cholinesterase "
             "triglycerides cholesterol HDL LDL urea creatinine "
             "cystatin "
             # fat
             "SAT_Volume SAT_Area "
             "IMAT_Volume IMAT_Area IMAT_Volume_Fat_Percentage IMAT_Area_Fat_Percentage "
             "VAT_Volume VAT_Area VAT_Volume_Fat_Percentage VAT_Area_Fat_Percentage "
             "Soft_Tissue_Volume Soft_Tissue_Area Soft_Tissue_Volume_Fat_Percentage Soft_Tissue_Area_Fat_Percentage "
             # radio
             "Peripheral_Circumference Major_Axis Minor_Axis Mean_Gray_Value Median_Gray_Value Physical_Size\n")
    log = open('./log.txt', 'w')
    # 输出到控制台
    print("title_num: %d" % len(title.split()))
    titles = title.split(' ')
    print("check: %d" % len(titles))
    summery.write(title)
    # 非常抽象的取数据，具体请见表格
    for i in tqdm(range(212), desc="Data Processing"):
        flag = True
        temp_line = ""
        Uid = cli_data.Cells(3 + i, 2).Text
        temp_line += str(cli_data.Cells(3 + i, 2).Text) + " "
        for j in range(3):
            temp_line, flag = _write_without_None(temp_line, cli_data, 3 + i, 3 + j, flag, log)
        temp_list = cli_data.Cells(3 + i, 7).Text.split()
        int_list = [int(num) for num in temp_list]
        for j in range(1, 8):
            if j in int_list:
                temp_line += "1" + " "
            else:
                temp_line += "0" + " "
        temp_line, flag = _write_without_None(temp_line, cli_data, 3 + i, 10, flag, log)
        for j in range(35):
            temp_line, flag = _write_without_None(temp_line, cli_data, 3 + i, 17 + j, flag, log)
        # find uid
        line_num = -1
        for j in range(212):
            if fat_data.Cells(2 + j, 3).Text == Uid:
                line_num = 2 + j
                break
        temp_line, flag = _write_without_None(temp_line, fat_data, line_num, 4, flag, log)
        temp_line, flag = _write_without_None(temp_line, fat_data, line_num, 5, flag, log)

        for j in range(12):
            temp_line, flag = _write_without_None(temp_line, fat_data, line_num, 8 + j, flag, log)

        # Radio
        line_num = -1
        for j in range(212):
            if radio_data.Cells(2 + j, 2).Text == Uid:
                line_num = 2 + j
                break
        for j in range(6):
            temp_line, flag = _write_without_None(temp_line, radio_data, line_num, 8 + j, flag, log)

        temp_line += '\n'
        if flag:
            summery.write(temp_line)
        else:
            print("delete -> uid:" + str(Uid))
    log.close()
    summery.close()
    print("check")
    pass


def add_radiomics():
    src = open('../data/summery.txt', 'r')
    dst = open('../data/summery_new.txt', 'w')
    radiomics = pd.read_csv('../raw_data/radiomics/1/liver.csv', skiprows=1)
    radiomics.drop(radiomics.columns[0], axis=1, inplace=True)
    for i in range(25):
        radiomics.drop(radiomics.columns[1], axis=1, inplace=True)
    # print(radiomics.columns.to_list())
    # set title
    titles = src.readline().split()
    # print(titles)
    titles = titles[:-5] + radiomics.columns.to_list()[1:]
    new_titles = ' '.join(titles) + '\n'
    dst.write(new_titles)
    for item in src:
        src_list = item.split(' ')
        try:
            row_data = radiomics.loc[radiomics['patientId'] == str(src_list[0])].values.tolist()[0]
        except:
            print("error @ " + str(src_list[0]))
            continue

        new_row = src_list[:-5] + row_data[1:]
        new_data = ' '.join(new_row) + '\n'
        dst.write(new_data)
    src.close()
    dst.close()



def _write_without_None(temp_line, sheet, _i, _j, flag, log):
    temp = sheet.Cells(_i, _j).Text
    if not temp.strip():
        err_msg = "[debug]error_: Nan in " + sheet.Cells(1, _j).Text + "&" + sheet.Cells(2,
                                                                                         _j).Text + " -> line:%d" % _i
        print(err_msg)
        log.write(err_msg + "\n")
        flag = False
    else:
        temp_line += temp.split(' ')[0] + " "
    return temp_line, flag


def main_preprocess():
    assert os.path.exists(TARGET_PATH), "Please cd to the fold named 'utils!' and run the code"
    app = win32com.client.Dispatch('Excel.Application')
    app.Visible = 0
    app.DisplayAlerts = 0
    # open all excels
    WorkBook_Clinical_data = app.Workbooks.Open(SOURCE_PATH + "/Clinical_data/data.xlsx")
    WorkBook_Fat = app.Workbooks.Open(SOURCE_PATH + "/Fat_analysis/Fat.xlsx")
    WorkBook_Radio = app.Workbooks.Open(SOURCE_PATH + "/ROI/liverROI/20240112170059.xlsx")

    cli_data = WorkBook_Clinical_data.Worksheets('sheet1')
    fat_data = WorkBook_Fat.Worksheets('sheet1')
    radio_data = WorkBook_Radio.Worksheets('Roi')

    # target file
    summery = open(TARGET_PATH + "/summery.txt", 'w')

    # overwrite data
    overwrite_data(cli_data, fat_data, radio_data, summery)

    # read only
    WorkBook_Clinical_data.Close()
    WorkBook_Fat.Close()
    WorkBook_Radio.Close()


if __name__ == '__main__':
    add_radiomics()
