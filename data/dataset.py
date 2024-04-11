import os
from transformers import AutoTokenizer, AutoModel
import numpy as np
import torch
import torch.utils.data
import SimpleITK as sitk


class Liver_dataset(torch.utils.data.Dataset):
    def __init__(self, summery_path: str, mode: str = 'bert'):
        print("Dataset init ...")
        self.data_dict = {}  # all details of data
        self.data_list = []  # all uids of data
        self.data = []       # all value of data
        self.mode = mode
        self.tokenizer = AutoTokenizer.from_pretrained("./models/Bio_ClinicalBERT")
        summery = open(summery_path, 'r')
        titles = summery.readline().split()
        titles = [title.replace('_', ' ') for title in titles]

        # print(titles[60])
        count = 0
        for item in summery:
            count += 1
            single_data_list = item.split()
            data = [float(x) for x in single_data_list]
            temp_dict = {}
            temp = {}
            for i in range(len(single_data_list)):
                try:
                    temp.update({titles[i]: single_data_list[i]})
                except:
                    print(f"Debug infos: uid{single_data_list[0]} titles{len(titles)} len{len(single_data_list)}")
                # print("insert : " + titles[i] + " : " + single_data_list[i])
            uid = temp.pop('uid')
            srcid = temp.pop('srcid')
            label = temp.pop('Outcome')
            string = ' '.join([f"{key}: {value}" for key, value in temp.items()])

            temp_dict.update({'srcid': srcid})
            temp_dict.update({'label': label})
            if self.mode == 'bert':
                temp_dict.update({'features': string})
            elif self.mode == 'fusion':
                temp_dict.update({'features': string})
            else:
                temp_dict.update({'features': list(temp.values())})

            # temp_dict.update({'source': temp})

            self.data_dict.update({uid: temp_dict})
            self.data_list.append(uid)
            self.data.append(data)
        
        print("Summery loaded --> Feature_num : %d  Data_num : %d" % (len(titles) - 3, count))

        summery.close()

    def __getitem__(self, index):
        uid = self.data_list[index]

        srcid = self.data_dict[uid]['srcid']
        text_feature = self.data_dict[uid]['features']
        img = sitk.ReadImage('./data/img/' + srcid + ".nii.gz")
        array = sitk.GetArrayFromImage(img)
        vision = torch.Tensor(array)
        vision_tensor = torch.unsqueeze(vision, 0)

        radio = self.data[index][60:1843]
        radio_tensor = torch.Tensor(radio)

        label = int(self.data_dict[uid]['label'])
        label_tensor = torch.from_numpy(np.array(label)).long()
        if self.mode == 'bert':
            text_tensor = self.text2id(text_feature)
            return text_tensor['input_ids'].squeeze(0), text_tensor['token_type_ids'].squeeze(0), text_tensor['attention_mask'].squeeze(0), label_tensor
        elif self.mode == 'fusion':
            text_tensor = self.text2id(text_feature)
            return text_tensor['input_ids'].squeeze(0), text_tensor['token_type_ids'].squeeze(0), text_tensor['attention_mask'].squeeze(0), vision_tensor, label_tensor
        elif self.mode == 'img':
            return None, label_tensor
        elif self.mode == 'self_supervised':
            return radio_tensor, vision_tensor
        elif self.mode == 'radio_img_label':
            return radio_tensor, vision_tensor, label_tensor
        elif self.mode == 'fusion_2':
            return text_tensor['input_ids'].squeeze(0), text_tensor['token_type_ids'].squeeze(0), text_tensor['attention_mask'].squeeze(0), radio_tensor, vision_tensor, label_tensor
        else:
            return None

    def __len__(self):
        return len(self.data_list)

    def text2id(self, batch_text):
        return self.tokenizer(batch_text, max_length=512,
                              truncation=True, padding='max_length', return_tensors='pt')





