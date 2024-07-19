import os
from transformers import AutoTokenizer, AutoModel
import numpy as np
import torch
import torch.utils.data
import SimpleITK as sitk
import pickle


class Liver_dataset(torch.utils.data.Dataset):
    def __init__(self, summery_path: str = 'data/summery_new.txt', mode: str = 'mamba_test'):
        print("Dataset init ...")
        self.data_dict = {}  # all details of data
        self.data_list = []  # all uids of data
        self.data = []  # all value of data
        self.mode = mode
        if self.mode == 'bert':
            self.tokenizer = AutoTokenizer.from_pretrained("./models/Bio_ClinicalBERT")
        summery = open(summery_path, 'r')
        titles = summery.readline().split()
        titles = [title.replace('_', ' ') for title in titles]

        # print(titles[60])
        count = 0
        for item in summery:
            count += 1
            single_data_list = item.split()
            temp_dict = {}
            temp = {}
            for i in range(len(single_data_list)):
                # try:
                temp.update({titles[i]: single_data_list[i]})
                # except:
                #     print(f"Debug infos: uid{single_data_list[0]} titles{len(titles)} len{len(single_data_list)}")
                # print("insert : " + titles[i] + " : " + single_data_list[i])
            uid = temp.pop('uid')
            srcid = temp.pop('srcid')
            label = temp.pop('Outcome')
            string = ' '.join([f"{key}: {value}" for key, value in temp.items()])
            _data = [float(x) for x in list(temp.values())]
            temp_dict.update({'srcid': srcid})
            temp_dict.update({'label': label})
            if self.mode == 'bert':
                temp_dict.update({'features': string})
            elif self.mode == 'fusion':
                temp_dict.update({'features': string})
            else:
                temp_dict.update({'features': _data})

            # temp_dict.update({'source': temp})

            self.data_dict.update({uid: temp_dict})
            self.data_list.append(uid)
            self.data.append(_data)
        # Normalization

        print("Summery loaded --> Feature_num : %d  Data_num : %d" % (len(titles) - 3, count))
        summery.close()

    def __getitem__(self, index):
        # To locate data
        uid = self.data_list[index]

        srcid = self.data_dict[uid]['srcid']
        text_feature = self.data_dict[uid]['features']

        img = sitk.ReadImage('./data/img/' + srcid + ".nii.gz")
        array = sitk.GetArrayFromImage(img)
        vision = torch.Tensor(array)
        vision_tensor = torch.unsqueeze(vision, 0)

        # usage of text data
        clinical = self.data_dict[uid]['features'][:-1781]
        clinical_tensor = torch.Tensor(clinical)
        clinical_tensor = torch.where(torch.isnan(clinical_tensor), torch.full_like(clinical_tensor, 0),
                                      clinical_tensor)
        radio = self.data_dict[uid]['features'][-1781:]
        radio_tensor = torch.Tensor(radio)
        radio_tensor = torch.where(torch.isnan(radio_tensor), torch.full_like(radio_tensor, 0), radio_tensor)

        label = int(self.data_dict[uid]['label'])
        label_tensor = torch.from_numpy(np.array(label)).long()
        if self.mode == 'bert':
            text_tensor = self.text2id(text_feature)
            return text_tensor['input_ids'].squeeze(0), text_tensor['token_type_ids'].squeeze(0), text_tensor[
                'attention_mask'].squeeze(0), label_tensor
        elif self.mode == 'fusion':
            text_tensor = self.text2id(text_feature)
            return text_tensor['input_ids'].squeeze(0), text_tensor['token_type_ids'].squeeze(0), text_tensor[
                'attention_mask'].squeeze(0), vision_tensor, label_tensor
        elif self.mode == 'img':
            return vision_tensor, label_tensor
        elif self.mode == 'radio':
            return radio_tensor, label_tensor
        elif self.mode == 'self_supervised':
            return radio_tensor, vision_tensor
        elif self.mode == 'mamba_test':
            return clinical_tensor, label_tensor
        elif self.mode == 'radio_img_label':
            return radio_tensor, vision_tensor, label_tensor
        else:
            return None

    def __len__(self):
        return len(self.data_list)

    def text2id(self, batch_text):
        return self.tokenizer(batch_text, max_length=512,
                              truncation=True, padding='max_length', return_tensors='pt')

    def normalization(self):
        data = torch.Tensor(self.data)
        mean = torch.mean(data, dim=0)
        std = torch.std(data, dim=0)
        normalized_dataset = (data - mean) / std
        with open('data/mean.pkl', 'wb') as f:
            pickle.dump(mean, f)
        with open('data/std.pkl', 'wb') as f:
            pickle.dump(std, f)
        with open('data/normalized_dataset.pkl', 'wb') as f:
            pickle.dump(normalized_dataset, f)
        print(mean)
        print(std)
        print(normalized_dataset)


class Liver_normalization_dataset(torch.utils.data.Dataset):
    def __init__(self, summery_path: str = 'data/summery_new.txt', mode: str = 'all_model'):
        print("Dataset(norm) init ...")
        self.data_dict = {}  # all details of data
        self.data_list = []  # all uids of data
        self.mode = mode
        self.tokenizer = AutoTokenizer.from_pretrained("./models/Bio_ClinicalBERT")
        if os.path.exists("data/mean.pkl"):
            with open('data/mean.pkl', 'rb') as f:
                self.mean = pickle.load(f)
            with open('data/std.pkl', 'rb') as f:
                self.std = pickle.load(f)
                self.std = np.where(self.std == 0, 1, self.std)
        else:
            _ = Liver_dataset('data/summery_new.txt')
            _.normalization()
            print("please relaunch")
            exit()
        summery = open(summery_path, 'r')
        titles = summery.readline().split()
        titles = [title.replace('_', ' ') for title in titles]

        count = 0
        for item in summery:
            count += 1
            single_data_list = item.split()
            temp_dict = {}
            temp = {}
            temp_clinical = {}
            for i in range(len(single_data_list)):
                temp.update({titles[i]: single_data_list[i]})

            uid = temp.pop('uid')
            srcid = temp.pop('srcid')
            label = temp.pop('Outcome')
            
            temp_clinical = {k: temp[k] for k in list(temp.keys())[:-1781]}
            clinical_text = ' '.join([f"{key}: {value}" for key, value in temp_clinical.items()])
            string = ' '.join([f"{key}: {value}" for key, value in temp.items()])
            # normalization
            _data = torch.Tensor([float(x) for x in list(temp.values())])
            _data = (_data - self.mean) / self.std
            temp_dict.update({'srcid': srcid})
            temp_dict.update({'label': label})
            if self.mode == 'bert':
                temp_dict.update({'features': string})
            elif self.mode == 'fusion' or 'two_model':
                temp_dict.update({'features': _data, 'clinical_text': clinical_text})
            else:
                temp_dict.update({'features': _data})

            self.data_dict.update({uid: temp_dict})
            # UID: dataset's index to identify UID, then use UID to locate data in self.data_dict[uid]
            self.data_list.append(uid)

        print("Summery loaded --> Feature_num : %d  Data_num : %d" % (len(titles) - 3, count))
        summery.close()

    def __getitem__(self, index):
        # To locate data
        uid = self.data_list[index]

        # usage of vision data
        srcid = self.data_dict[uid]['srcid']
        img = sitk.ReadImage('./data/new_img/' + srcid + ".nii.gz")
        array = sitk.GetArrayFromImage(img)
        vision = torch.Tensor(array)
        vision_tensor = torch.unsqueeze(vision, 0)

        if self.mode == 'fusion' or 'two_model':
            text_feature = self.data_dict[uid]['clinical_text']
        
        # usage of text data
        clinical = self.data_dict[uid]['features'][:-1781]
        clinical_tensor = torch.Tensor(clinical)
        clinical_tensor = torch.where(torch.isnan(clinical_tensor), torch.full_like(clinical_tensor, 0), clinical_tensor)

        radio = self.data_dict[uid]['features'][-1781:]
        radio_tensor = torch.Tensor(radio)
        radio_tensor = torch.where(torch.isnan(radio_tensor), torch.full_like(radio_tensor, 0), radio_tensor)

        label = int(self.data_dict[uid]['label'])
        label_tensor = torch.from_numpy(np.array(label)).long()
        if self.mode == 'img':
            return vision_tensor, label_tensor
        elif self.mode == 'self_supervised':
            return radio_tensor, vision_tensor
        elif self.mode == 'mamba_test':
            return radio_tensor, label_tensor
        elif self.mode == 'radio_img_label':
            return radio_tensor, vision_tensor, label_tensor
        elif self.mode == 'all_model':
            return clinical_tensor, radio_tensor, vision_tensor, label_tensor
        elif self.mode == 'fusion':
            text_tensor = self.text2id(text_feature)
            return text_tensor['input_ids'].squeeze(0), text_tensor['token_type_ids'].squeeze(0), text_tensor[
                'attention_mask'].squeeze(0), radio_tensor, vision_tensor, label_tensor
        elif self.mode == 'two_model':
            text_tensor = self.text2id(text_feature)
            return text_tensor['input_ids'].squeeze(0), text_tensor['token_type_ids'].squeeze(0), text_tensor[
                'attention_mask'].squeeze(0), vision_tensor, label_tensor
        elif self.mode == 'two_textmodel':
            text_tensor = self.text2id(text_feature)
            return text_tensor['input_ids'].squeeze(0), text_tensor['token_type_ids'].squeeze(0), text_tensor[
                'attention_mask'].squeeze(0), radio_tensor, label_tensor
        else:
            return None
        
    def text2id(self, batch_text):
        return self.tokenizer(batch_text, max_length=512,
                              truncation=True, padding='max_length', return_tensors='pt')
    def __len__(self):
        return len(self.data_list)

