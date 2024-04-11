import torch
import torch.nn as nn
from torch.utils.data import random_split
from torchvision import models
from transformers import AutoTokenizer, AutoModel, AutoConfig

from Net.networks import *
from data.dataset import Liver_dataset
from transformers import BertModel, BertConfig, BertTokenizer
# config = AutoConfig.from_pretrained("./models/Bio_ClinicalBERT")
# bert = AutoModel.from_pretrained("./models/Bio_ClinicalBERT", config=config)

# dataset = Liver_dataset("./data/summery.txt", mode='fusion')
# testDataLoader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)
# for i, (token, segment, mask, img, labels) in enumerate(testDataLoader):
#     # print(i, token, segment, mask, labels)
#     # print(token.shape)
#     # print(segment.shape)
#     # print(mask.shape)
#     # print(labels.shape)
#     # output = bert(input_ids=token, attention_mask=mask, token_type_ids=segment).pooler_output
#     exit()

# tokenizer = AutoTokenizer.from_pretrained("./models/Bio_ClinicalBERT")
#
# for i, (text, label) in enumerate(testDataLoader):
#     print(text)
#     token_with_mask = tokenizer(text, padding='max_length',
#                                 max_length=512,
#                                 return_tensors="pt").to('cuda')
#     label = label.to('cuda')
#     print(text)
#     output = model(token_with_mask)
#     print("Output shape:", output.shape)
#     print("Output values:", output)
#     exit()

# # model = Vis_only()
# model = Fusion_SelfAttention().to('cuda')
# img = torch.randn((2, 1, 32, 224, 224)).to('cuda')
# tokenizer = AutoTokenizer.from_pretrained("./models/Bio_ClinicalBERT")
# text_bag = tokenizer(('hello world', 'Aloha world'), max_length=512, truncation=True, padding='max_length', return_tensors='pt')
# # text_bag['input_ids'].squeeze(0), text_bag['token_type_ids'].squeeze(0), text_bag['attention_mask'].squeeze(0)
# output = model(text_bag['input_ids'].to('cuda'), text_bag['token_type_ids'].to('cuda'), text_bag['attention_mask'].to('cuda'), img)
# # output = model(img)
# print(output)
# print(output.shape)
# # params = [p for p in model.parameters()]
# # print("step1: " + str(len(params)))


if __name__ =='__main__':

    from torch.utils.data import DataLoader
    import numpy as np

    # train_loader = DataLoader(Liver_dataset("./data/summery.txt", mode='self_supervised'), batch_size=1, shuffle=True, num_workers=4, drop_last=True)

    # for batch_idx, (radio, img) in enumerate(train_loader):
    #     print(f"batch_idx: {batch_idx}  | radio shape: {radio.shape} | ;img shape: {img.shape}")

    train_val_loader = DataLoader(Liver_dataset("./data/summery.txt", mode='radio_img_label'), batch_size=2, shuffle=True, num_workers=4, drop_last=True)
    labels_train = []
    for batch_idx, (radio, img, label) in enumerate(train_val_loader):
        print(f"batch_idx: {batch_idx}  | radio shape: {radio.shape} | img shape: {img.shape} | label shape: {label.shape} ")
        
        labels = label.numpy().tolist()
        labels_train += labels

    labels_train = np.array(labels_train)
    print(labels_train.shape)