import torch
from transformers import AutoModel

from Net.header import DenseNet
from Net.vision_encoder import _3D_ResNet_50, get_pretrained_Vision_Encoder, pretrained_Resnet
from Net.fusions import SelfAttention
from Net.radiomic_encoder import *
import torch.nn as nn


class Vis_only(nn.Module):
    def __init__(self, use_pretrained = False):
        super(Vis_only, self).__init__()
        self.name = 'Vis_only'
        if use_pretrained:
            self.Resnet = get_pretrained_Vision_Encoder()
        else:
            self.Resnet = _3D_ResNet_50()
        self.output = nn.Linear(400, 2)

    def forward(self, x):
        '''
        :param x: torch.Size([B, 1, 64, 512, 512])
        :return: torch.Size([B, 2])
        '''
        x = self.Resnet(x)

        return self.output(x)


class Vis_only_header(nn.Module):
    def __init__(self):
        super(Vis_only_header, self).__init__()
        self.name = 'Vis_only_header'
        self.Resnet = _3D_ResNet_50()
        self.classify_head = DenseNet(layer_num=(6, 12, 24, 16), growth_rate=32, in_channels=1, classes=2)

    def forward(self, x):
        '''
        :param x: torch.Size([B, 1, 64, 512, 512])
        :return: torch.Size([B, 2])
        '''
        x = self.Resnet(x)
        x = torch.unsqueeze(x, dim=1)
        return self.classify_head(x)


class Text_only_header(nn.Module):
    def __init__(self):
        super(Text_only_header, self).__init__()
        self.name = 'Text_only_header'
        self.bert = AutoModel.from_pretrained("./models/Bio_ClinicalBERT")
        self.classify_head = DenseNet(layer_num=(6, 12, 24, 16), growth_rate=32, in_channels=1, classes=2)

    def forward(self, input_ids, attention_mask, token_type_ids):
        '''
        :param input_ids, attention_mask, token_type_ids: dict(3,), input_ids, attention_mask from Bert
        :return: torch.Size([B, 2])
        b = encoder(a['input_ids'], attention_mask=a['attention_mask'])
        print(b.last_hidden_state.shape)
        print(b.pooler_output.shape) -> set as text_feature
        '''
        x = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids).pooler_output
        x = torch.unsqueeze(x, dim=1)
        return self.classify_head(x)


class Fusion_Concat(nn.Module):
    def __init__(self):
        super(Fusion_Concat, self).__init__()
        self.name = 'Fusion_base'
        self.bert = AutoModel.from_pretrained("./models/Bio_ClinicalBERT")
        self.Resnet = _3D_ResNet_50()
        self.classify_head = DenseNet(layer_num=(6, 12, 24, 16), growth_rate=32, in_channels=1, classes=2)

    def forward(self, input_ids, attention_mask, token_type_ids, img):
        '''
        :param tokens_with_mask: input_ids, attention_mask<torch.Size([B, n])> from Bert
        :param img: torch.Size([B, 1, 64, 512, 512])
        :return: torch.Size([B, 2])
        b = encoder(tokens_with_mask['input_ids'], attention_mask=tokens_with_mask['attention_mask'])
        b.pooler_output -> set as text_feature
        '''
        text_feature = self.bert(input_ids=input_ids, attention_mask=attention_mask,
                                 token_type_ids=token_type_ids).pooler_output
        vision_feature = self.Resnet(img)
        global_feature = torch.cat((text_feature, vision_feature), dim=1)
        global_feature = torch.unsqueeze(global_feature, dim=1)
        return self.classify_head(global_feature)


class Fusion_SelfAttention(nn.Module):
    def __init__(self):
        super(Fusion_SelfAttention, self).__init__()
        self.name = 'Fusion_SelfAttention'
        self.bert = AutoModel.from_pretrained("./models/Bio_ClinicalBERT")
        self.Resnet = _3D_ResNet_50()
        self.SA = SelfAttention(16, 1280, 1280, hidden_dropout_prob=0.2)
        self.classify_head = DenseNet(layer_num=(6, 12, 24, 16), growth_rate=32, in_channels=1, classes=2)
        self.fc_text = nn.Linear(768, 640)
        self.fc_vis = nn.Linear(400, 640)

    def forward(self, input_ids, attention_mask, token_type_ids, img):
        '''
        :param tokens_with_mask: input_ids, attention_mask<torch.Size([B, n])> from Bert -> output[B, 768]
        :param img: torch.Size([B, 1, 64, 512, 512]) -> output[B, 768]
        :return: torch.Size([B, 2]) -> output[B, 400]
        b = encoder(tokens_with_mask['input_ids'], attention_mask=tokens_with_mask['attention_mask'])
        b.pooler_output -> set as text_feature
        '''
        text_feature = self.bert(input_ids=input_ids, attention_mask=attention_mask,
                                 token_type_ids=token_type_ids).pooler_output
        vision_feature = self.Resnet(img)
        text_feature = self.fc_text(text_feature)
        vision_feature = self.fc_vis(vision_feature)
        global_feature = torch.cat((text_feature, vision_feature), dim=1)
        global_feature = torch.unsqueeze(global_feature, dim=1)
        global_feature = self.SA(global_feature)
        return self.classify_head(global_feature)
    
    
class Contrastive_Learning(nn.Module):
    def __init__(self):
        super(Contrastive_Learning, self).__init__()
        self.name = 'Contrastive_Learning'
        self.Radio_encoder = Radiomic_encoder(num_features=1783)
        # self.Resnet = _3D_ResNet_50()
        self.Resnet = get_pretrained_Vision_Encoder()
        self.projection_head_radio = nn.Sequential(
                            nn.Linear(512, 128, bias = False),
                            nn.BatchNorm1d(128),
                            nn.ReLU(inplace=True),
                            nn.Linear(128, 128, bias = False)
                            ) 
        
        self.projection_head_vision = nn.Sequential(
                            nn.Linear(400, 128, bias = False),
                            nn.BatchNorm1d(128),
                            nn.ReLU(inplace=True),
                            nn.Linear(128, 128, bias = False)
                            ) 
        
    def forward(self, radio, img):
        '''
        :param radio: torch.Size([B, 1783]) 
        :param img: torch.Size([B, 1, 64, 512, 512]) 
        :return: radiomic_feature: torch.Size([B, 128])   vision_feature: torch.Size([B, 128])
        '''
        radiomic_feature = self.Radio_encoder(radio)
        vision_feature = self.Resnet(img)
        radiomic_feature = self.fc_radio(radiomic_feature)
        vision_feature = self.fc_vis(vision_feature)

        return radiomic_feature, vision_feature


"""
This function has been deprecated, please refer to train.py for more information.
"""
# class Contrastive_Learning(nn.Module):
#     def __init__(self):
#         super(Contrastive_Learning, self).__init__()
#         self.name = 'Contrastive_Learning'
#         self.Radio_encoder = Radiomic_encoder(num_features=1783)
#         # self.Resnet = _3D_ResNet_50()
#         self.Resnet = get_pretrained_Vision_Encoder()
#         self.projection_head_radio = nn.Sequential(
#                             nn.Linear(512, 128, bias = False),
#                             nn.BatchNorm1d(128),
#                             nn.ReLU(inplace=True),
#                             nn.Linear(128, 128, bias = False)
#                             )

#         self.projection_head_vision = nn.Sequential(
#                             nn.Linear(400, 128, bias = False),
#                             nn.BatchNorm1d(128),
#                             nn.ReLU(inplace=True),
#                             nn.Linear(128, 128, bias = False)
#                             )

#     def forward(self, radio, img):
#         '''
#         :param radio: torch.Size([B, 1783])
#         :param img: torch.Size([B, 1, 64, 512, 512])
#         :return: radiomic_feature: torch.Size([B, 128])   vision_feature: torch.Size([B, 128])
#         '''
#         radiomic_feature = self.Radio_encoder(radio)
#         vision_feature = self.Resnet(img)
#         radiomic_feature = self.fc_radio(radiomic_feature)
#         vision_feature = self.fc_vis(vision_feature)

#         return radiomic_feature, vision_feature

"""
Integration of radiomics and deep learning features utilizing a self-supervised trained encoder. (self-attention)
"""


class Fusion_radio_img(nn.Module):
    def __init__(self):
        super(Fusion_radio_img, self).__init__()
        self.name = 'Fusion_radio_img'

        self.Radio_encoder = Radiomic_encoder(num_features=1782)
        radio_state_dict = torch.load("./logs/classification/2024-04-12_15-10/checkpoints/radio_model_best.pth")
        self.Radio_encoder.load_state_dict(radio_state_dict)
        # 冻结Radiomic编码器的参数
        for param in self.Radio_encoder.parameters():
            param.requires_grad = False
        # 去除投影头
        self.Radio_encoder.projection_head = nn.Identity()

        self.Resnet = pretrained_Resnet()
        resnet_state_dict = torch.load("./logs/classification/2024-04-12_15-10/checkpoints/img_model_best.pth")
        self.Resnet.load_state_dict(resnet_state_dict)
        # 冻结Resnet的参数
        for param in self.Resnet.parameters():
            param.requires_grad = False
        self.Resnet.projection_head = nn.Identity()

        self.fc_Radio = nn.Linear(512, 256)
        self.fc_img = nn.Linear(400, 256)
        self.SA = SelfAttention(16, 512, 512, hidden_dropout_prob=0.2)
        self.classify_head = DenseNet(layer_num=(6, 12, 24, 16), growth_rate=32, in_channels=1, classes=2)

    def forward(self, radio, img):
        '''
        :param radio: torch.Size([B, 1782])
        :param img: torch.Size([B, 1, 64, 512, 512])
        :return: torch.Size([B, 2])
        '''
        radiomic_feature = self.Radio_encoder(radio)[0]
        vision_feature = self.Resnet(img)[0]
        radiomic_feature = self.fc_Radio(radiomic_feature)
        vision_feature = self.fc_img(vision_feature)
        global_feature = torch.cat((radiomic_feature, vision_feature), dim=1)
        global_feature = torch.unsqueeze(global_feature, dim=1)
        global_feature = self.SA(global_feature)
        return self.classify_head(global_feature)


"""
try to fusion radiomic,img and text.
Still coding....
"""


# class Fusion_Main(nn.Module):
#     def __init__(self):
#         super(Fusion_Main, self).__init__()
#         self.name = 'Fusion_Main'
#         self.extract_model = Contrastive_Learning().load_state_dict(torch.load("./models/liver/vision_radiomic_model.pth")['state_dict'] , strict=False)
#         self.fc_radio = nn.Linear(256, 320)
#         self.fc_text = nn.Linear(768, 320)
#         self.fc_vis = nn.Linear(400, 320)
#         self.SA = SelfAttention(16, 1280, 1280, hidden_dropout_prob=0.2)
#         self.classify_head = DenseNet(layer_num=(6, 12, 24, 16), growth_rate=32, in_channels=1, classes=2)

#     def forward(self, radio, input_ids, attention_mask, token_type_ids, img):

#         radiomic_feature, vision_feature = self.extract_model(radio, img)

#         text_feature = self.bert(input_ids=input_ids, attention_mask=attention_mask,
#                                  token_type_ids=token_type_ids).pooler_output

#         global_feature = torch.cat((radiomic_feature, vision_feature, text_feature), dim=1)
#         global_feature = torch.unsqueeze(global_feature, dim=1)
#         global_feature = self.SA(global_feature)
#         return self.classify_head(global_feature)


class Radio_only_Mamba(nn.Module):
    def __init__(self):
        super(Radio_only_Mamba, self).__init__()
        self.name = 'Radiomic_only with Mamba'
        self.mamba_block = Radiomic_mamba_encoder(num_features=1775)
        self.classify_head = DenseNet(layer_num=(6, 12, 24, 16), growth_rate=32, in_channels=1, classes=2)

    def forward(self, radio):
        mamba_output = self.mamba_block(radio)
        feature = torch.unsqueeze(mamba_output, dim=1)
        return self.classify_head(feature)


class Radio_only_SA(nn.Module):
    def __init__(self):
        super(Radio_only_SA, self).__init__()
        self.name = 'Radiomic_only with SelfAttention'
        self.projection1 = nn.Sequential(
            nn.Linear(1775, 2048, bias=False),
            nn.BatchNorm1d(2048),
        )
        self.SA = SelfAttention(16, 2048, 2048, hidden_dropout_prob=0.2)
        self.classify_head = DenseNet(layer_num=(6, 12, 24, 16), growth_rate=32, in_channels=1, classes=2)

    def forward(self, radio):
        radio = self.projection1(radio)
        radio = torch.unsqueeze(radio, dim=1)
        feature = self.SA(radio)
        return self.classify_head(feature)