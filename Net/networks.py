import torch
from transformers import AutoModel

from Net.header import DenseNet
from Net.vision_encoder import _3D_ResNet_50
from Net.fusions import SelfAttention
import torch.nn as nn


class Vis_only(nn.Module):
    def __init__(self):
        super(Vis_only, self).__init__()
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
    

class Fusion_Cross_selfAttention(nn.Module):
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
