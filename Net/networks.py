import torch
from transformers import AutoModel

from Net.basicArchs import *
from Net.mamba_modules import *
from Net.fusions import *
import torch.nn as nn


class Vis_only(nn.Module):
    def __init__(self, use_pretrained=True):
        super(Vis_only, self).__init__()
        self.name = 'Vis_only'
        if use_pretrained:
            self.Resnet = get_pretrained_Vision_Encoder()
        else:
            self.Resnet = M3D_ResNet_50()
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
        self.Resnet = M3D_ResNet_50()
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
        self.Resnet = M3D_ResNet_50()
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
        self.Resnet = M3D_ResNet_50()
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
        # self.Resnet = M3D_ResNet_50()
        self.Resnet = get_pretrained_Vision_Encoder()
        self.projection_head_radio = nn.Sequential(
            nn.Linear(512, 128, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128, bias=False)
        )

        self.projection_head_vision = nn.Sequential(
            nn.Linear(400, 128, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128, bias=False)
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
#         # self.Resnet = M3D_ResNet_50()
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

        self.Radio_encoder = Radiomic_encoder(num_features=1781)
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
        :param radio: torch.Size([B, 1781])
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


class Fusion_2stage(nn.Module):
    def __init__(self, radio_encoder_path, img_encoder_path):
        super(Fusion_2stage, self).__init__()
        self.name = 'Fusion_2stage'
        self.Radio_encoder = Radiomic_encoder(num_features=1781)
        radio_state_dict = torch.load(radio_encoder_path)
        self.Radio_encoder.load_state_dict(radio_state_dict)
        # 冻结Radiomic编码器的参数
        for param in self.Radio_encoder.parameters():
            param.requires_grad = False
        # 去除投影头
        self.Radio_encoder.projection_head = nn.Identity()

        self.Resnet = pretrained_Resnet()
        resnet_state_dict = torch.load(img_encoder_path)
        self.Resnet.load_state_dict(resnet_state_dict)
        # 冻结Resnet的参数
        for param in self.Resnet.parameters():
            param.requires_grad = False
        self.Resnet.projection_head = nn.Identity()

        self.bert = AutoModel.from_pretrained("./models/Bio_ClinicalBERT")
        # self.fc_Radio = nn.Linear(512, 256)
        # self.fc_img = nn.Linear(400, 256)
        self.fc_text = nn.Linear(768, 256)
        self.SA = SelfAttention(16, 512 + 400 + 256, 512 + 400 + 256, hidden_dropout_prob=0.2)
        self.classify_head = DenseNet(layer_num=(6, 12, 24, 16), growth_rate=32, in_channels=1, classes=2)

    def forward(self, input_ids, attention_mask, token_type_ids, radio, img):
        '''
        :param radio: torch.Size([B, 1781])
        :param img: torch.Size([B, 1, 64, 512, 512])
        :return: torch.Size([B, 2])
        '''
        with torch.no_grad():
            radiomic_feature = self.Radio_encoder(radio)[0]
            vision_feature = self.Resnet(img)[0]
        text_feature = self.bert(input_ids=input_ids, attention_mask=attention_mask,
                                 token_type_ids=token_type_ids).pooler_output

        # radiomic_feature = self.fc_Radio(radiomic_feature)
        # vision_feature = self.fc_img(vision_feature)
        text_feature = self.fc_text(text_feature)

        global_feature = torch.cat((radiomic_feature, vision_feature, text_feature), dim=1)
        global_feature = torch.unsqueeze(global_feature, dim=1)
        global_feature = self.SA(global_feature)
        return self.classify_head(global_feature)


class Fusion_Main(nn.Module):
    def __init__(self):
        super(Fusion_Main, self).__init__()
        self.name = 'Fusion_Main'
        self.Radio_encoder = Radiomic_encoder(num_features=1781)
        self.Radio_encoder.projection_head = nn.Identity()

        self.Resnet = get_pretrained_Vision_Encoder()

        self.bert = AutoModel.from_pretrained("./models/Bio_ClinicalBERT")
        self.radio_projection_head = nn.Sequential(
            nn.Linear(512, 128, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128, bias=False)
        )
        self.img_projection_head = nn.Sequential(
            nn.Linear(400, 128, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128, bias=False)
        )

        # self.fc_Radio = nn.Linear(512, 256)
        # self.fc_img = nn.Linear(400, 256)
        self.fc_text = nn.Linear(768, 256)
        self.SA = SelfAttention(16, 400 + 512 + 256, 400 + 512 + 256, hidden_dropout_prob=0.2)
        self.classify_head = DenseNet(layer_num=(6, 12, 24, 16), growth_rate=32, in_channels=1, classes=2)

    def forward(self, input_ids, attention_mask, token_type_ids, radio, img):
        '''
        :param radio: torch.Size([B, 1781])
        :param img: torch.Size([B, 1, 64, 512, 512])
        :return: torch.Size([B, 2])
        '''
        radiomic_feature = self.Radio_encoder(radio)[0]
        vision_feature = self.Resnet(img)

        radiomic_feature_pj = self.radio_projection_head(radiomic_feature)
        vision_feature_pj = self.img_projection_head(vision_feature)

        text_feature = self.bert(input_ids=input_ids, attention_mask=attention_mask,
                                 token_type_ids=token_type_ids).pooler_output

        # radiomic_feature = self.fc_Radio(radiomic_feature)
        # vision_feature = self.fc_img(vision_feature)
        text_feature = self.fc_text(text_feature)

        global_feature = torch.cat((radiomic_feature, vision_feature, text_feature), dim=1)
        global_feature = torch.unsqueeze(global_feature, dim=1)
        global_feature = self.SA(global_feature)
        return radiomic_feature_pj, vision_feature_pj, self.classify_head(global_feature)


class Radio_only_Mamba(nn.Module):
    def __init__(self):
        super(Radio_only_Mamba, self).__init__()
        self.name = 'Radiomic_only with Mamba'
        self.mamba_block = Radiomic_mamba_encoder(num_features=1781)
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
            nn.ReLU(),
            nn.Linear(1781, 2048),
            nn.LayerNorm(2048)
        )
        self.SA = SelfAttention(16, 2048, 2048, hidden_dropout_prob=0.2)
        self.classify_head = DenseNet(layer_num=(6, 12, 24, 16), growth_rate=32, in_channels=1, classes=2)

    def forward(self, radio):
        radio = self.projection1(radio)
        radio = torch.unsqueeze(radio, dim=1)
        feature = self.SA(radio)
        return self.classify_head(feature)


class Multi_model_Mamba_SA(nn.Module):
    def __init__(self):
        super(Multi_model_Mamba_SA, self).__init__()
        self.name = 'Multi_model_Mamba_SA'
        self.mamba_block = Radiomic_mamba_encoder(num_features=1781)
        self.mamba_block_clinic = Radiomic_mamba_encoder(num_features=58)
        self.Resnet = get_pretrained_Vision_Encoder()
        self.projection = nn.Sequential(
            nn.Linear(912, 512),
            nn.LayerNorm(512),
            nn.ReLU()
        )
        self.SA = SelfAttention(16, 512, 512, hidden_dropout_prob=0.2)
        self.classify_head = DenseNet(layer_num=(6, 12, 24, 16), growth_rate=32, in_channels=1, classes=2)

    def forward(self, cli, radio, img):
        radio_mamba_output = self.mamba_block(radio)
        cli_mamba_output = self.mamba_block_clinic(cli)
        vision_feature = self.Resnet(img)
        global_feature = torch.cat((radio_mamba_output, cli_mamba_output, vision_feature), dim=1)
        global_feature = self.projection(global_feature)
        feature = torch.unsqueeze(global_feature, dim=1)
        feature = self.SA(feature)
        output = self.classify_head(feature)
        return output


class Multi_model_mambacli(nn.Module):
    def __init__(self):
        super(Multi_model_mambacli, self).__init__()
        self.name = 'Multi_model_Mamba_SA'
        # self.mamba_block = Radiomic_mamba_encoder(num_features=1781)
        self.mamba_block_clinic = Radiomic_mamba_encoder(num_features=58)
        self.Resnet = get_pretrained_Vision_Encoder()
        self.projection = nn.Sequential(
            nn.Linear(912 - 256, 512),
            nn.LayerNorm(512),
            nn.ReLU()
        )
        self.SA = SelfAttention(16, 512, 512, hidden_dropout_prob=0.2)
        self.classify_head = DenseNet(layer_num=(6, 12, 24, 16), growth_rate=32, in_channels=1, classes=2)

    def forward(self, cli, img):
        # radio_mamba_output = self.mamba_block(radio)
        cli_mamba_output = self.mamba_block_clinic(cli)
        vision_feature = self.Resnet(img)
        global_feature = torch.cat([cli_mamba_output, vision_feature], dim=1)
        global_feature = self.projection(global_feature)
        feature = torch.unsqueeze(global_feature, dim=1)
        feature = self.SA(feature)
        output = self.classify_head(feature)
        return output


class Multi_model_MLP(nn.Module):
    def __init__(self):
        super(Multi_model_MLP, self).__init__()
        self.name = 'Multi_model_Mamba_SA'
        self.fc_radio = nn.Sequential(
            nn.ReLU(),
            nn.Linear(1781, 512),
            nn.Linear(512, 256),
            nn.LayerNorm(256)
        )
        self.fc_cli = nn.Sequential(
            nn.ReLU(),
            nn.Linear(58, 512),
            nn.Linear(512, 256),
            nn.LayerNorm(256)
        )
        self.Resnet = get_pretrained_Vision_Encoder()
        self.projection = nn.Sequential(
            nn.ReLU(),
            nn.Linear(912, 512),
            nn.LayerNorm(512)
        )
        self.SA = SelfAttention(16, 512, 512, hidden_dropout_prob=0.2)
        self.classify_head = DenseNet(layer_num=(6, 12, 24, 16), growth_rate=32, in_channels=1, classes=2)

    def forward(self, cli, radio, img):
        radio_mamba_output = self.fc_radio(radio)
        cli_mamba_output = self.fc_cli(cli)
        vision_feature = self.Resnet(img)
        global_feature = torch.cat((radio_mamba_output, cli_mamba_output, vision_feature), dim=1)
        global_feature = self.projection(global_feature)
        feature = torch.unsqueeze(global_feature, dim=1)
        feature = self.SA(feature)
        output = self.classify_head(feature)
        return output


class Triple_model_CrossAttentionFusion(nn.Module):
    def __init__(self):
        super(Triple_model_CrossAttentionFusion, self).__init__()
        self.name = 'Triple_model_CrossAttentionFusion'
        self.mamba_block_radio = Radiomic_mamba_encoder(num_features=1781)
        self.mamba_block_clinic = Radiomic_mamba_encoder(num_features=58)
        self.Resnet = get_pretrained_Vision_Encoder()
        self.fc_vis = nn.Linear(400, 256)
        self.fusion = TriModalCrossAttention(input_dim=1)
        self.classify_head = DenseNet(layer_num=(6, 12, 24, 16), growth_rate=16, in_channels=1, classes=2)

    def forward(self, cli, radio, img):
        cli_feature = self.mamba_block_clinic(cli)
        radio_feature = self.mamba_block_radio(radio)
        vision_feature = self.Resnet(img)
        vision_feature = self.fc_vis(vision_feature)

        cli_feature = torch.unsqueeze(cli_feature, dim=-1)
        radio_feature = torch.unsqueeze(radio_feature, dim=-1)
        vision_feature = torch.unsqueeze(vision_feature, dim=-1)
        global_feature = self.fusion(cli_feature, radio_feature, vision_feature)

        global_feature = global_feature.permute(0, 2, 1)
        output = self.classify_head(global_feature)
        return output


# class Triple_model_Self_CrossAttentionFusion_3(nn.Module):
#     def __init__(self):
#         super(Triple_model_Self_CrossAttentionFusion_3, self).__init__()
#         self.name = 'Triple_model_Self_CrossAttentionFusion_3'
#         self.mamba_block_radio = Radiomic_mamba_encoder(num_features=1781)
#         self.mamba_block_clinic = Radiomic_mamba_encoder(num_features=58)
#         self.Resnet = get_pretrained_Vision_Encoder()
#         self.fc_vis = nn.Linear(400, 256)
#         self.fusion = TriModalCrossAttention(input_dim=1)
#         self.SA1 = MultiheadAttention(8, 256, hidden_dropout_prob=0.2)
#         self.SA2 = MultiheadAttention(8, 256, hidden_dropout_prob=0.2)
#         self.SA3 = MultiheadAttention(8, 256, hidden_dropout_prob=0.2)
#         self.classify_head = DenseNet(layer_num=(6, 12, 24, 16), growth_rate=16, in_channels=1, classes=2)

#     def forward(self, cli, radio, img):

#         cli_feature = self.mamba_block_clinic(cli)
#         radio_feature = self.mamba_block_radio(radio)
#         vision_feature = self.Resnet(img)
#         vision_feature = self.fc_vis(vision_feature)

#         cli_feature = torch.unsqueeze(cli_feature, dim=1)
#         radio_feature = torch.unsqueeze(radio_feature, dim=1)
#         vision_feature = torch.unsqueeze(vision_feature, dim=1)

#         cli_feature = self.SA1(cli_feature)
#         radio_feature = self.SA2(radio_feature)
#         vision_feature = self.SA3(vision_feature)

#         cli_feature_tr = cli_feature.permute(0,2,1)
#         radio_feature_tr = radio_feature.permute(0,2,1)
#         vision_feature_tr = vision_feature.permute(0,2,1)

#         global_feature = self.fusion(cli_feature_tr, radio_feature_tr, vision_feature_tr)      #[B,N,1]

#         global_feature = global_feature.permute(0, 2, 1)
#         output = self.classify_head(global_feature)

#         return radio_feature, vision_feature, cli_feature, output

class Triple_model_Self_CrossAttentionFusion(nn.Module):
    """
    simple radio encoder, pretrained_Vision_Encoder, biobert
    """

    def __init__(self):
        super(Triple_model_Self_CrossAttentionFusion, self).__init__()
        self.name = 'Triple_model_Self_CrossAttentionFusion'
        self.Radio_encoder = Radiomic_encoder(num_features=1781)
        self.Radio_encoder.projection_head = nn.Identity()

        self.Resnet = get_pretrained_Vision_Encoder()

        self.bert = AutoModel.from_pretrained("./models/Bio_ClinicalBERT")

        self.fc_Radio = nn.Linear(512, 256)
        self.fc_vis = nn.Linear(400, 256)
        self.fc_text = nn.Linear(768, 256)

        self.fusion = TriModalCrossAttention(input_dim=1)
        self.SA1 = SelfAttention(16, 256, 256, hidden_dropout_prob=0.2)
        self.SA2 = SelfAttention(16, 256, 256, hidden_dropout_prob=0.2)
        self.SA3 = SelfAttention(16, 256, 256, hidden_dropout_prob=0.2)

        self.classify_head = DenseNet(layer_num=(6, 12, 24, 16), growth_rate=32, in_channels=1, classes=2)

    def forward(self, input_ids, attention_mask, token_type_ids, radio, img):
        '''
        :param radio: torch.Size([B, 1781])
        :param img: torch.Size([B, 1, 64, 512, 512])
        :return: torch.Size([B, 2])
        '''
        radio_feature = self.Radio_encoder(radio)[0]
        vision_feature = self.Resnet(img)
        cli_feature = self.bert(input_ids=input_ids, attention_mask=attention_mask,
                                token_type_ids=token_type_ids).pooler_output

        radio_feature = self.fc_Radio(radio_feature)
        vision_feature = self.fc_vis(vision_feature)
        cli_feature = self.fc_text(cli_feature)

        cli_feature = torch.unsqueeze(cli_feature, dim=1)
        radio_feature = torch.unsqueeze(radio_feature, dim=1)
        vision_feature = torch.unsqueeze(vision_feature, dim=1)

        cli_feature = self.SA1(cli_feature)
        radio_feature = self.SA2(radio_feature)
        vision_feature = self.SA3(vision_feature)

        cli_feature_tr = cli_feature.permute(0, 2, 1)
        radio_feature_tr = radio_feature.permute(0, 2, 1)
        vision_feature_tr = vision_feature.permute(0, 2, 1)

        _, _, _, global_feature = self.fusion(cli_feature_tr, radio_feature_tr, vision_feature_tr)  # [B,N,1]

        global_feature = global_feature.permute(0, 2, 1)
        output = self.classify_head(global_feature)

        return radio_feature, vision_feature, cli_feature, output


class Triple_model_Cross_SelfAttentionFusion(nn.Module):
    """
    simple radio encoder, pretrained_Vision_Encoder, biobert
    """

    def __init__(self):
        super(Triple_model_Cross_SelfAttentionFusion, self).__init__()
        self.name = 'Triple_model_Cross_SelfAttentionFusion'
        self.Radio_encoder = Radiomic_encoder(num_features=1781)
        self.Radio_encoder.projection_head = nn.Identity()

        self.Resnet = get_pretrained_Vision_Encoder()

        self.bert = AutoModel.from_pretrained("./models/Bio_ClinicalBERT")

        self.fc_Radio = nn.Linear(512, 256)
        self.fc_vis = nn.Linear(400, 256)
        self.fc_text = nn.Linear(768, 256)

        self.fusion = TriModalCrossAttention(input_dim=1)
        self.SA1 = SelfAttention(16, 256, 256, hidden_dropout_prob=0.2)
        self.SA2 = SelfAttention(16, 256, 256, hidden_dropout_prob=0.2)
        self.SA3 = SelfAttention(16, 256, 256, hidden_dropout_prob=0.2)

        self.classify_head = DenseNet(layer_num=(6, 12, 24, 16), growth_rate=32, in_channels=1, classes=2)

    def forward(self, input_ids, attention_mask, token_type_ids, radio, img):
        '''
        :param radio: torch.Size([B, 1781])
        :param img: torch.Size([B, 1, 64, 512, 512])
        :return: torch.Size([B, 2])
        '''
        radio_feature = self.Radio_encoder(radio)[0]
        vision_feature = self.Resnet(img)
        cli_feature = self.bert(input_ids=input_ids, attention_mask=attention_mask,
                                token_type_ids=token_type_ids).pooler_output

        radio_feature = self.fc_Radio(radio_feature)
        vision_feature = self.fc_vis(vision_feature)
        cli_feature = self.fc_text(cli_feature)

        cli_feature = torch.unsqueeze(cli_feature, dim=2)
        radio_feature = torch.unsqueeze(radio_feature, dim=2)
        vision_feature = torch.unsqueeze(vision_feature, dim=2)

        cli_feature, radio_feature, vision_feature, _ = self.fusion(cli_feature, radio_feature,
                                                                    vision_feature)  # [B,N,1]

        cli_feature_tr = cli_feature.permute(0, 2, 1)
        radio_feature_tr = radio_feature.permute(0, 2, 1)
        vision_feature_tr = vision_feature.permute(0, 2, 1)

        cli_feature = self.SA1(cli_feature_tr)
        radio_feature = self.SA2(radio_feature_tr)
        vision_feature = self.SA3(vision_feature_tr)

        global_feature = torch.cat([cli_feature, radio_feature, vision_feature], dim=2)

        output = self.classify_head(global_feature)

        return radio_feature, vision_feature, cli_feature, output


class Vis_mamba_only(nn.Module):
    def __init__(self):
        super(Vis_mamba_only, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=4, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm3d(4),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(in_channels=4, out_channels=8, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm3d(8),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv3d(in_channels=8, out_channels=16, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True)
        )
        # self.mamba_layer = Omnidirectional_3D_Mamba(in_channels=4)
        self.mamba_layer2 = Omnidirectional_3D_Mamba(in_channels=8)
        self.mamba_layer3 = Omnidirectional_3D_Mamba(in_channels=16)
        # self.classify_head = DenseNet(layer_num=(1, 3, 6, 4), growth_rate=4, in_channels=1, classes=2)
        self.projection = nn.Sequential(
            nn.BatchNorm1d(256 * 6),
            nn.ReLU(inplace=True),
            nn.Linear(256 * 6, 256)
        )
        self.mlp = nn.Linear(256, 2)

    def forward(self, x):
        '''
        :param x: torch.Size([B, 1, 64, 512, 512])
        :return: torch.Size([B, 2])
        '''
        x = self.conv(x)
        # feature_level_1 = self.mamba_layer(x)
        x = self.conv2(x)
        feature_level_2 = self.mamba_layer2(x)
        x = self.conv3(x)
        feature_level_3 = self.mamba_layer3(x)
        global_feature = torch.cat([feature_level_2, feature_level_3], dim=-1)
        global_feature = torch.squeeze(global_feature, dim=-2)
        global_feature = self.projection(global_feature)
        return self.mlp(global_feature)


class Two_model_CrossAttentionFusion(nn.Module):
    """
    pretrained_Vision_Encoder, biobert
    """

    def __init__(self):
        super(Two_model_CrossAttentionFusion, self).__init__()
        self.name = 'Two_model_CrossAttentionFusion'
        self.Resnet = get_pretrained_Vision_Encoder()
        self.bert = AutoModel.from_pretrained("./models/Bio_ClinicalBERT")

        self.fc_vis = nn.Linear(400, 256)
        self.fc_text = nn.Linear(768, 256)

        self.fusion = CrossAttention(input_dim=1)

        self.classify_head = DenseNet(layer_num=(6, 12, 24, 16), growth_rate=32, in_channels=1, classes=2)

    def forward(self, input_ids, attention_mask, token_type_ids, img):
        '''
        :param img: torch.Size([B, 1, 64, 512, 512])
        :return: torch.Size([B, 2])
        '''
        vision_feature = self.Resnet(img)
        cli_feature = self.bert(input_ids=input_ids, attention_mask=attention_mask,
                                token_type_ids=token_type_ids).pooler_output

        vision_feature = self.fc_vis(vision_feature)
        cli_feature = self.fc_text(cli_feature)

        cli_feature = torch.unsqueeze(cli_feature, dim=-1)
        vision_feature = torch.unsqueeze(vision_feature, dim=-1)

        global_feature = self.fusion(cli_feature, vision_feature)  # [B,N,1]

        global_feature = global_feature.permute(0, 2, 1)
        output = self.classify_head(global_feature)

        return output


class Two_textmodel_Fusion(nn.Module):
    """
    all biobert with cross_attention
    """

    def __init__(self):
        super(Two_textmodel_Fusion, self).__init__()
        self.name = 'Two_textmodel_Fusion'
        self.bert = AutoModel.from_pretrained("./models/Bio_ClinicalBERT")
        self.radio = Radiomic_mamba_encoder()

        self.fc_cli = nn.Linear(768, 256)

        self.fusion = CrossAttention(input_dim=1)

        self.classify_head = DenseNet(layer_num=(6, 12, 24, 16), growth_rate=32, in_channels=1, classes=2)

    def forward(self, input_ids, attention_mask, token_type_ids, img):
        '''
        :param img: torch.Size([B, 1, 64, 512, 512])
        :return: torch.Size([B, 2])
        '''
        radio_feature = self.radio(img)
        cli_feature = self.bert(input_ids=input_ids, attention_mask=attention_mask,
                                token_type_ids=token_type_ids).pooler_output

        cli_feature = self.fc_cli(cli_feature)

        cli_feature = torch.unsqueeze(cli_feature, dim=-1)
        radio_feature = torch.unsqueeze(radio_feature, dim=-1)

        global_feature = self.fusion(cli_feature, radio_feature)  # [B,N,1]

        global_feature = global_feature.permute(0, 2, 1)
        output = self.classify_head(global_feature)

        return output


class Two_model_CrossAttentionFusion_radio(nn.Module):
    """
    pretrained_Vision_Encoder, biobert
    """

    def __init__(self):
        super(Two_model_CrossAttentionFusion_radio, self).__init__()
        self.name = 'Two_model_CrossAttentionFusion_radio'
        self.Resnet = get_pretrained_Vision_Encoder()
        self.Radio_encoder = Radiomic_encoder(num_features=1781)
        self.fc_vis = nn.Linear(400, 256)
        self.fc_radio = nn.Linear(512, 256)
        self.fusion = CrossAttention(input_dim=1)

        self.classify_head = DenseNet(layer_num=(6, 12, 24, 16), growth_rate=32, in_channels=1, classes=2)

    def forward(self, radio, img):
        '''
        :param img: torch.Size([B, 1, 64, 512, 512])
        :return: torch.Size([B, 2])
        '''
        vision_feature = self.Resnet(img)
        radio_feature = self.Radio_encoder(radio)[0]
        vision_feature = self.fc_vis(vision_feature)
        radio_feature = self.fc_radio(radio_feature)

        radio_feature = torch.unsqueeze(radio_feature, dim=-1)
        vision_feature = torch.unsqueeze(vision_feature, dim=-1)

        global_feature = self.fusion(radio_feature, vision_feature)  # [B,N,1]

        global_feature = global_feature.permute(0, 2, 1)
        output = self.classify_head(global_feature)

        return output