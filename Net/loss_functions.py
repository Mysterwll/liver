import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Masked_Language_Modeling_Loss(nn.Module):
    """
    Masked Language Modeling (MLM) Loss
    """

    def __init__(self):
        super(Masked_Language_Modeling_Loss, self).__init__()
        self.criterion = nn.NLLLoss(ignore_index=0)

    def forward(self, datas, labels):
        loss = 0.0
        for i in range(datas):
            next_sent_output, mask_lm_output = torch.eq(datas[i + 1], datas[i])
            next_loss = self.criterion(next_sent_output, datas[i + 1])
            mask_loss = self.criterion(mask_lm_output.transpose(1, 2), labels[i])
            loss += (next_loss + mask_loss)
        return loss


class Constract_Loss(nn.Module):
    def __init__(self, device="cpu"):
        super(Constract_Loss, self).__init__()

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.device = device

    def forward(self, image_features, text_features):
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        batch_size = image_features.shape[0]
        labels = torch.arange(batch_size, device=self.device).long()
        total_loss = (
                             F.cross_entropy(logits_per_image, labels) +
                             F.cross_entropy(logits_per_text, labels)
                     ) / 2

        return total_loss

class Similarity_Distribution_Matching_Loss(nn.Module):
    """
    Similarity Distribution Matching (SDM) Loss,
    Adapted from: https://github.com/anosorae/IRRA
    """

    def __init__(self, length):
        super(Similarity_Distribution_Matching_Loss, self).__init__()
        self.length = length

    def forward(self, vision_fetures, text_fetures, labels, epsilon=1e-8):
        logit_scale = self.length
        labels = labels - labels.t()
        labels = (labels == 0).float()

        image_norm = vision_fetures / vision_fetures.norm(dim=1, keepdim=True)
        text_norm = text_fetures / text_fetures.norm(dim=1, keepdim=True)

        t2i_cosine_theta = text_norm @ image_norm.t()
        i2t_cosine_theta = t2i_cosine_theta.t()

        text_proj_image = logit_scale * t2i_cosine_theta
        vision_proj_text = logit_scale * i2t_cosine_theta

        # normalize the true matching distribution
        labels_distribute = labels / labels.sum(dim=1)

        i2t_pred = F.softmax(vision_proj_text, dim=1)
        i2t_loss = i2t_pred * (F.log_softmax(vision_proj_text, dim=1) - torch.log(labels_distribute + epsilon))
        t2i_pred = F.softmax(text_proj_image, dim=1)
        t2i_loss = t2i_pred * (F.log_softmax(text_proj_image, dim=1) - torch.log(labels_distribute + epsilon))

        loss = torch.mean(torch.sum(i2t_loss, dim=1)) + torch.mean(torch.sum(t2i_loss, dim=1))

        return loss

class joint_loss(nn.Module):
    def __init__(self, w1=0.2, w2=0.01):
        super(joint_loss, self).__init__()
        self.w1 = w1
        self.w2 = w2
        self.Cross_Entropy_Loss = nn.CrossEntropyLoss()
        self.Similarity_Distribution_Matching_Loss_1 = Similarity_Distribution_Matching_Loss(2)
        self.Similarity_Distribution_Matching_Loss_2 = Similarity_Distribution_Matching_Loss(2)
        self.Similarity_Distribution_Matching_Loss_3 = Similarity_Distribution_Matching_Loss(2)

    def forward(self, modality1_features, modality2_features,  modality3_features, labels, scores):
        w1 = self.w1
        w2 = self.w2

        modality1_features = modality1_features.squeeze()
        modality2_features = modality2_features.squeeze()
        modality3_features = modality3_features.squeeze()

        cross_entropy_loss = self.Cross_Entropy_Loss(scores, labels)

        if labels.dim() == 1:
            labels = labels.unsqueeze(1)
        rv_sdm = self.Similarity_Distribution_Matching_Loss_1(modality1_features, modality2_features, labels)
        rc_sdm = self.Similarity_Distribution_Matching_Loss_2(modality2_features, modality3_features, labels)
        vc_sdm = self.Similarity_Distribution_Matching_Loss_3(modality1_features, modality3_features, labels)

        multi_loss = (1 - w1) * rv_sdm + w1 * (rc_sdm + vc_sdm)/2
        task_loss = cross_entropy_loss
        return task_loss + w2 * multi_loss

        # SDM_loss = self.Similarity_Distribution_Matching_Loss_1(modality1_features, modality2_features, labels)+\
        #             self.Similarity_Distribution_Matching_Loss_2(modality2_features, modality3_features, labels)+\
        #              self.Similarity_Distribution_Matching_Loss_3(modality1_features, modality3_features, labels)

        
        # return cross_entropy_loss + 0.01 * SDM_loss

if __name__ == '__main__':
    # smaller to test on local
    # tensor = torch.randn(size=(1, 768))
    # ltensor = torch.randn(size=(1, 1))
    # crien = Masked_Language_Modeling_Loss()
    # output = crien(tensor, ltensor)
    # print(output)
    # print(output.shape)

    img = torch.randn(size=(2, 512))
    radio = torch.randn(size=(2, 512))
    crien = Constract_Loss()
    output = crien(img, radio)
    print(output)
    print(output.shape)
