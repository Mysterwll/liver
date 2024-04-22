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
    def __init__(self, device = "cpu"):
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
    
class joint_loss(nn.Module):
    def __init__(self):
        super(joint_loss, self).__init__()
        self.loss_cl = Constract_Loss('cuda')
        self.loss_task = nn.CrossEntropyLoss()
    def forward(self,modality1_features, modality2_features, logits, label):
        cl_loss = self.loss_cl(modality1_features, modality2_features, 0.1)
        
        task_loss = self.loss_task(logits, label)
        loss =  cl_loss + task_loss
        return loss


if __name__ == '__main__':
    # smaller to test on local
    # tensor = torch.randn(size=(1, 768))
    # ltensor = torch.randn(size=(1, 1))
    # crien = Masked_Language_Modeling_Loss()
    # output = crien(tensor, ltensor)
    # print(output)
    # print(output.shape)

    img = torch.randn(size=(2,512))
    radio = torch.randn(size=(2,512))
    crien = Constract_Loss()
    output = crien(img, radio)
    print(output)
    print(output.shape)

