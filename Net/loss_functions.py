import torch
import torch.nn as nn


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


if __name__ == '__main__':
    # smaller to test on local
    tensor = torch.randn(size=(1, 768))
    ltensor = torch.randn(size=(1, 1))
    crien = Masked_Language_Modeling_Loss()
    output = crien(tensor, ltensor)
    print(output)
    print(output.shape)