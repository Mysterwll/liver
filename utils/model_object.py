from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from Net.loss_functions import *
from Net.api import *
from Net.networks import *
from Net.radiomic_encoder import *
from Net.vision_encoder import *

models = {
    'Base': {
        'Name': 'Example',
        'Data': './data/summery.txt',
        'Batch': 2,
        'Lr': 0.0001,
        'Epoch': 300,
        'Dataset_mode': 'default',
        'Model': Vis_only,
        'Optimizer': Adam,
        'Loss': CrossEntropyLoss,
        'Run': run
    },
    'RadioSA': {
        'Name': 'Radio SelfAttention only with header',
        'Data': './data/summery_new.txt',
        'Batch': 16,
        'Lr': 0.001,
        'Epoch': 200,
        'Dataset_mode': 'mamba_test',
        'Model': Radio_only_SA,
        'Optimizer': Adam,
        'Loss': CrossEntropyLoss,
        'Run': run
    },
    'RadioMamba': {
        'Name': 'Radio Mamba Norm',
        'Data': './data/summery_new.txt',
        'Batch': 16,
        'Lr': 0.001,
        'Epoch': 200,
        'Dataset_mode': 'mamba_test',
        'Model': Radio_only_Mamba,
        'Optimizer': Adam,
        'Loss': CrossEntropyLoss,
        'Run': run
    },
    'Fusion': {
        'Name': 'Fusion based on contrast learning',
        'Data': './data/summery_new.txt',
        'Batch': 2,
        'Lr': 0.0001,
        'Epoch': 300,
        'Dataset_mode': 'fusion',
        'Model': Fusion_Main,
        'Optimizer': Adam,
        'Loss': joint_loss,
        'Run': run_main
    },

    'AllMamba': {
        'Name': 'ALLModel with Mamba and SA',
        'Data': './data/summery_new.txt',
        'Batch': 2,
        'Lr': 0.001,
        'Epoch': 200,
        'Dataset_mode': 'all_model',
        'Model': Multi_model_Mamba_SA,
        'Optimizer': Adam,
        'Loss': CrossEntropyLoss,
        'Run': run_fusion_all
    }

}
