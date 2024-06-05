from torch.nn import CrossEntropyLoss
from torch.optim import Adam, SGD
from Net.loss_functions import *
from Net.api import *
from Net.networks import *
from Net.radiomic_encoder import *
from Net.vision_encoder import *
from Net.cp_networks import *

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
    'Vision': {
        'Name': 'Vision only with pre_resnet3D',
        'Data': './data/summery_new.txt',
        'Batch': 2,
        'Lr': 0.0001,
        'Epoch': 300,
        'Dataset_mode': 'img',
        'Model': Vis_only,
        'Optimizer': Adam,
        'Loss': CrossEntropyLoss,
        'Run': run
    },
    'RadioSA': {
        'Name': 'Radio SelfAttention Norm',
        'Data': './data/summery_new.txt',
        'Batch': 16,
        'Lr': 0.0001,
        'Epoch': 300,
        'Dataset_mode': 'mamba_test',
        'Model': Radio_only_SA,
        'Optimizer': Adam,
        'Loss': CrossEntropyLoss,
        'Run': run_single
    },
    'RadioMamba': {
        'Name': 'Radio Mamba Norm',
        'Data': './data/summery_new.txt',
        'Batch': 16,
        'Lr': 0.0001,
        'Epoch': 300,
        'Dataset_mode': 'mamba_test',
        'Model': Radio_only_Mamba,
        'Optimizer': Adam,
        'Loss': CrossEntropyLoss,
        'Run': run_single
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
        'Name': 'ALLModelC',
        'Data': './data/summery_new.txt',
        'Batch': 2,
        'Lr': 0.001,
        'Epoch': 200,
        'Dataset_mode': 'all_model',
        'Model': Multi_model_MLP,
        'Optimizer': Adam,
        'Loss': CrossEntropyLoss,
        'Run': run_fusion_all
    },
    'climamba': {
        'Name': '2model',
        'Data': './data/summery_new.txt',
        'Batch': 2,
        'Lr': 0.001,
        'Epoch': 200,
        'Dataset_mode': '2_model',
        'Model': Multi_model_mambacli,
        'Optimizer': Adam,
        'Loss': CrossEntropyLoss,
        'Run': run_fusion_test
    },
    'CA3fusion': {
        'Name': 'CA3fusion',
        'Data': './data/summery_new.txt',
        'Batch': 2,
        'Lr': 0.0001,
        'Epoch': 300,
        'Dataset_mode': 'all_model',
        'Model': Triple_model_CrossAttentionFusion,
        'Optimizer': Adam,
        'Loss': CrossEntropyLoss,
        'Run': run_fusion_all
    },
    'SCA3fusion': {
        'Name': 'SCA3fusion',
        'Data': './data/summery_new.txt',
        'Batch': 2,
        'Lr': 0.0001,
        'Epoch': 300,
        'Dataset_mode': 'all_model',
        'Model': Triple_model_Self_CrossAttentionFusion,
        'Optimizer': Adam,
        'Loss': CrossEntropyLoss,
        'Run': run_fusion_all
    },
    'SCA3fusion_1': {
        'Name': 'SCA3fusion without mamba',
        'Data': './data/summery_new.txt',
        'Batch': 2,
        'Lr': 0.0001,
        'Epoch': 300,
        'Dataset_mode': 'fusion',
        'Model': Triple_model_Self_CrossAttentionFusion_1,
        'Optimizer': Adam,
        'Loss': CrossEntropyLoss,
        'Run': run_main
    },
    'SCA3fusion_2': {
        'Name': 'SCA3fusion without mamba with joint loss',
        'Data': './data/summery_new.txt',
        'Batch': 2,
        'Lr': 0.0001,
        'Epoch': 300,
        'Dataset_mode': 'fusion',
        'Model': Triple_model_Self_CrossAttentionFusion_2,
        'Optimizer': Adam,
        'Loss': joint_loss,
        'Run': run_main_1
    },
    'SCA3fusion_3': {
        'Name': 'SCA3fusion with mamba with joint loss',
        'Data': './data/summery_new.txt',
        'Batch': 2,
        'Lr': 0.0001,
        'Epoch': 300,
        'Dataset_mode': 'all_model',
        'Model': Triple_model_Self_CrossAttentionFusion_3,
        'Optimizer': Adam,
        'Loss': joint_loss,
        'Run': run_fusion_all_1
    },
    'SCA3fusion_new': {
        'Name': 'SCA3fusion without mamba',
        'Data': './data/summery_new.txt',
        'Batch': 2,
        'Lr': 0.0001,
        'Epoch': 300,
        'Dataset_mode': 'fusion',
        'Model': Triple_model_Self_CrossAttentionFusion,
        'Optimizer': Adam,
        'Loss': joint_loss,
        'Run': run_main_1
    },
    'CSA3fusion': {
        'Name': 'CSA3fusion without mamba',
        'Data': './data/summery_new.txt',
        'Batch': 2,
        'Lr': 0.0001,
        'Epoch': 300,
        'Dataset_mode': 'fusion',
        'Model': Triple_model_Cross_SelfAttentionFusion,
        'Optimizer': Adam,
        'Loss': joint_loss,
        'Run': run_main_1
    },
    'SCA3fusion_test_1': {
        'Name': 'SCA3fusion without mamba with joint loss',
        'Data': './data/summery_new.txt',
        'Batch': 2,
        'Lr': 0.0001,
        'Epoch': 300,
        'Dataset_mode': 'fusion',
        'Model': Triple_model_test_1,
        'Optimizer': Adam,
        'Loss': joint_loss,
        'Run': run_main_1
    },

    'SCA3fusion_test_2': {
        'Name': 'SCA3fusion without mamba with joint loss',
        'Data': './data/summery_new.txt',
        'Batch': 2,
        'Lr': 0.0001,
        'Epoch': 300,
        'Dataset_mode': 'fusion',
        'Model': Triple_model_test_2,
        'Optimizer': Adam,
        'Loss': joint_loss,
        'Run': run_main_1
    },
    'Resnet50': {
        'Name': 'Resnet50',
        'Data': './data/summery_new.txt',
        'Batch': 2,
        'Lr': 0.0001,
        'Epoch': 300,
        'Dataset_mode': 'img',
        'Model': Resnet50,
        'Optimizer': Adam,
        'Loss': CrossEntropyLoss,
        'Run': run
    },
    'ViT': {
        'Name': 'ViT',
        'Data': './data/summery_new.txt',
        'Batch': 2,
        'Lr': 0.0001,
        'Epoch': 300,
        'Dataset_mode': 'img',
        'Model': MyViT,
        'Optimizer': Adam,
        'Loss': CrossEntropyLoss,
        'Run': run
    },
    'HFBSurv': {
        'Name': 'HFBSurv',
        'Data': './data/summery_new.txt',
        'Batch': 2,
        'Lr': 0.0001,
        'Epoch': 300,
        'Dataset_mode': 'fusion',
        'Model': HFBSurv,
        'Optimizer': Adam,
        'Loss': CrossEntropyLoss,
        'Run': run_main
    },
    'MMD': {
        'Name': 'MMD',
        'Data': './data/summery_new.txt',
        'Batch': 2,
        'Lr': 0.0001,
        'Epoch': 300,
        'Dataset_mode': 'fusion',
        'Model': MMD,
        'Optimizer': Adam,
        'Loss': CrossEntropyLoss,
        'Run': run_main
    }

}
