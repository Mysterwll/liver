import argparse
from datetime import datetime
from pathlib import Path
from sklearn.model_selection import KFold

import torch.utils.data
from torch.utils.data import random_split

from utils.model_object import models
from data.dataset import Liver_dataset, Liver_normalization_dataset
from utils.observer import Runtime_Observer


def prepare_to_train(model_index, seed, device, fold):
    global experiment_settings
    assert torch.cuda.is_available(), "Please ensure codes are executed on cuda."
    try:
        experiment_settings = models[model_index]
    except KeyError:
        print('model not in model_object!')
    torch.cuda.empty_cache()

    '''
    Dataset init, You can refer to the dataset format defined in data/dataset.py to define your private dataset
    '''
    dataset = Liver_normalization_dataset(experiment_settings['Data'], mode=experiment_settings['Dataset_mode'])
    # dataset = Liver_dataset(summery_path=experiment_settings['Data'], mode=experiment_settings['Dataset_mode'])
    torch.manual_seed(seed)
    # train_ratio = 0.7 train_dataset, test_dataset = random_split(dataset, [int(train_ratio * len(dataset)),
    # len(dataset) - int(train_ratio * len(dataset))]) trainDataLoader = torch.utils.data.DataLoader(train_dataset,
    # batch_size=experiment_settings['Batch'], shuffle=True, num_workers=4, drop_last=False) testDataLoader =
    # torch.utils.data.DataLoader(test_dataset, batch_size=experiment_settings['Batch'], shuffle=False, num_workers=4)
    '''
    The seed in Kfold should be same!
    '''
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    train_index, test_index = [[t1, t2] for t1, t2 in kf.split(dataset)][fold]
    train_dataset = torch.utils.data.Subset(dataset, train_index)
    test_dataset = torch.utils.data.Subset(dataset, test_index)

    trainDataLoader = torch.utils.data.DataLoader(train_dataset, batch_size=experiment_settings['Batch'], shuffle=True,
                                                  num_workers=4, drop_last=True)
    testDataLoader = torch.utils.data.DataLoader(test_dataset, batch_size=experiment_settings['Batch'], shuffle=False,
                                                 num_workers=4)
    '''
    Training logs and monitors
    '''
    target_dir = Path('./logs/')
    target_dir.mkdir(exist_ok=True)
    target_dir = target_dir.joinpath('classification')
    target_dir.mkdir(exist_ok=True)
    current_time = str(datetime.now().strftime('%Y-%m-%d_%H-%M'))
    target_dir = target_dir.joinpath(experiment_settings['Name'] + current_time)
    target_dir.mkdir(exist_ok=True)
    checkpoints_dir = target_dir.joinpath('checkpoints')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = target_dir.joinpath('logs')
    log_dir.mkdir(exist_ok=True)

    observer = Runtime_Observer(log_dir=log_dir, device=device, name=experiment_settings['Name'], seed=seed)
    observer.log(f'[DEBUG]Observer init successfully, program start @{current_time}\n')
    '''
    Model load
    '''
    _model = experiment_settings['Model']
    model = _model()
    # 如果有多个GPU可用，使用DataParallel来并行化模型
    if torch.cuda.device_count() > 1:
        observer.log("Using" + str(torch.cuda.device_count()) + "GPUs for training.\n")
        model = torch.nn.DataParallel(model)
    observer.log(f'Use model : {str(experiment_settings)}\n')
    num_params = 0
    for p in model.parameters():
        if p.requires_grad:
            num_params += p.numel()
    observer.log("\n===============================================\n")
    observer.log("model parameters: " + str(num_params))
    observer.log("\n===============================================\n")

    '''
    Hyper parameter settings
    '''
    optimizer = experiment_settings['Optimizer'](model.parameters(), experiment_settings['Lr'])
    criterion = experiment_settings['Loss']()

    print("prepare completed! launch training!\U0001F680")

    # launch
    _run = experiment_settings['Run']
    _run(observer, experiment_settings['Epoch'], trainDataLoader, testDataLoader, model, device, optimizer, criterion)


if __name__ == "__main__":
    # Adding necessary input arguments
    parser = argparse.ArgumentParser(description="add arguments to test")
    parser.add_argument("--model", default='climamba', type=str, help="model")
    parser.add_argument("--seed", default=42, type=int, help="seed given by LinkStart.py on cross Val")
    parser.add_argument("--device", default='cuda', type=str)
    parser.add_argument("--fold", default=0, type=int, help="0~4")
    args = parser.parse_args()

    print(args)
    prepare_to_train(model_index=args.model, seed=args.seed, device=args.device, fold=args.fold)
