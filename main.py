import argparse
from datetime import datetime
from pathlib import Path

import torch.utils.data
from torch.utils.data import random_split

from Net.api import *
from Net.networks import *
from Net.loss_functions import *
from data.dataset import Liver_dataset
from utils.observer import Runtime_Observer


def prepare_to_train(model_selection, batch_size, epochs, optimize_selection, loss_selection, learning_rate, name,
                     seed, device, vision_pre):
    assert torch.cuda.is_available(), "Please ensure codes are executed on cuda."
    torch.cuda.empty_cache()
    '''
    Dataset init, You can refer to the dataset format defined in data/dataset.py to define your private dataset
    '''
    if model_selection >= 8:
        dataset = Liver_dataset("./data/summery.txt", mode='self_supervised')
    elif 7 >= model_selection >= 6:
        dataset = Liver_dataset("./data/summery.txt", mode='fusion')
    elif 6 > model_selection >= 3:
        dataset = Liver_dataset("./data/summery.txt", mode='bert')
    else:
        dataset = Liver_dataset("./data/summery.txt", mode='img')
    
    torch.manual_seed(seed if (seed is not None) else 42)
    train_ratio = 0.7
    train_dataset, test_dataset = random_split(dataset, [int(train_ratio * len(dataset)), len(dataset) - int(train_ratio * len(dataset))])
    trainDataLoader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
    testDataLoader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    '''
    Training logs and monitors
    '''
    target_dir = Path('./logs/')
    target_dir.mkdir(exist_ok=True)
    target_dir = target_dir.joinpath('classification')
    target_dir.mkdir(exist_ok=True)
    current_time = str(datetime.now().strftime('%Y-%m-%d_%H-%M'))
    if name is None:
        target_dir = target_dir.joinpath(current_time)
    else:
        target_dir = target_dir.joinpath(name + current_time)
    target_dir.mkdir(exist_ok=True)
    checkpoints_dir = target_dir.joinpath('checkpoints')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = target_dir.joinpath('logs')
    log_dir.mkdir(exist_ok=True)

    observer = Runtime_Observer(log_dir=log_dir, device=device, name=name, seed=seed)
    observer.log(f'[DEBUG]Observer init successfully, program start @{current_time}\n')
    '''
    Model load
    '''
    if model_selection == 1:
        model = Vis_only(use_pretrained= vision_pre)
    elif model_selection == 2:
        model = Vis_only_header()
    elif model_selection == 3:
        model = Text_only_header()
    elif model_selection == 6:
        model = Fusion_Concat()
    elif model_selection == 7:
        model = Fusion_SelfAttention()
    elif model_selection == 8:
        model = Contrastive_Learning()
    else:
        model = Vis_only()
    observer.log(f'Use model : {model_selection} -> {model.name}\n')
    
    num_params = 0
    for p in model.parameters():
        if p.requires_grad:
            num_params += p.numel()
    print("===============================================")
    print("model parameters: " + str(num_params))
    print("===============================================")
    
    '''
    Hyper parameter settings
    '''
    if optimize_selection == 'Adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=1e-4
        )
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    if loss_selection == 1:
        criterion = nn.CrossEntropyLoss()
    elif loss_selection == 3:
        criterion = Constract_Loss(device='cuda')
    # elif loss_selection == 2:
    #     criterion = Masked_Language_Modeling_Loss()
    else:
        criterion = nn.CrossEntropyLoss()
    observer.log(f'Use optimizer : {optimize_selection} learning_rate: {learning_rate} weight_decay: {1e-4}\n')
    observer.log(f'Use loss function : {loss_selection}\n')
    print("prepare completed! launch training!\U0001F680")

    # launch
    if model_selection ==8:
        run_Selfsupervision(epochs=epochs, train_loader=trainDataLoader, 
                   model=model, device=device, optimizer=optimizer, criterion=criterion)
    elif model_selection >= 6:
        run_fusion(observer=observer, epochs=epochs, train_loader=trainDataLoader, test_loader=testDataLoader,
                   model=model, device=device, optimizer=optimizer, criterion=criterion)
    elif model_selection == 3:
        run_bert(observer=observer, epochs=epochs, train_loader=trainDataLoader, test_loader=testDataLoader,
                 model=model, device=device, optimizer=optimizer, criterion=criterion)
    else:
        run(observer=observer, epochs=epochs, train_loader=trainDataLoader, test_loader=testDataLoader, model=model,
            device=device, optimizer=optimizer, criterion=criterion)


if __name__ == "__main__":
    # Adding necessary input arguments
    parser = argparse.ArgumentParser(description="add arguments to test")
    parser.add_argument("--bs", default=2, type=int, help="the batch_size of training")
    parser.add_argument("--ep", default=300, type=int, help="the epochs of training")
    parser.add_argument("--lr", default=0.0001, type=float, help="learning_rate")
    parser.add_argument("--name", default=None, type=str, help="Anything given by LinkStart.py on cross Val")
    parser.add_argument("--seed", default=None, type=int, help="seed given by LinkStart.py on cross Val")
    parser.add_argument("--model", default=1, type=int, help="the exp model")
    parser.add_argument("--optimizer", default='Adam', type=str, help="optimizer")
    parser.add_argument("--loss", default=1, type=int, help="optimizer")
    parser.add_argument("--device", default='cuda', type=str)
    parser.add_argument("--vision_pre", default=True, help="whether use pre-trained vision encoder")
    args = parser.parse_args()

    print(args)
    prepare_to_train(model_selection=args.model, batch_size=args.bs, epochs=args.ep, learning_rate=args.lr,
                     optimize_selection=args.optimizer, name=args.name, seed=args.seed, device=args.device,
                     loss_selection=args.loss, vision_pre=args.vision_pre)
