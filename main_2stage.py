import argparse
from datetime import datetime
from pathlib import Path

import torch.utils.data
from torch.utils.data import random_split

from Net.vision_encoder import _3D_ResNet_50, get_pretrained_Vision_Encoder, pretrained_Resnet

from Net.radiomic_encoder import Radiomic_encoder

from Net.networks import *
from Net.api import *
from Net.loss_functions import *
from data.dataset import Liver_dataset, Liver_normalization_dataset
from utils.observer import Runtime_Observer, Runtime_Observer_test, Test_Observer
from utils.observer import AverageMeter

def train_epoch(model, trainDataLoader, optimizer, criterion, device):
    train_loss = AverageMeter()

    train_bar = tqdm(trainDataLoader, leave=True, file=sys.stdout)
    for i, (token, segment, mask, radio, img, label) in enumerate(train_bar):
        optimizer.zero_grad()
        token, segment, mask = token.to(device), segment.to(device), mask.to(device)
        img = img.to(device)
        radio = radio.to(device)
        label = label.to(device)
            
        outputs = model(input_ids=token, attention_mask=mask, token_type_ids=segment, radio=radio, img=img)     

        loss = criterion(outputs, label)
        loss.backward()
        optimizer.step()

        train_loss.update(loss.item(), radio.shape[0])

    return train_loss

def val_epoch(model, valDataLoader, criterion, device, observer):
    val_loss = AverageMeter()
    
    val_bar = tqdm(valDataLoader, leave=True, file=sys.stdout)
    for i, ( token, segment, mask, radio, img, label ) in enumerate(val_bar):

        token, segment, mask = token.to(device), segment.to(device), mask.to(device)
        img = img.to(device)
        radio = radio.to(device)
        label = label.to(device)
        
        outputs = model(input_ids=token, attention_mask=mask, token_type_ids=segment, radio=radio, img=img)
        loss = criterion(outputs, label)
        _, predictions = torch.max(outputs, dim=1)
        observer.update(predictions, label)

        val_loss.update(loss.item(), radio.shape[0])

    return val_loss

def test(model, testDataLoader, observer, device):
    test_bar = tqdm(testDataLoader, leave=True, file=sys.stdout)
    for i, (token, segment, mask, radio, img, label) in enumerate(test_bar):

        token, segment, mask = token.to(device), segment.to(device), mask.to(device)
        img = img.to(device)
        radio = radio.to(device)
        label = label.to(device)
        
        outputs = model(input_ids=token, attention_mask=mask, token_type_ids=segment, radio=radio, img=img)
        _, predictions = torch.max(outputs, dim=1)
        observer.update(predictions, label)


def train(batch_size, epochs, optimize_selection, learning_rate, name, seed, device):
    assert torch.cuda.is_available(), "Please ensure codes are executed on cuda."
    torch.cuda.empty_cache()
    '''
    Dataset init, You can refer to the dataset format defined in data/dataset.py to define your private dataset
    '''
    
    dataset = Liver_normalization_dataset("./data/summery_new.txt", mode='fusion')

    torch.manual_seed(seed if (seed is not None) else 42)

    train_ratio = 0.6
    val_ratio = 0.2
    test_ratio = 0.2

    total_length = len(dataset)
    train_length = int(train_ratio * total_length)
    val_length = int(val_ratio * total_length)
    test_length = total_length - train_length - val_length

    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_length, val_length, test_length])

    trainDataLoader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=False)
    valDataLoader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
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

    observer = Runtime_Observer_test(log_dir=log_dir, device=device, name=name, seed=seed)
    observer.log(f'[DEBUG]Observer init successfully, program start @{current_time}\n')
    '''
    Model load
    '''
    model = Fusion_2stage('./models/liver/radio_model_best.pth', './models/liver/img_model_best.pth').to(device)

    parameters = list(model.parameters())
    num_params = 0

    for p in parameters:
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
            parameters,
            lr=learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=1e-4
        )
    else:
        optimizer = torch.optim.SGD(parameters, lr=0.01, momentum=0.9)

    criterion = nn.CrossEntropyLoss(device=device)

    observer.log(f'Use optimizer : {optimize_selection} learning_rate: {learning_rate} weight_decay: {1e-4}\n')
    
    print("prepare completed! launch training!\U0001F680")

    # launch
    print("start training")
    # Train the model for a fixed number of epochs

    for epoch in range(epochs):
        print(f"Epoch: {epoch + 1}/{epochs}")
        observer.reset()

        model.train()
        train_loss = train_epoch(model, trainDataLoader, optimizer, criterion, device)

        model.eval()
        with torch.no_grad():
            val_loss = val_epoch(model, valDataLoader, criterion, device, observer)

        observer.record(epoch, train_loss.avg, val_loss.avg)

        flag = observer.excute(epoch)
        
        if flag:
            print('==> Saving Best Model...')
            torch.save(model.state_dict(), checkpoints_dir/"fusion2stage_model_best.pth")
    
    observer.finish()
  
    print('==> Saving Last Model...')
    torch.save(model.state_dict(), checkpoints_dir/"fusion2stage_model_last.pth")


    print("------start testing--------")
    test_observer = Test_Observer(log_dir=log_dir, device=device, name=name, seed=seed)

    test_observer.reset()
    state_dict = torch.load(checkpoints_dir/"fusion2stage_model_best.pth")
    model.load_state_dict(state_dict)
    model.eval()
    with torch.no_grad():
        test(model, testDataLoader, test_observer, device)

    test_observer.finish()


if __name__ == "__main__":
    # Adding necessary input arguments
    parser = argparse.ArgumentParser(description="add arguments to test")
    parser.add_argument("--bs", default=2, type=int, help="the batch_size of training")
    parser.add_argument("--ep", default=300, type=int, help="the epochs of training")
    parser.add_argument("--lr", default=0.0001, type=float, help="learning_rate")
    parser.add_argument("--name", default=None, type=str, help="Anything given by LinkStart.py on cross Val")
    parser.add_argument("--seed", default=None, type=int, help="seed given by LinkStart.py on cross Val")
    parser.add_argument("--optimizer", default='Adam', type=str, help="optimizer")
    parser.add_argument("--device", default='cuda', type=str)
    args = parser.parse_args()

    print(args)
    train(batch_size=args.bs, epochs=args.ep, optimize_selection=args.optimizer, learning_rate=args.lr,
             name=args.name, seed=args.seed, device=args.device )