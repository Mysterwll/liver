import argparse
from datetime import datetime
from pathlib import Path

import torch.utils.data
from torch.utils.data import random_split

from Net.vision_encoder import _3D_ResNet_50, get_pretrained_Vision_Encoder, pretrained_Resnet

from Net.radiomic_encoder import Radiomic_encoder

from Net.api import *
from Net.loss_functions import *
from data.dataset import Liver_dataset
from utils.observer import Runtime_Observer
from utils.observer import AverageMeter
from sklearn.svm import SVC

def train(batch_size, epochs, optimize_selection, learning_rate, name, seed, device):
    assert torch.cuda.is_available(), "Please ensure codes are executed on cuda."
    torch.cuda.empty_cache()
    '''
    Dataset init, You can refer to the dataset format defined in data/dataset.py to define your private dataset
    '''
    
    dataset = Liver_dataset("./data/summery.txt", mode='self_supervised')

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
    Radio_encoder = Radiomic_encoder(num_features=1782).to(device)
    Image_encoder = pretrained_Resnet().to(device)

    parameters = list(Radio_encoder.parameters()) + list(Image_encoder.parameters())
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

    criterion = Constract_Loss(device=device)

    observer.log(f'Use optimizer : {optimize_selection} learning_rate: {learning_rate} weight_decay: {1e-4}\n')
    
    print("prepare completed! launch training!\U0001F680")

    # launch
    print("start training")

    # lowest_loss = float('inf')
    best_acc = 0
    best_epoch = 0
    for epoch in range(epochs):
        print(f"Epoch: {epoch + 1}/{epochs}")

        train_losses = AverageMeter()

        Radio_encoder.train()
        Image_encoder.train()
        train_bar = tqdm(trainDataLoader, leave=True, file=sys.stdout)

        for i, (radio, img) in enumerate(train_bar):
            # print("radio.shpae = " + str(radio.shape))
            # print("img.shpae = " + str(img.shape))
            optimizer.zero_grad()
            radio = radio.to(device)
            img = img.to(device)
            
            radiomic_feature = Radio_encoder(radio)[1]
            vision_feature = Image_encoder(img)[1]

            radiomic_feature = radiomic_feature / radiomic_feature.norm(dim=1, keepdim=True)
            vision_feature = vision_feature / vision_feature.norm(dim=1, keepdim=True)

            loss = criterion(radiomic_feature, vision_feature)
            loss.backward()
            optimizer.step()

            train_losses.update(loss.item(), batch_size)
    
        val_dataset = Liver_dataset("./data/summery.txt", mode='radio_img_label')

        train_val_dataset, test_val_dataset = random_split(val_dataset, [int(train_ratio * len(dataset)), len(dataset) - int(train_ratio * len(dataset))])
        train_val_loader = torch.utils.data.DataLoader(train_val_dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
        test_val_loader = torch.utils.data.DataLoader(test_val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)


        feats_train = []
        labels_train = []

        Radio_encoder.eval()
        Image_encoder.eval()

         # Testing
        for i, (radio, img, label) in enumerate(train_val_loader):
            labels = label.numpy().tolist()

            radio = radio.to(device)
            img = img.to(device)

            with torch.no_grad():
                radiomic_feature = Radio_encoder(radio)[0]
                vision_feature = Image_encoder(img)[0]
                feats = torch.cat((radiomic_feature, vision_feature), dim=1)

            feats = feats.detach().cpu().numpy()
            for feat in feats:
                feats_train.append(feat)
            labels_train += labels

        feats_train = np.array(feats_train)
        labels_train = np.array(labels_train)

        feats_test = []
        labels_test = []

        for i, (radio, img, label) in enumerate(test_val_loader):
            labels = label.numpy().tolist()
            radio = radio.to(device)
            img = img.to(device)
            with torch.no_grad():
                radiomic_feature = Radio_encoder(radio)[0]
                vision_feature = Image_encoder(img)[0]
                feats = torch.cat((radiomic_feature, vision_feature), dim=1)

            feats = feats.detach().cpu().numpy()
            for feat in feats:
                feats_test.append(feat)
            labels_test += labels

        feats_test = np.array(feats_test)
        labels_test = np.array(labels_test)
        
        model_tl = SVC(C = 0.1, kernel ='linear')
        model_tl.fit(feats_train, labels_train)
        test_accuracy = model_tl.score(feats_test, labels_test)

        print(f"Epoch {epoch+1}, Average Loss: {train_losses.avg}")
        print(f"Linear Accuracy : {test_accuracy}")

        observer.record(epoch, train_losses.avg, test_accuracy)
        
        if test_accuracy > best_acc:
            best_acc = test_accuracy
            best_epoch = epoch
            print('==> Saving Best Model...')
            torch.save(Radio_encoder.state_dict(), checkpoints_dir/"radio_model_best.pth")
            torch.save(Image_encoder.state_dict(), checkpoints_dir/"img_model_best.pth")
  
    print('==> Saving Last Model...')
    torch.save(Radio_encoder.state_dict(), checkpoints_dir/"radio_model_last.pth")
    torch.save(Image_encoder.state_dict(), checkpoints_dir/"img_model_last.pth")
    print( "---experiment ended---\n" \
           + "Best Epoch %d:\n" % (best_epoch+1) \
           + "Best Linear Accuracy : %4.2f%%" % (best_acc) )

if __name__ == "__main__":
    # Adding necessary input arguments
    parser = argparse.ArgumentParser(description="add arguments to test")
    parser.add_argument("--bs", default=2, type=int, help="the batch_size of training")
    parser.add_argument("--ep", default=300, type=int, help="the epochs of training")
    parser.add_argument("--lr", default=0.0001, type=float, help="learning_rate")
    parser.add_argument("--name", default=None, type=str, help="Anything given by LinkStart.py on cross Val")
    parser.add_argument("--seed", default=None, type=int, help="seed given by LinkStart.py on cross Val")
    parser.add_argument("--model", default=7, type=int, help="the exp model")
    parser.add_argument("--optimizer", default='Adam', type=str, help="optimizer")
    parser.add_argument("--loss", default=1, type=int, help="optimizer")
    parser.add_argument("--device", default='cuda', type=str)
    args = parser.parse_args()

    print(args)
    train(batch_size=args.bs, epochs=args.ep, optimize_selection=args.optimizer, learning_rate=args.lr,
             name=args.name, seed=args.seed, device=args.device )

