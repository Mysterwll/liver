import sys

import torch
import torch.utils.data
from tqdm import tqdm
from transformers import AutoTokenizer


def run(observer, epochs, train_loader, test_loader, model, device, optimizer, criterion):
    model = model.to(device)
    print("start training")
    for epoch in range(epochs):
        print(f"Epoch: {epoch + 1}/{epochs}")
        observer.reset()
        model.train()
        train_bar = tqdm(train_loader, leave=True, file=sys.stdout)

        for i, (img, label) in enumerate(train_bar):
            optimizer.zero_grad()
            img = img.to(device)
            label = label.to(device)
            outputs = model(img)
            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            model.eval()
            test_bar = tqdm(test_loader, leave=True, file=sys.stdout)

            for i, (img, label) in enumerate(test_bar):
                img = img.to(device)
                label = label.to(device)
                outputs = model(img)
                _, predictions = torch.max(outputs, dim=1)
                observer.update(predictions, label)
        observer.excute(epoch)
    observer.finish()


def run_bert(observer, epochs, train_loader, test_loader, model, device, optimizer, criterion):
    model = model.to(device)
    print("start training")
    for epoch in range(epochs):
        print(f"Epoch: {epoch + 1}/{epochs}")
        observer.reset()
        model.train()
        train_bar = tqdm(train_loader, leave=True, file=sys.stdout)

        for i, (token, segment, mask, label) in enumerate(train_bar):
            optimizer.zero_grad()
            token, segment, mask = token.to(device), segment.to(device), mask.to(device)
            label = label.to(device)
            outputs = model(input_ids=token, attention_mask=mask, token_type_ids=segment)
            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            model.eval()
            test_bar = tqdm(test_loader, leave=True, file=sys.stdout)

            for i, (token, segment, mask, label) in enumerate(test_bar):
                token, segment, mask = token.to(device), segment.to(device), mask.to(device)
                label = label.to(device)
                outputs = model(input_ids=token, attention_mask=mask, token_type_ids=segment)
                _, predictions = torch.max(outputs, dim=1)
                observer.update(predictions, label)
        observer.excute(epoch)
    observer.finish()


def run_fusion(observer, epochs, train_loader, test_loader, model, device, optimizer, criterion):
    model = model.to(device)
    print("start training")
    for epoch in range(epochs):
        print(f"Epoch: {epoch + 1}/{epochs}")
        observer.reset()
        model.train()
        train_bar = tqdm(train_loader, leave=True, file=sys.stdout)

        for i, (token, segment, mask, img, label) in enumerate(train_bar):
            optimizer.zero_grad()
            token, segment, mask = token.to(device), segment.to(device), mask.to(device)
            img = img.to(device)
            label = label.to(device)
            outputs = model(input_ids=token, attention_mask=mask, token_type_ids=segment, img=img)
            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            model.eval()
            test_bar = tqdm(test_loader, leave=True, file=sys.stdout)

            for i, (token, segment, mask, img, label) in enumerate(test_bar):
                token, segment, mask = token.to(device), segment.to(device), mask.to(device)
                img = img.to(device)
                label = label.to(device)
                outputs = model(input_ids=token, attention_mask=mask, token_type_ids=segment, img=img)
                _, predictions = torch.max(outputs, dim=1)
                observer.update(predictions, label)
        observer.excute(epoch)
    observer.finish()


"""
This function has been deprecated, please refer to train.py for more information.
"""
# def run_Selfsupervision(epochs, train_loader, model, device, optimizer, criterion):
#     model = model.to(device)
#     print("start training")

#     lowest_loss = float('inf')

#     for epoch in range(epochs):
#         print(f"Epoch: {epoch + 1}/{epochs}")
        
#         model.train()
#         train_bar = tqdm(train_loader, leave=True, file=sys.stdout)

#         for i, (radio, img) in enumerate(train_bar):
#             # print("radio.shpae = " + str(radio.shape))
#             # print("img.shpae = " + str(img.shape))
#             optimizer.zero_grad()
#             radio = radio.to(device)
#             img = img.to(device)

#             radiomic_feature, vision_feature = model(radio=radio, img=img)

#             loss = criterion(radiomic_feature, vision_feature)
#             loss.backward()
#             optimizer.step()
     
#         avg_loss = loss.item() / len(train_loader)
        
#         print(f"Epoch {epoch+1}, Average Loss: {avg_loss}")

#         if avg_loss < lowest_loss:
#             lowest_loss = avg_loss
#             torch.save(model.state_dict(), "./models/liver/vision_radiomic_model.pth")
#             print(f"Saving model with lowest loss: {lowest_loss}")

def run_fusion_2(observer, epochs, train_loader, test_loader, model, device, optimizer, criterion):
    model = model.to(device)
    print("start training")
    for epoch in range(epochs):
        print(f"Epoch: {epoch + 1}/{epochs}")
        observer.reset()
        model.train()
        train_bar = tqdm(train_loader, leave=True, file=sys.stdout)

        for i, (token, segment, mask, radio, img, label) in enumerate(train_bar):
            optimizer.zero_grad()
            token, segment, mask = token.to(device), segment.to(device), mask.to(device)
            radio = radio.to(device)
            img = img.to(device)
            label = label.to(device)
            outputs = model(radio=radio, input_ids=token, attention_mask=mask, token_type_ids=segment, img=img)
            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            model.eval()
            test_bar = tqdm(test_loader, leave=True, file=sys.stdout)

            for i, (token, segment, mask, radio, img, label) in enumerate(test_bar):
                token, segment, mask = token.to(device), segment.to(device), mask.to(device)
                radio = radio.to(device)
                img = img.to(device)
                label = label.to(device)
                outputs = model(radio=radio, input_ids=token, attention_mask=mask, token_type_ids=segment, img=img)
                _, predictions = torch.max(outputs, dim=1)
                observer.update(predictions, label)
        observer.excute(epoch)
    observer.finish()