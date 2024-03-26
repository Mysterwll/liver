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