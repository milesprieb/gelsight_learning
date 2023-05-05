import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import os
import cv2
import matplotlib.pyplot as plt
from torchvision.models import resnet50, ResNet50_Weights, resnet
import pandas as pd
import time
import datetime
import copy
import glob
import json
import re
from sklearn.metrics import confusion_matrix, classification_report
import tqdm
import seaborn as sns
import numpy as np
import pandas as pd
import wandb

start_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

label_map = {'bishop': 0, 
             'king': 1,
             'knight': 2,
             'pawn': 3,
             'queen': 4,
             'rook': 5,
             }

inverse_label_map = {0: 'bishop', 
                     1: 'king',
                     2: 'knight',
                     3: 'pawn',
                     4: 'queen',
                     5: 'rook',
                     }


        
class GelsightDepth(Dataset):
    def __init__(self, root_dir, transform=None):
        self.output_path = os.path.join(root_dir, 'train.json')
        self.root_dir = root_dir
        self.transform = transform
        # self.image_names = glob.glob(os.path.join(root_dir, '*.tif'))

    def __len__(self):
         with open(self.output_path) as f:
            data = json.load(f)
            return len(data)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # img_name_list = os.listdir(image_path)
        with open(self.output_path) as f:
            data = json.load(f)
            rgb_name = data[idx]['RGB_image']
            # print(rgb_name)
            label_name = re.split(r'[_\d]'  ,rgb_name)[1]
            img_path = os.path.join(self.root_dir, rgb_name)
        # print(img_path)
        # print(img_path)
        # image = Image.open(img_path)
        # image = np.array(image)
        image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED) 
        # image = image / 65535.0
        # plt.imshow(image)
        # plt.show()
        # print(image.max())
        # print(image.dtype)
        # print(image.shape)
        label = label_map[label_name]
        # image = np.expand_dims(image [:, :, 0], 2)
        # print(image.shape)
        # image = np.clip(image, 0, 1)
        # image = np.transpose(image, (2, 0, 1))
        # image = image.permute(2, 0, 1)
        img_name = rgb_name
        if self.transform:
            image = self.transform(image)
        # print(image.shape)
        return image, label, img_name

class GelsightRealDepth(Dataset):
    def __init__(self, root_dir, transform=None):
        # self.output_path = os.path.join(root_dir, 'out.json')
        self.root_dir = root_dir
        self.transform = transform
        self.image_names = glob.glob(os.path.join(root_dir, '*.png'))

    def __len__(self):
        #  with open(self.output_path) as f:
            # data = json.load(f)
            return len(self.image_names)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # print(self.image_names)
        img_path = self.image_names[idx]
        img_name = img_path.split('/')[-1]
        # print(img_name)
        label_name = img_name.split('_')[1]
        # print(label_name)
        # img_name_list = os.listdir(image_path)
        # with open(self.output_path) as f:
        #     data = json.load(f)
        #     img_name = data[idx]['Depth_image'].split('.')[0]+'.png'
        #     # print(img_name)
        #     label_name = re.split(r'[_\d]'  ,img_name)[1]
        #     img_path = os.path.join(self.root_dir, img_name)
        # print(img_path)
        depth_image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED) / 65535.0
        # print(depth_image.min())
        depth_image = depth_image / (depth_image.max() - depth_image.min())
        # print(depth_image.max())
        # plt.imshow(depth_image)
        # plt.show()
        # image = np.where(depth_image > 0, 1, 0)
        image = depth_image
        # plt.imshow(image)
        # plt.show()
        label = label_map[label_name]
        # print(label)
        # image = image / 65535 * 255 
        # image = np.expand_dims(image , 2)
        # print(image.shape)
        # image = np.clip(image, 0, 1)
        # image = np.transpose(image, (2, 0, 1))
        # image = image.permute(2, 0, 1)
        if self.transform:
            image = self.transform(image)
        # print(image.shape)
        return image, label, img_name

def train_model(model, criterion, optimizer, scheduler, dataloaders, dataset_sizes, num_epochs=25, device='cpu'):

    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            loss_list = []
            acc_list = []
            tqdm_data = tqdm.tqdm(dataloaders[phase], total=len(dataloaders[phase]))
            for inputs, labels, img_name in tqdm_data:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                tqdm_data.set_postfix_str(f'{phase} Loss: {loss.item():.4f} Acc: {torch.sum(preds == labels.data).double() / inputs.size(0):.4f}')
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            wandb.log({f'{phase}_loss': epoch_loss}, step=epoch)
            wandb.log({f'{phase}_acc': epoch_acc}, step=epoch)
            loss_list.append(epoch_loss)
            acc_list.append(epoch_acc)
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

def test_model(model, dataloaders, dataset_sizes, device='cpu'):
    df = pd.DataFrame(columns=['img_name', 'label', 'pred'])
    model.eval()
    running_corrects = 0
    y_pred = []
    y_true = []
    incorrect_names = []
    for inputs, labels, img_name in tqdm.tqdm(dataloaders):
        # print(inputs.shape)
        inputs = inputs.to(device)
        labels = labels.to(device)
        y_true.extend(labels.data.cpu().numpy())
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        y_pred.extend(preds.data.cpu().numpy())
        running_corrects += torch.sum(preds == labels.data)
        incorrect = torch.argwhere(preds != labels.data)
        for i in incorrect:
            incorrect_names.append(img_name[i])
            # print(labels.data[i].cpu(), preds[i].cpu())
            # df = df.append({'img_name': img_name[i], 
            #                 'label': inverse_label_map[int(labels.data[i].cpu()[0].numpy())], 
            #                 'pred': inverse_label_map[int(preds[i].cpu()[0].numpy())]}, ignore_index=True)
            df = pd.concat([df, pd.Series([img_name[i], 
                            inverse_label_map[int(labels.data[i].cpu()[0].numpy())], 
                            inverse_label_map[int(preds[i].cpu()[0].numpy())]], index = ['img_name', 'label', 'pred']).to_frame().T], ignore_index=True)
        # print(df)
    test_acc = running_corrects.double() / dataset_sizes['test']
    conf_matrix = confusion_matrix(y_true, y_pred)
    # print(f'Incorrect: {len(incorrect_names)}')
    # print('Incorrect names: ', incorrect_names)
    wandb_table = wandb.Table(dataframe=df)
    incorrect_df_artifact = wandb.Artifact('incorrect_df', type='dataset')
    incorrect_df_artifact.add(wandb_table, 'incorrect_df')
    wandb.log({'incorrect_df': wandb_table})
    wandb.log_artifact(incorrect_df_artifact)
    print(f'Test Acc: {test_acc:.4f}')
    print(conf_matrix)
    print(classification_report(y_true, y_pred, digits=4))
    # wandb.summary['classification_report'] = classification_report(y_true, y_pred, digits=4)
    wandb.sklearn.plot_confusion_matrix(y_true, y_pred, ['bishop', 
             'king',
             'knight',
             'pawn',
             'queen',
             'rook']) 
    wandb.log({'conf_matrix': wandb.plot.confusion_matrix(probs=None, y_true=y_true, preds=y_pred, class_names=['bishop', 
             'king',
             'knight',
             'pawn',
             'queen',
             'rook'])})



def view_images(inp, mean, std, title=None,):
    # print(inp.shape)
    inp = inp.numpy().transpose((1, 2, 0))
    # print(inp.shape)
    # mean = np.array([0.485, 0.456, 0.406])
    # std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    # inp = np.clip(inp, 0, 1)
    # print(inp.max())
    plt.imshow(inp/ inp.max())
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated
    plt.show()

def main():

    wandb.init(
    # set the wandb project where this run will be logged
    project="tacto-mask-sim",
    id="classification_rgb_{}".format(start_time),
    # track hyperparameters and run metadata
    config={
    "epochs": 24,
    })
    
    device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
    torch.cuda.empty_cache()
    train_transform = transforms.Compose([transforms.ToTensor(),
                                        #   transforms.CenterCrop((270, 362)), 
                                          transforms.Resize((224, 224)),  
                                          transforms.RandomHorizontalFlip(), 
                                          transforms.RandomVerticalFlip(),
                                        #   transforms.Normalize((0.1050,), (0.2312,)),
                                        # transforms.ConvertImageDtype(torch.float32),
                                        ])
    
    real_transform = transforms.Compose([transforms.ToTensor(), 
                                         transforms.Resize((224, 224)), 
                                        #  transforms.Normalize((0.1050,), (0.2312,)),
                                        #  transforms.Grayscale(1),
                                         transforms.ConvertImageDtype(torch.float32),
                                         ])
    
    
    # simulated dataset
    dataset = GelsightDepth('/home/user/gelsight/data/Real_rgb🌈', transform=train_transform)
    dataset_sim = dataset
    image1, label, img_name = dataset.__getitem__(0)
    # print(image1.shape)
    # plt.title(img_name)
    # plt.imshow(image1.numpy().transpose((1, 2, 0)))
    # plt.show()

    gen = torch.Generator().manual_seed(42)
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [0.25, 0.25, 0.5], generator=gen)

    train_dataloaders = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=8)
    val_dataloaders = DataLoader(val_dataset, batch_size=32, shuffle=True, num_workers=8)
    test_dataloaders = DataLoader(test_dataset, batch_size=32, shuffle=True, num_workers=8)

    dataloader = {'train': train_dataloaders, 'val': val_dataloaders}
    dataset_sizes = {'train': train_dataset.__len__(), 'val': val_dataset.__len__(), 'test': test_dataset.__len__()}

    # mean, std = 0, 1
    # inputs, classes, _ = next(iter(train_dataloaders))
    # # print(inputs.shape)
    
    # grid = torchvision.utils.make_grid(inputs)
    # view_images(grid, mean, std, title=None)

  
    

    model = torchvision.models.resnet50(pretrained=True)
    model_ftrs = model.fc.in_features
    # # print(model_ftrs)
    model.fc = nn.Linear(model_ftrs, 6)
    model = model.to(device)

    # # real dataset
    # dataset = GelsightRealDepth('data/real_depth/gs_data', transform=real_transform)
    # image2, label, img_name = dataset.__getitem__(0)
    # # print(image.shape)
    # # plt.title(img_name)
    # # plt.imshow(image.numpy().transpose((1, 2, 0)))
    # # plt.show()
    # dataset_sizes = {'test': dataset.__len__()}
    # test_dataloaders = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=8)

    # for i in range(100):
    #     i_sim = np.random.randint(0, dataset_sim.__len__())
    #     image1, label1, img_name1 = dataset_sim[i_sim]
    #     image2, label2, img_name2 = dataset[i]
    #     fig = plt.figure(figsize=(10, 10))
    #     fig.add_subplot(1, 2, 1)
    #     plt.title(img_name1)
    #     plt.imshow(image1.numpy().transpose((1, 2, 0)))
    #     fig.add_subplot(1, 2, 2)
    #     plt.title(img_name2)
    #     plt.imshow(image2.numpy().transpose((1, 2, 0)))
    #     plt.show()

    num_epochs = 10
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    criterion = nn.CrossEntropyLoss()
    
    # training
    model.load_state_dict(torch.load('models/new_rgb_classifier.pth'))
    model = train_model(model, criterion, optimizer, scheduler, dataloader, dataset_sizes, num_epochs, device)
    torch.save(model.state_dict(), 'models/new_rgb_ft_classifier.pth')

    #testing on real data
    model.load_state_dict(torch.load('models/new_rgb_ft_classifier.pth'))
    test_model(model, test_dataloaders, dataset_sizes, device)
    
    wandb.finish()

if __name__ == '__main__':
    main()