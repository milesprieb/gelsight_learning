import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import os
import io
import cv2
import matplotlib.pyplot as plt
from torchvision.models import resnet50, ResNet50_Weights
import pandas as pd
import time
import copy

class GelsightDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.csv = os.path.join(root_dir, 'images.csv')
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.csv)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        image_path = os.path.join(self.root_dir, 'images')
        # img_name_list = os.listdir(image_path)
        csv = pd.read_csv(self.csv)
        img_name = os.path.join(image_path, csv.iloc[idx, 0])
        image = cv2.imread(img_name, cv2.COLOR_BGR2RGB)
        label = csv.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        return image, label
    

def main():
    train_transform = transforms.Compose([transforms.ToTensor(), 
                                          transforms.Resize((224, 224)), 
                                          transforms.RandomRotation(45), 
                                          transforms.RandomHorizontalFlip(), 
                                          transforms.RandomVerticalFlip(),])
    
    dataset = GelsightDataset('classification_data', transform=train_transform)
    image, label = dataset.__getitem__(10)
    # print(label)
    # print(image.shape)
    print(len(dataset))
    plt.imshow(image.permute(1, 2, 0))
    plt.show()
    dataloaders = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)
    model = torchvision.models.resnet50(pretrained=True)
    # print(model)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    num_epochs = 25
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    dataset_sizes = len()

    # training model
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
            for inputs, labels in dataloaders[phase]:
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
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

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

if __name__ == "__main__":
    main()