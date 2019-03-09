from torch.utils.data import DataLoader
import torch
import torchvision
import sys
from src.data import TrainDataset
from src.data import get_test_time_transform, get_train_time_transform_simple, get_train_time_transform
from src.utils import show_data_batch
from src.run import train_model
from src.model import initialize_model, get_model_input_size

# set the path and directory for train and validation
train_dataset_csv = 'processed_data/train_train.csv'
val_dataset_csv = 'processed_data/train_val.csv'
train_root_dir = 'datasets'
test_root_dir = 'datasets'

# Models to choose from [resnet, alexnet, vgg, squeezenet, densenet, inception]
model_name = 'resnet50'
input_size = get_model_input_size(model_name)

# Number of classes in the dataset
num_classes = 58

# Batch size for training (change depending on how much memory you have)
batch_size = 128

# Number of epochs to train for 
num_epochs = 15

# Flag for feature extracting. When False, we finetune the whole model, 
#   when True we only update the reshaped layer params
feature_extract = False

crop_size = input_size
# we use (scale_size: 256, crop_size: 224) and (scale_size: 320, crop_size: 299)
# for resize and cropping
scale_size = 320 if input_size == 299 else 256

print('input size: {} scale size: {} crop size: {}'.format(input_size, scale_size, crop_size))

# transforms settings
train_transform_aug = get_train_time_transform(scale_size=scale_size, crop_size=crop_size)
train_transform_simple = get_train_time_transform_simple(scale_size=scale_size, crop_size=crop_size)
test_transform = get_test_time_transform(scale_size=crop_size, crop_size=crop_size)

# datasets with transforms
train_dataset_simple = TrainDataset(
    csv_file=train_dataset_csv, root_dir=train_root_dir, transform=train_transform_simple)
train_dataset_aug = TrainDataset(
    csv_file=train_dataset_csv, root_dir=train_root_dir, transform=train_transform_aug)
val_dataset = TrainDataset(
    csv_file=val_dataset_csv, root_dir=train_root_dir, transform=test_transform)

# data loaders
train_data_loader_simple = DataLoader(
    train_dataset_simple, batch_size=batch_size, shuffle=True, num_workers = 8)
train_data_loader_aug = DataLoader(
    train_dataset_aug, batch_size=batch_size, shuffle=True, num_workers = 8)
val_data_loader = DataLoader(
    val_dataset, batch_size=batch_size, shuffle=False, num_workers = 8)

# data loaders in dict
data_loaders_simple = {
    'train': train_data_loader_simple,
    'val': val_data_loader
}

data_loaders_aug = {
    'train': train_data_loader_aug,
    'val': val_data_loader
}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('you are using device: ', device)

model_ft = initialize_model(
    model_name=model_name, num_classes=num_classes, feature_extract=feature_extract, 
    use_pretrained=True)

print(model_ft)

# Send the model to GPU
model_ft = model_ft.to(device)

# Gather the parameters to be optimized/updated in this run. If we are
#  finetuning we will be updating all parameters. However, if we are 
#  doing feature extract method, we will only update the parameters
#  that we have just initialized, i.e. the parameters with requires_grad
#  is True.
params_to_update = model_ft.parameters()
print("Params to learn:")
if feature_extract:
    params_to_update = []
    for name,param in model_ft.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            print("\t",name)
else:
    for name,param in model_ft.named_parameters():
        if param.requires_grad == True:
            print("\t",name)
            
# define your loss type
logloss_criterion = torch.nn.CrossEntropyLoss()

# Observe that only parameters of final layer are being optimized as
# opposed to before.
lr = 1e-3
# sgd_optimizer = torch.optim.SGD(params_to_update, lr=0.001, momentum=0.9)
adam_optimizer = torch.optim.Adam(params_to_update, lr=lr, weight_decay=0)

# Decay LR by a factor of 0.1 every 7 epochs
# exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(sgd_optimizer, step_size=5, gamma=0.1)

# model training
trained_model, full_performance_history = train_model(
    model=model_ft, device=device, dataloaders=data_loaders_simple, 
    criterion=logloss_criterion, optimizer=adam_optimizer, scheduler=None, 
    num_epochs=num_epochs, is_inception=False, in_notebook=False)

# save the trained model
# Training complete in 7m 48s
# Best validation Loss: 0.740628 Acc: 0.8553

# adam
# Training complete in 7m 42s
# Best validation Loss: 0.718583 Acc: 0.8514
# 0.001
# Training complete in 11m 2s
# Best validation Loss: 0.522613 Acc: 0.8426
# 0.0005
# Training complete in 11m 14s
# Best validation Loss: 0.514895 Acc: 0.8368
torch.save(trained_model.state_dict(), 'model/resnet50_fe_adam_' + str(num_epochs) + 'epoch_simple_' + str(lr) + '.pt')
# display(full_performance_history.head())
# save the history
full_performance_history.to_csv('model/resnet50_fe_adam_' + str(num_epochs) + 'epoch_simple_' + str(lr) + '_history.csv', index=False)