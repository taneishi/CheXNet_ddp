import numpy as np
import torchvision
from torchvision import transforms
import torch
from sklearn.metrics import roc_auc_score
import argparse
import timeit
import os

from datasets import ChestXrayDataSet
from model import DenseNet121, CLASS_NAMES, N_CLASSES

def permute_params(model, to_filters_last, lazy_mode):
    if htcore.is_enabled_weight_permute_pass() is True:
        return

    with torch.no_grad():
        for name, param in model.named_parameters():
            if(param.ndim == 4):
                if to_filters_last:
                    param.data = param.data.permute((2, 3, 1, 0))
                else:
                    param.data = param.data.permute((3, 2, 0, 1))  # permute RSCK to KCRS

    if lazy_mode:
        htcore.mark_step()

def permute_momentum(optimizer, to_filters_last, lazy_mode):
    # Permute the momentum buffer before using for checkpoint
    if htcore.is_enabled_weight_permute_pass() is True:
        return

    for group in optimizer.param_groups:
        for p in group['params']:
            param_state = optimizer.state[p]
            if 'momentum_buffer' in param_state:
                buf = param_state['momentum_buffer']
                if(buf.ndim == 4):
                    if to_filters_last:
                        buf = buf.permute((2,3,1,0))
                    else:
                        buf = buf.permute((3,2,0,1))
                    param_state['momentum_buffer'] = buf

    if lazy_mode:
        htcore.mark_step()

def main(args):
    if args.hpu:
        from habana_frameworks.torch.utils.library_loader import load_habana_module
        import habana_frameworks.torch.core as htcore
        load_habana_module()
        device = torch.device('hpu')
        torch.cuda.current_device = lambda: None
        torch.cuda.set_device = lambda x: None
        os.environ['PT_HPU_LAZY_MODE'] = '1'
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('Using %s device.' % (device))

    normalize = transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225])

    transform = torchvision.transforms.Compose([
        transforms.Resize(256),
        transforms.TenCrop(224),
        transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
        transforms.Lambda(lambda crops: torch.stack([normalize(crop) for crop in crops]))
        ])

    train_dataset = ChestXrayDataSet(
            data_dir=args.data_dir,
            image_list_file=args.train_image_list,
            transform=transform,
            )

    train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset, 
            batch_size=args.batch_size, 
            shuffle=True, 
            pin_memory=False)

    print('training %d batches %d images' % (len(train_loader), len(train_dataset)))
    
    # initialize and load the model
    net = DenseNet121(N_CLASSES)
    if args.model_path:
        net.load_state_dict(torch.load(args.model_path, map_location=device))
        print('model state has loaded')

    if torch.cuda.device_count() > 1:
        net = torch.nn.DataParallel(net)
        print('Using %d cuda devices' % (torch.cuda.device_count()))

    net = net.to(device)

    optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum)

    if args.hpu:
        permute_params(net, True, args.use_lazy_mode)
        permute_momentum(optimizer, True, args.use_lazy_mode)

    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        start_time = timeit.default_timer()

        train_loss = 0
        net.train()
        for index, (images, labels) in enumerate(train_loader):
            # each image has 10 crops.
            batch_size, n_crops, c, h, w = images.size()
            images = images.view(-1, c, h, w).to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = net(images)
            outputs = outputs.view(batch_size, n_crops, -1).mean(1)
            loss = criterion(outputs, labels)

            # backward and optimize
            loss.backward()
            if args.hpu:
                htcore.mark_step()
            optimizer.step()
            if args.hpu:
                htcore.mark_step()
            train_loss += loss.item()

            print('\repoch %3d batch %5d/%5d train loss %6.4f' % (epoch+1, index+1, len(train_loader), train_loss / (index+1)), end='')
            print(' %6.3fsec' % (timeit.default_timer() - start_time), end='')
        print('')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default=None, type=str)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--data_dir', default='images', type=str)
    parser.add_argument('--train_image_list', default='labels/bmt_list.txt', type=str)
    parser.add_argument('--hpu', action='store_true', default=False)
    parser.add_argument('--use_lazy_mode', action='store_true', default=True)
    args = parser.parse_args()
    print(vars(args))

    main(args)
