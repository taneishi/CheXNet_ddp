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

    val_dataset = ChestXrayDataSet(
            data_dir=args.data_dir,
            image_list_file=args.val_image_list,
            transform=transform,
            )

    val_loader = torch.utils.data.DataLoader(
            dataset=val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            pin_memory=False)

    print('training %d batches %d images' % (len(train_loader), len(train_dataset)))
    print('validation %d batches %d images' % (len(val_loader), len(val_dataset)))

    # initialize and load the model
    net = DenseNet121(N_CLASSES)

    if args.model_path:
        net.load_state_dict(torch.load(args.model_path, map_location=device))
        print('model state has loaded.')

    if args.dataparallel and torch.cuda.device_count() > 1:
        net = torch.nn.DataParallel(net)
        print('Using %d cuda devices.' % (torch.cuda.device_count()))

    net = net.to(device)

    if args.hpu:
        permute_params(net, True, args.use_lazy_mode)
        permute_momentum(optimizer, True, args.use_lazy_mode)

    criterion = torch.nn.CrossEntropyLoss()
    #criterion = torch.nn.MultiLabelSoftMarginLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum)

    for epoch in range(args.epochs):
        start_time = timeit.default_timer()

        # initialize the ground truth and output tensor
        y_true = torch.FloatTensor()
        y_pred = torch.FloatTensor()
        train_loss = 0
        net.train()
        for index, (images, labels) in enumerate(train_loader, 1):
            # each image has 10 crops
            batch_size, n_crops, c, h, w = images.size()
            images = images.view(-1, c, h, w).to(device)
            labels = labels.to(device)

            outputs = net(images)

            outputs_mean = outputs.view(batch_size, n_crops, -1).mean(1)
            loss = criterion(outputs_mean, labels)
            train_loss += loss.item()

            y_true = torch.cat((y_true, labels.cpu()))
            y_pred = torch.cat((y_pred, outputs_mean.detach().cpu()))

            # backward and optimize
            optimizer.zero_grad()
            loss.backward()
            if args.hpu:
                htcore.mark_step()
            optimizer.step()
            if args.hpu:
                htcore.mark_step()

            print('\repoch %3d batch %5d/%5d train loss %6.4f' % (epoch+1, index, len(train_loader), train_loss / index), end='')
            print(' %6.3fsec' % (timeit.default_timer() - start_time), end='')

            aucs = [roc_auc_score(y_true[:, i], y_pred[:, i]) if y_true[:, i].sum() > 0 else np.nan for i in range(N_CLASSES)]
            print(' average AUC %5.3f' % (np.mean(aucs)), end='')

        print('')

        y_true = torch.FloatTensor()
        y_pred = torch.FloatTensor()

        net.eval()
        for index, (images, labels) in enumerate(val_loader, 1):
            start_time = timeit.default_timer()

            batch_size, n_crops, c, h, w = images.size()
            images = images.view(-1, c, h, w).to(device)

            with torch.no_grad():
                outputs = net(images)

            outputs_mean = outputs.view(batch_size, n_crops, -1).mean(1)

            y_true = torch.cat((y_true, labels.cpu()))
            y_pred = torch.cat((y_pred, outputs_mean.cpu()))

            print('\repoch %3d batch %4d/%4d %6.3fsec' % (epoch+1, index, len(val_loader), (timeit.default_timer() - start_time)), end='')

        aucs = [roc_auc_score(y_true[:, i], y_pred[:, i]) for i in range(N_CLASSES)]
        auc_classes = ' '.join(['%5.3f' % (aucs[i]) for i in range(N_CLASSES)])
        print(' average AUC %5.3f (%s)' % (np.mean(aucs), auc_classes))

        torch.save(net.state_dict(), 'model/checkpoint.pth')

    test_dataset = ChestXrayDataSet(
            data_dir=args.data_dir,
            image_list_file=args.test_image_list,
            transform=transform,
            )

    test_loader = torch.utils.data.DataLoader(
            dataset=test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            pin_memory=False)

    print('test %d batches %d images' % (len(test_loader), len(test_dataset)))

    y_true = torch.FloatTensor()
    y_pred = torch.FloatTensor()

    net.eval()
    for index, (images, labels) in enumerate(test_loader, 1):
        start_time = timeit.default_timer()

        batch_size, n_crops, c, h, w = images.size()
        images = images.view(-1, c, h, w)

        with torch.no_grad():
            outputs = net(images)

        output_mean = outputs.view(batch_size, n_crops, -1).mean(1)

        y_true = torch.cat((y_true, labels.cpu()))
        y_pred = torch.cat((y_pred, outputs_mean.cpu()))

        print('\r%4d/%4d, time: %6.3fsec' % (index, len(test_loader), (timeit.default_timer() - start_time)), end='')

        aucs = [roc_auc_score(y_true[:, i], y_pred[:, i]) if y_true[:, i].sum() > 0 else np.nan for i in range(N_CLASSES)]
        auc_classes = ' '.join(['%5.3f' % (aucs[i]) for i in range(N_CLASSES)])
        print(' average AUC %5.3f (%s)' % (np.mean(aucs), auc_classes), end='')
    print('')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default=None, type=str)
    parser.add_argument('--epochs', default=150, type=int)
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--data_dir', default='images', type=str)
    parser.add_argument('--hpu', action='store_true', default=False)
    parser.add_argument('--use_lazy_mode', action='store_true', default=True)
    parser.add_argument('--dataparallel', action='store_true', default=False)
    parser.add_argument('--train_image_list', default='labels/train_list.txt', type=str)
    parser.add_argument('--val_image_list', default='labels/val_list.txt', type=str)
    parser.add_argument('--test_image_list', default='labels/test_list.txt', type=str)
    args = parser.parse_args()
    print(vars(args))

    main(args)
