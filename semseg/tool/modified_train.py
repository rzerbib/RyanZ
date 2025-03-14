import os
import random
import time
import cv2
import numpy as np
import logging
import argparse

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.multiprocessing as mp
import torch.distributed as dist
from tensorboardX import SummaryWriter

from semseg.util import dataset, transform, config
from semseg.util.util import AverageMeter, poly_learning_rate, intersectionAndUnionGPU, find_free_port

from semseg.tool.FireSpreadDataset import FireSpreadDataset

input_nan, target_nan = [], []

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)


'''
import torchvision.transforms.functional as Ftf

class ResizePair(object):
    """Custom transform to resize both input x and label y to the same shape."""
    def __init__(self, size=(257,257)):
        self.size = size
    def __call__(self, x, y):
        # x shape: [C, H, W], y shape: [H, W]
        # Convert x, y to e.g. torch tensors if not already
        x_resized = Ftf.resize(x, self.size)  # bilinear for input
        y_resized = Ftf.resize(y.unsqueeze(0), self.size, interpolation=Ftf.InterpolationMode.NEAREST).squeeze(0)
        return x_resized, y_resized

train_transform = ResizePair(size=(257,257))
val_transform   = ResizePair(size=(257,257))
'''



def get_parser():
    parser = argparse.ArgumentParser(description='PyTorch Semantic Segmentation')
    parser.add_argument('--config', type=str, default='config/ade20k/ade20k_pspnet50.yaml', help='config file')
    parser.add_argument('opts', help='see config/ade20k/ade20k_pspnet50.yaml for all options', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    assert args.config is not None
    cfg = config.load_cfg_from_cfg_file(args.config)
    if args.opts is not None:
        cfg = config.merge_cfg_from_list(cfg, args.opts)
    return cfg


def get_logger():
    logger_name = "main-logger"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    fmt = "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)
    return logger


def worker_init_fn(worker_id):
    random.seed(args.manual_seed + worker_id)


def main_process():
    return not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % args.ngpus_per_node == 0)


def check(args):
    assert args.classes > 1
    assert args.zoom_factor in [1, 2, 4, 8]
    if args.arch == 'psp':
        assert (args.train_h - 1) % 8 == 0 and (args.train_w - 1) % 8 == 0
    elif args.arch == 'psa':
        if args.compact:
            args.mask_h = (args.train_h - 1) // (8 * args.shrink_factor) + 1
            args.mask_w = (args.train_w - 1) // (8 * args.shrink_factor) + 1
        else:
            assert (args.mask_h is None and args.mask_w is None) or (
                        args.mask_h is not None and args.mask_w is not None)
            if args.mask_h is None and args.mask_w is None:
                args.mask_h = 2 * ((args.train_h - 1) // (8 * args.shrink_factor) + 1) - 1
                args.mask_w = 2 * ((args.train_w - 1) // (8 * args.shrink_factor) + 1) - 1
            else:
                assert (args.mask_h % 2 == 1) and (args.mask_h >= 3) and (
                        args.mask_h <= 2 * ((args.train_h - 1) // (8 * args.shrink_factor) + 1) - 1)
                assert (args.mask_w % 2 == 1) and (args.mask_w >= 3) and (
                        args.mask_w <= 2 * ((args.train_h - 1) // (8 * args.shrink_factor) + 1) - 1)
    else:
        raise Exception('architecture not supported yet'.format(args.arch))


def main():
    args = get_parser()
    check(args)
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in args.train_gpu)
    if args.manual_seed is not None:
        random.seed(args.manual_seed)
        np.random.seed(args.manual_seed)
        torch.manual_seed(args.manual_seed)
        torch.cuda.manual_seed(args.manual_seed)
        torch.cuda.manual_seed_all(args.manual_seed)
        cudnn.benchmark = False
        cudnn.deterministic = True
    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])
    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    args.ngpus_per_node = len(args.train_gpu)
    if len(args.train_gpu) == 1:
        args.sync_bn = False
        args.distributed = False
        args.multiprocessing_distributed = False
    if args.multiprocessing_distributed:
        port = find_free_port()
        args.dist_url = f"tcp://127.0.0.1:{port}"
        args.world_size = args.ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=args.ngpus_per_node, args=(args.ngpus_per_node, args))
    else:
        main_worker(args.train_gpu, args.ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, argss):
    global args
    args = argss

    #explicitly import os as a module
    import os

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank)

    criterion = nn.CrossEntropyLoss(ignore_index=args.ignore_label)
    if args.arch == 'psp':
        from semseg.model.pspnet import PSPNet
        model = PSPNet(layers=args.layers, classes=args.classes, zoom_factor=args.zoom_factor, criterion=criterion)
        modules_ori = [model.layer0, model.layer1, model.layer2, model.layer3, model.layer4]
        modules_new = [model.ppm, model.cls, model.aux]
    elif args.arch == 'psa':
    
        from semseg.model.psanet import PSANet

        '''
        class CustomPSANet(nn.Module):
            def __init__(self, original_psanet):
                super(CustomPSANet, self).__init__()
                self.psanet = original_psanet
                self.input_layer = nn.Conv1d(in_channels=your_time_series_channels, 
                                            out_channels=3, kernel_size=1)  # Adjust input channels

            def forward(self, x):
                x = self.input_layer(x)  # Convert time series data into image-like format
                return self.psanet(x)
        '''

        original_psanet = PSANet(layers=args.layers, classes=args.classes, zoom_factor=args.zoom_factor, 
                                psa_type=args.psa_type, compact=args.compact, shrink_factor=args.shrink_factor, 
                                mask_h=args.mask_h, mask_w=args.mask_w, normalization_factor=args.normalization_factor, 
                                psa_softmax=args.psa_softmax, criterion=criterion)

        #model = CustomPSANet(original_psanet)  # Use the modified PSANet
        model = original_psanet




        modules_ori = [model.layer0, model.layer1, model.layer2, model.layer3, model.layer4]
        modules_new = [model.psa, model.cls, model.aux]
    params_list = []
    for module in modules_ori:
        params_list.append(dict(params=module.parameters(), lr=args.base_lr))
    for module in modules_new:
        params_list.append(dict(params=module.parameters(), lr=args.base_lr * 10))
    args.index_split = 5
    optimizer = torch.optim.SGD(params_list, lr=args.base_lr, momentum=args.momentum, weight_decay=args.weight_decay)
    if args.sync_bn:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

    if main_process():
        global logger, writer
        logger = get_logger()
        writer = SummaryWriter(args.save_path)
        logger.info(args)
        logger.info("=> creating model ...")
        logger.info("Classes: {}".format(args.classes))
        logger.info(model)
    if args.distributed:
        torch.cuda.set_device(gpu)
        args.batch_size = int(args.batch_size / ngpus_per_node)
        args.batch_size_val = int(args.batch_size_val / ngpus_per_node)
        args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
        model = torch.nn.parallel.DistributedDataParallel(model.cuda(), device_ids=[gpu])
    else:
        model = torch.nn.DataParallel(model.cuda())

    #if hasattr(args, 'weight') and args.weight:  # Ensure args.weight exists
    if args.weight:
        if os.path.isfile(args.weight):
            if main_process():
                logger.info("=> loading weight '{}'".format(args.weight))
            checkpoint = torch.load(args.weight)
            model.load_state_dict(checkpoint['state_dict'])
            if main_process():
                logger.info("=> loaded weight '{}'".format(args.weight))
        else:
            if main_process():
                logger.info("=> no weight found at '{}'".format(args.weight))

    if args.resume:
        if os.path.isfile(args.resume):
            if main_process():
                logger.info("=> loading checkpoint '{}'".format(args.resume))
            # checkpoint = torch.load(args.resume)
            checkpoint = torch.load(args.resume, map_location=lambda storage, loc: storage.cuda())
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'], strict=False)
            optimizer.load_state_dict(checkpoint['optimizer'])
            if main_process():
                logger.info("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            if main_process():
                logger.info("=> no checkpoint found at '{}'".format(args.resume))

    value_scale = 255
    mean = [0.485, 0.456, 0.406]
    mean = [item * value_scale for item in mean]
    std = [0.229, 0.224, 0.225]
    std = [item * value_scale for item in std]

    train_transform = transform.Compose([
        transform.Crop([257, 257], crop_type='center', padding=mean, ignore_label=args.ignore_label),
        #transform.ToTensor(),
        transform.Normalize(mean=mean, std=std)])


    train_data = FireSpreadDataset(
    data_dir="data",
    included_fire_years=[2018, 2019],
    n_leading_observations=1,
    crop_side_length=65,
    load_from_hdf5=True,
    is_train=True,
    remove_duplicate_features=False,
    stats_years=[2020, 2021],
    transform=train_transform)


    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
    else:
        train_sampler = None
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=8, shuffle=True, num_workers=4, drop_last=True)
    if args.evaluate:
        val_transform = transform.Compose([
            transform.Crop([257, 257], crop_type='center', padding=mean, ignore_label=args.ignore_label),
            #transform.ToTensor(),
            transform.Normalize(mean=mean, std=std)])
        val_data = FireSpreadDataset(
            data_dir="data",  # Make sure this path is correct
            included_fire_years=[2018, 2019],  # Adjust based on available years
            n_leading_observations=1,
            crop_side_length=65,
            load_from_hdf5=True,
            is_train=False,  # Ensure validation is correctly set
            remove_duplicate_features=False,
            stats_years=[2020, 2021],
            transform=val_transform,
        )
        global val_testing_debug
        val_testing_debug = len(val_data)
        if args.distributed:
            val_sampler = torch.utils.data.distributed.DistributedSampler(val_data)
        else:
            val_sampler = None
        val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.batch_size_val, shuffle=False, num_workers=args.workers, pin_memory=True, sampler=val_sampler)

    # 1A. Initialize lists to store train & val metrics
    train_loss_list, val_loss_list = [], []
    train_mIoU_list, val_mIoU_list = [], []
    train_mAcc_list, val_mAcc_list = [], []
    train_allAcc_list, val_allAcc_list = [], []

    import matplotlib.pyplot as plt
    import os

    def plot_metric(train_vals, val_vals, metric_name, save_dir):
        """
        Plots Train vs. Val curves for the specified metric, saves as PNG.
        """
        plt.figure(figsize=(8, 6))
        
        # Plot Train
        if len(train_vals) > 0:
            plt.plot(range(1, len(train_vals) + 1), train_vals,
                    label=f"Train {metric_name}", marker='o')
        
        # Plot Val
        if len(val_vals) > 0:
            plt.plot(range(1, len(val_vals) + 1), val_vals,
                    label=f"Val {metric_name}", marker='s')
        
        plt.xlabel("Epoch")
        plt.ylabel(metric_name)
        plt.title(f"Train vs. Val {metric_name}")
        plt.legend()
        plt.grid(True)

        # Save figure
        plot_path = os.path.join(save_dir, f"{metric_name.lower()}_plot.png")
        plt.savefig(plot_path, dpi=300)
        plt.close()
        print(f"✅ Saved {metric_name} plot to {plot_path}")

        
        
    for epoch in range(args.start_epoch, args.epochs):
        epoch_log = epoch + 1

        global loss_train, mIoU_train, mAcc_train, allAcc_train

        # 2A. Get train metrics from the train() function
        loss_train, mIoU_train, mAcc_train, allAcc_train = train(train_loader, model, optimizer, epoch)

        # 2B. Append them to the train lists
        train_loss_list.append(loss_train)
        train_mIoU_list.append(mIoU_train)
        train_mAcc_list.append(mAcc_train)
        train_allAcc_list.append(allAcc_train)

        if args.distributed:
            train_sampler.set_epoch(epoch)
        #loss_train, mIoU_train, mAcc_train, allAcc_train = train(train_loader, model, optimizer, epoch)
        if main_process():
            writer.add_scalar('loss_train', loss_train, epoch_log)
            writer.add_scalar('mIoU_train', mIoU_train, epoch_log)
            writer.add_scalar('mAcc_train', mAcc_train, epoch_log)
            writer.add_scalar('allAcc_train', allAcc_train, epoch_log)

        if (epoch_log % args.save_freq == 0) and main_process():
            filename = args.save_path + '/train_epoch_' + str(epoch_log) + '.pth'
            logger.info('Saving checkpoint to: ' + filename)
            torch.save({'epoch': epoch_log, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}, filename)
            if epoch_log / args.save_freq > 2:
                deletename = args.save_path + '/train_epoch_' + str(epoch_log - args.save_freq * 2) + '.pth'
                os.remove(deletename)
        if args.evaluate:
            loss_val, mIoU_val, mAcc_val, allAcc_val = validate(val_loader, model, criterion)
            
            # Append them to the val lists
            val_loss_list.append(loss_val)
            val_mIoU_list.append(mIoU_val)
            val_mAcc_list.append(mAcc_val)
            val_allAcc_list.append(allAcc_val)
            
            if main_process():
                writer.add_scalar('loss_val', loss_val, epoch_log)
                writer.add_scalar('mIoU_val', mIoU_val, epoch_log)
                writer.add_scalar('mAcc_val', mAcc_val, epoch_log)
                writer.add_scalar('allAcc_val', allAcc_val, epoch_log)

    # 3. After the training loop, plot everything
    png_dir = os.path.join(args.save_path, "training_plots")
    os.makedirs(png_dir, exist_ok=True)

    if main_process():  # Only the main process plots
        plot_metric(train_loss_list, val_loss_list, "Loss", png_dir)
        plot_metric(train_mIoU_list, val_mIoU_list, "mIoU", png_dir)
        plot_metric(train_mAcc_list, val_mAcc_list, "mAcc", png_dir)
        plot_metric(train_allAcc_list, val_allAcc_list, "AllAcc", png_dir)

        print(f"✅ Training & Validation plots saved in {png_dir}")



def train(train_loader, model, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    main_loss_meter = AverageMeter()
    aux_loss_meter = AverageMeter()
    loss_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()

    model.train()
    end = time.time()
    max_iter = args.epochs * len(train_loader)
    
    
    
    
    
    '''
    
    for i, (input, target) in enumerate(train_loader):
        global input_nan, target_nan
        # Immediately after reading input and target
        print("DEBUG: Checking input for NaNs or infinities...")
        if torch.isnan(input).any():
            input_nan.append("nans")
            print(f"Found NaNs in input at iteration={i}!")
        if torch.isinf(input).any():
            print(f"Found infinities in input at iteration={i}!")
            input_nan.append("infs")
        # Repeat for target
        print("DEBUG: Checking target for NaNs or infinities...")
        if torch.isnan(target).any():
            target_nan.append("nans")
            print(f"Found NaNs in target at iteration={i}!")
        if torch.isinf(target).any():
            target_nan.append("infs")
            print(f"Found infinities in target at iteration={i}!")
        
        
        data_time.update(time.time() - end)
        #if args.zoom_factor != 8:
            #h = int((target.size()[1] - 1) / 8 * args.zoom_factor + 1)
            #w = int((target.size()[2] - 1) / 8 * args.zoom_factor + 1)
            # 'nearest' mode doesn't support align_corners mode and 'bilinear' mode is fine for downsampling
            #target = F.interpolate(target.unsqueeze(1).float(), size=(h, w), mode='bilinear', align_corners=True).squeeze(1).long()
        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        
        #printing tensors as a debug
        print('Training tensors:')
        print(f'Input type: {input.dtype}, Input shape: {input.shape}')
        print(f'Target type: {target.dtype}, Target shape: {target.shape}')
        print(input[0, 0])
        #input = input.view(input.shape[0], -1, 65, 65)
        print(input[0, 0])

        # Extract channels 10, 15, and 23 from the original input
        input = input[:, [9, 14, 22], :, :]

    '''

    nan_log = []  # List to store NaN-related debugging messages

    for i, (input, target) in enumerate(train_loader):
        data_time.update(time.time() - end)

        # Move to GPU
        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # 🛑 DEBUG: Check for NaNs in input
        if torch.isnan(input).any():
            nan_log.append(f"⚠️ NaNs found in INPUT at iteration {i}.")

        # 🛑 DEBUG: Check for NaNs in target
        if torch.isnan(target).any():
            nan_log.append(f"⚠️ NaNs found in TARGET at iteration {i}.")

        # Forward pass
        # Select only 3 specific channels (change indices if needed)
        input = input[:, [9, 14, 22], :, :]  # Extract channels 10, 15, and 23 (0-based indexing)

        # Now pass to the model
        output, main_loss, aux_loss = model(input, target)


        # 🛑 DEBUG: Check for NaNs in model output
        if torch.isnan(output).any():
            nan_log.append(f"⚠️ NaNs found in MODEL OUTPUT at iteration {i}.")



        output, main_loss, aux_loss = model(input, target)
        if not args.multiprocessing_distributed:
            main_loss, aux_loss = torch.mean(main_loss), torch.mean(aux_loss)
        loss = main_loss + args.aux_weight * aux_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 🛑 DEBUG: Check for NaNs in loss
        if torch.isnan(loss).any():
            nan_log.append(f"⚠️ NaNs found in LOSS at iteration {i}.")

        n = input.size(0)
        if args.multiprocessing_distributed:
            main_loss, aux_loss, loss = main_loss.detach() * n, aux_loss * n, loss * n  # not considering ignore pixels
            count = target.new_tensor([n], dtype=torch.long)
            dist.all_reduce(main_loss), dist.all_reduce(aux_loss), dist.all_reduce(loss), dist.all_reduce(count)
            n = count.item()
            main_loss, aux_loss, loss = main_loss / n, aux_loss / n, loss / n

        intersection, union, target = intersectionAndUnionGPU(output, target, args.classes, args.ignore_label)
        if args.multiprocessing_distributed:
            dist.all_reduce(intersection), dist.all_reduce(union), dist.all_reduce(target)
        intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()
        intersection_meter.update(intersection), union_meter.update(union), target_meter.update(target)

        accuracy = sum(intersection_meter.val) / (sum(target_meter.val) + 1e-10)
        main_loss_meter.update(main_loss.item(), n)
        aux_loss_meter.update(aux_loss.item(), n)
        loss_meter.update(loss.item(), n)
        batch_time.update(time.time() - end)
        end = time.time()

        # ✅ Save NaN debug log at the end of training
        if nan_log:  # Only write if there were NaNs
            with open("nan_debug_log.txt", "w") as f:
                f.write("\n".join(nan_log))
            print("🚨 NaN Debug log saved to nan_debug_log.txt 🚨")
        else:
            print("✅ No NaNs detected during training.")


        current_iter = epoch * len(train_loader) + i + 1
        current_lr = poly_learning_rate(args.base_lr, current_iter, max_iter, power=args.power)
        for index in range(0, args.index_split):
            optimizer.param_groups[index]['lr'] = current_lr
        for index in range(args.index_split, len(optimizer.param_groups)):
            optimizer.param_groups[index]['lr'] = current_lr * 10
        remain_iter = max_iter - current_iter
        remain_time = remain_iter * batch_time.avg
        t_m, t_s = divmod(remain_time, 60)
        t_h, t_m = divmod(t_m, 60)
        remain_time = '{:02d}:{:02d}:{:02d}'.format(int(t_h), int(t_m), int(t_s))

        if (i + 1) % args.print_freq == 0 and main_process():
            logger.info('Epoch: [{}/{}][{}/{}] '
                        'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                        'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                        'Remain {remain_time} '
                        'MainLoss {main_loss_meter.val:.4f} '
                        'AuxLoss {aux_loss_meter.val:.4f} '
                        'Loss {loss_meter.val:.4f} '
                        'Accuracy {accuracy:.4f}.'.format(epoch+1, args.epochs, i + 1, len(train_loader),
                                                          batch_time=batch_time,
                                                          data_time=data_time,
                                                          remain_time=remain_time,
                                                          main_loss_meter=main_loss_meter,
                                                          aux_loss_meter=aux_loss_meter,
                                                          loss_meter=loss_meter,
                                                          accuracy=accuracy))
        if main_process():
            writer.add_scalar('loss_train_batch', main_loss_meter.val, current_iter)
            writer.add_scalar('mIoU_train_batch', np.mean(intersection / (union + 1e-10)), current_iter)
            writer.add_scalar('mAcc_train_batch', np.mean(intersection / (target + 1e-10)), current_iter)
            writer.add_scalar('allAcc_train_batch', accuracy, current_iter)

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    mIoU = np.mean(iou_class)
    mAcc = np.mean(accuracy_class)
    allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)
    if main_process():
        logger.info('Train result at epoch [{}/{}]: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(epoch+1, args.epochs, mIoU, mAcc, allAcc))
    return main_loss_meter.avg, mIoU, mAcc, allAcc


def validate(val_loader, model, criterion):
    if main_process():
        logger.info('>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>')
    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()

    model.eval()
    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        data_time.update(time.time() - end)
        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        
        #Printing tensors as a debug
        print('Validation tensors:')
        print(f'Input type: {input.dtype}, Input shape: {input.shape}')
        print(f'Target type: {target.dtype}, Target shape: {target.shape}') 
        
        # Squeeze out any extra dimension if necessary (e.g., if shape is [N, 1, 40, H, W])
        input = input.squeeze(1)  # [N, 40, H, W]

        # Slice channels 10th, 15th, and 23rd → indices (9, 14, 22)
        input = input[:, [9, 14, 22], :, :]  # Now shape is [N, 3, H, W]

         # 2) Fix height/width to satisfy (H-1)%8=0
        # Optionally do a resize if your transforms didn't fix it
        #input = F.interpolate(input, size=(257, 257), mode='bilinear', align_corners=False)

        # 4) *Instead*, resize your target to match 257×257
        #target = F.interpolate(
            #target.unsqueeze(1).float(),       # shape: [N,1,H,W]
            #size=(257, 257),                  # <-- ADDED/CHANGED
            #mode='nearest'                    # important for integer labels
        #).squeeze(1).long()                   # back to shape [N,H,W]

        # Now forward pass
        output = model(input)                 # output shape => [N, Classes, 257, 257]


        #if args.zoom_factor != 8:
            #output = F.interpolate(output, size=target.size()[1:], mode='bilinear', align_corners=True)
        loss = criterion(output, target)

        n = input.size(0)
        if args.multiprocessing_distributed:
            loss = loss * n  # not considering ignore pixels
            count = target.new_tensor([n], dtype=torch.long)
            dist.all_reduce(loss), dist.all_reduce(count)
            n = count.item()
            loss = loss / n
        else:
            loss = torch.mean(loss)

        output = output.max(1)[1]
        intersection, union, target = intersectionAndUnionGPU(output, target, args.classes, args.ignore_label)
        if args.multiprocessing_distributed:
            dist.all_reduce(intersection), dist.all_reduce(union), dist.all_reduce(target)
        intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()
        intersection_meter.update(intersection), union_meter.update(union), target_meter.update(target)

        accuracy = sum(intersection_meter.val) / (sum(target_meter.val) + 1e-10)
        loss_meter.update(loss.item(), input.size(0))
        batch_time.update(time.time() - end)
        end = time.time()
        if ((i + 1) % args.print_freq == 0) and main_process():
            logger.info('Test: [{}/{}] '
                        'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                        'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                        'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f}) '
                        'Accuracy {accuracy:.4f}.'.format(i + 1, len(val_loader),
                                                          data_time=data_time,
                                                          batch_time=batch_time,
                                                          loss_meter=loss_meter,
                                                          accuracy=accuracy))

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    mIoU = np.mean(iou_class)
    mAcc = np.mean(accuracy_class)
    allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)
    if main_process():
        logger.info('Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(mIoU, mAcc, allAcc))
        for i in range(args.classes):
            logger.info('Class_{} Result: iou/accuracy {:.4f}/{:.4f}.'.format(i, iou_class[i], accuracy_class[i]))
        logger.info('<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<')

        #LIST DEBUGGING
        print('<<<<<<<<<<<<<<<<< LIST DEBUGGING <<<<<<<<<<<<<<<<<')
        print('loss_train')
        print(loss_train)
        print('mIoU_train')
        print(mIoU_train)
        print('mAcc_train')
        print(mAcc_train)
        print('allAcc_train')
        print(allAcc_train)

        print(f"evaluation running: {args.evaluate}")
        
        print("DEBUG: len(val_data) =", val_testing_debug)

        print(f"DEBUG: input has nans/infs? {input_nan}")
        print(f"DEBUG: target has nans/infs? {target_nan}")


    return loss_meter.avg, mIoU, mAcc, allAcc




if __name__ == '__main__':
    main()

