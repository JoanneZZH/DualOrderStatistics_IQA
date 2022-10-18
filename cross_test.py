import os
import argparse

import numpy as np
import pandas as pd
import torch
import torch.utils.data as Data
import scipy
import pretty_errors
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error

from dataset import LiveCDataSet, Tid2013DataSet, Kadid10KDataSet
from iqanet_final import IQANet_DGap
# from iqanet_init import IQANet_init
from main import distribution_to_mos


def validate(val_loader, model):
    result = ResultMeter()
    progress = ProgressMeter(0, [result], prefix='*Test')

    # switch to evaluate mode
    model.eval()
    with torch.no_grad():
        for i, (images, target) in enumerate(val_loader):
            images = images.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            output = model(images)
            result.update(output, target)

            # print(i)

    progress.display(0)

    return result


class ResultMeter(object):
    def __init__(self):
        self.y_pred_all = torch.zeros(0, dtype=torch.float32)
        self.y_all = torch.zeros(0, dtype=torch.float32)

    def update(self, y_pred, y):
        self.y_pred_all = torch.cat((self.y_pred_all, y_pred.cpu()), dim=0)
        self.y_all = torch.cat((self.y_all, y.cpu()), dim=0)

    def __str__(self):
        # y_all = self.y_all.detach().flatten().numpy()
        # y_pred_all = distribution_to_mos(self.y_pred_all)
        y_all = self.y_all
        y_all = y_all.detach().numpy()
        y_all = np.squeeze(y_all)
        y_pred_all = self.y_pred_all
        # y_pred_all = distribution_to_mos(self.y_pred_all)
        # y_pred_all = y_pred_all.detach().numpy()
        y_pred_all = np.squeeze(y_pred_all)

        self.PLCC = pearsonr(y_all, y_pred_all)[0]
        self.SRCC = spearmanr(y_all, y_pred_all)[0]
        self.RMSE = np.sqrt(mean_squared_error(y_all, y_pred_all))
        return 'PLCC=%.4f|SRCC=%.4f|RMSE=%.4f' % (self.PLCC, self.SRCC, self.RMSE)


class ProgressMeter(object):
    """Control print"""

    def __init__(self, num_epochs, meters, prefix=""):
        """
        Args:
            num_epochs (int): total number of all epochs
            meters (list): list of AverageMeter
            prefix (str, optional): Defaults to "".
        """
        self.epoch_fmtstr = self._get_epoch_fmtstr(num_epochs)
        self.meters = meters
        self.prefix = prefix

    def display(self, epoch, param_groups=None):
        entries = [self.prefix, self.epoch_fmtstr.format(epoch)]
        entries += [str(meter) for meter in self.meters]
        if param_groups is not None:
            entries += ["(lr:{})".format('/'.join(['{:.0e}'.format(p['lr'])
                                                for p in param_groups]))]
        print(' '.join(entries))

    def _get_epoch_fmtstr(self, num_epochs):
        num_digits = len(str(num_epochs // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_epochs) + ']'


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='cross test in Live-itW')
    parser.add_argument('--data', metavar='DIR', help='path to dataset')
    parser.add_argument('--checkpoint', metavar='CHECKPOINT',
                        help='file name of checkpoint(without suffix)')
    parser.add_argument('--dataset', metavar='Test Set', help='test whole dataset')
    args = parser.parse_args()

    if args.dataset == 'livec':
        # check image folder
        images_folder = os.path.join(args.data, 'Images')
        assert os.path.exists(images_folder), 'Live-itW images folder not exists'

        # check mos files
        mos_file = os.path.join(args.data, 'Data/livec_mos.csv')
        assert os.path.exists(mos_file), 'Live-itW mos file not exists'
        mos = pd.read_csv(mos_file, header=None)

        # validate
        # Because the dataset has multiple resolution images,
        # so the batch_size we set 1.
        loader = Data.DataLoader(dataset=LiveCDataSet(mos, images_folder),
                                batch_size=1, pin_memory=True)
    
    if args.dataset == 'tid2013':
        # check image folder
        images_folder = os.path.join(args.data, 'distorted_images')
        assert os.path.exists(images_folder), 'Tid2013 images folder not exists'

        # check mos files
        mos_file = os.path.join(args.data, 'tid13_mos.csv')
        assert os.path.exists(mos_file), 'Tid2013 mos file not exists'
        mos = pd.read_csv(mos_file, header=None)

        # validate
        # Because the dataset has multiple resolution images,
        # so the batch_size we set 1.
        loader = Data.DataLoader(dataset=Tid2013DataSet(mos, images_folder, 'False'),
                                batch_size=1, pin_memory=True)

    if args.dataset == 'kadid10k':
        # check image folder
        images_folder = os.path.join(args.data, 'images')
        assert os.path.exists(images_folder), 'Kadid images folder not exists'

        # check mos files
        mos_file = os.path.join(args.data, 'kadid10k_mos.csv')
        assert os.path.exists(mos_file), 'Kadid mos file not exists'
        mos = pd.read_csv(mos_file, header=None)

        # validate
        # Because the dataset has multiple resolution images,
        # so the batch_size we set 1.
        loader = Data.DataLoader(dataset=Kadid10KDataSet(mos, images_folder, 'False'),
                                batch_size=1, pin_memory=True)



    # load checkpoint
    checkpoint_path = 'checkpoints/{}.pth.tar'.format(args.checkpoint)
    assert os.path.exists(checkpoint_path), 'checkpoint not exists'
    state_dict = torch.load(checkpoint_path, map_location='cpu')['state_dict']

    # load model
    # model = IQANet_init()
    model = IQANet_DGap()
    model.load_state_dict(state_dict)
    model.cuda()

    result = validate(loader, model)

    # b = result.y_pred_all.cpu().data.numpy()
    # scipy.io.savemat('./test.mat',{'pred_dual':b})
    # c = result.y_all.cpu().data.numpy()
    # scipy.io.savemat('./test.mat',{'gt':c})

    debug = 1
