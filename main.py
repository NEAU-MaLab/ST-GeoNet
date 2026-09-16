"""
@Author: QiHeng Wu
@Contact: s.t_labmem001@foxmail.com
@File: main.py
Training and evaluation entry point for ST-GeoNet field-road segmentation.
@Time: 2025/4/16 1:49 PM
"""

from __future__ import print_function
import copy
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR

from ST_GeoNet_data import TrajectoryDataset
from ST_GeoNet_model import ST_GeoNet
from ST_GeoNet_util import cal_loss, IOStream, load_checkpoint, save_checkpoint
from trajectory_schema import make_feature_config
from torch.utils.data import DataLoader
import sklearn.metrics as metrics

PROJECT_ROOT = Path(__file__).resolve().parent

def project_path(value):
    """Resolve a user path against the project directory on Windows or Linux."""
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path

def parse_bool(value):
    """Parse explicit True/False CLI values without treating 'False' as true."""
    if isinstance(value, bool):
        return value
    if value.lower() in ('true', '1', 'yes', 'y'):
        return True
    if value.lower() in ('false', '0', 'no', 'n'):
        return False
    raise argparse.ArgumentTypeError('Expected True or False.')

def _init_():
    """Create the project-local experiment and checkpoint directories."""
    (args.output_dir / 'models').mkdir(parents=True, exist_ok=True)

def train(args, io):
    """Train ST-GeoNet and report point-level metrics on the test split."""
    train_dataset = TrajectoryDataset(
        path=args.data_path, partition='train', num_points=args.num_points,
        T_D=args.T_D_point_cloud, extra_features=args.extra_features,
        motion_source=args.motion_source)
    test_dataset = TrajectoryDataset(
        path=args.data_path, partition='test', num_points=args.num_points,
        T_D=args.T_D_point_cloud, extra_features=args.extra_features,
        motion_source=args.motion_source)
    train_loader = DataLoader(train_dataset, num_workers=0, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, num_workers=0, batch_size=args.test_batch_size, shuffle=False)

    device = torch.device("cuda" if args.cuda else "cpu")

    # Construct the network and load a checkpoint when requested.
    model = ST_GeoNet(args).to(device)

    model = nn.DataParallel(model)
    if args.model_path is not None:
        load_checkpoint(model, args.model_path, device)
    print("Let's use", torch.cuda.device_count(), "GPUs!")

    if args.use_sgd:
        print("Use SGD")
        opt = optim.SGD(model.parameters(), lr=args.lr * 100, momentum=args.momentum, weight_decay=1e-4)
    else:
        print("Use Adam")
        opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    if args.scheduler == 'cos':
        scheduler = CosineAnnealingLR(opt, args.epochs, eta_min=1e-3)
    elif args.scheduler == 'step':
        scheduler = StepLR(opt, step_size=20, gamma=0.5)

    criterion = cal_loss

    best_test_acc = 0
    for epoch in range(args.epochs):
        # Optimize on the training split, excluding labels equal to -1.
        train_loss = 0.0
        count = 0.0
        model.train()
        train_true_cls = []
        train_pred_cls = []
        
        for data, seg, mask, adj in train_loader:

            data, seg, mask, adj = data.to(device), seg.to(device), mask.to(device), adj.to(device)
            batch_size = data.size()[0]
            opt.zero_grad()
            seg_pred = model(data, mask=mask, adj=adj)

            # Flatten point labels and two-class logits for cross-entropy.
            seg_flat = seg.view(-1)
            pred_flat = seg_pred.reshape(-1, 2)
            loss = criterion(pred_flat, seg_flat)

            loss.backward()
            opt.step()
            pred = seg_pred.max(dim=2)[1]  # (batch_size, num_points)
            count += batch_size
            train_loss += loss.item() * batch_size
            seg_np = seg.cpu().numpy()  # (batch_size, num_points)
            pred_np = pred.detach().cpu().numpy()  # (batch_size, num_points)
            train_true_cls.append(seg_np[seg_np != -1].reshape(-1))  # One entry per valid, non-padding point.
            train_pred_cls.append(pred_np[seg_np != -1].reshape(-1))  # One entry per valid, non-padding point.
        if args.scheduler == 'cos':
            scheduler.step()
        elif args.scheduler == 'step':
            if opt.param_groups[0]['lr'] > 1e-5:
                scheduler.step()
            if opt.param_groups[0]['lr'] < 1e-5:
                for param_group in opt.param_groups:
                    param_group['lr'] = 1e-5
        train_true_cls = np.concatenate(train_true_cls)
        train_pred_cls = np.concatenate(train_pred_cls)
        train_acc = metrics.accuracy_score(train_true_cls, train_pred_cls)
        avg_per_class_acc = metrics.balanced_accuracy_score(train_true_cls, train_pred_cls)
        outstr = 'Train %d, loss: %.6f, train acc: %.6f, train avg acc: %.6f' % (epoch,
                                                                                 train_loss * 1.0 / count,
                                                                                 train_acc,
                                                                                 avg_per_class_acc,)
        io.cprint(outstr)

        # Evaluate every five epochs and retain improvements in point accuracy.
        if epoch % 5 == 0:
            test_loss = 0.0
            count = 0.0
            model.eval()
            test_true_cls = []
            test_pred_cls = []

            for data, seg, mask, adj in test_loader:

                data, seg, mask, adj = data.to(device), seg.to(device), mask.to(device), adj.to(device)
                batch_size = data.size()[0]
                seg_pred = model(data, mask=mask, adj=adj)

                seg_flat = seg.view(-1)
                pred_flat = seg_pred.reshape(-1, 2)
                loss = criterion(pred_flat, seg_flat)

                pred = seg_pred.max(dim=2)[1]
                count += batch_size
                test_loss += loss.item() * batch_size
                seg_np = seg.cpu().numpy()
                pred_np = pred.detach().cpu().numpy()
                test_true_cls.append(seg_np[seg_np != -1].reshape(-1))
                test_pred_cls.append(pred_np[seg_np != -1].reshape(-1))
            test_true_cls = np.concatenate(test_true_cls)
            test_pred_cls = np.concatenate(test_pred_cls)
            test_acc = metrics.accuracy_score(test_true_cls, test_pred_cls)
            test_pre = metrics.precision_score(test_true_cls, test_pred_cls)
            test_rec = metrics.recall_score(test_true_cls, test_pred_cls)
            test_fl = metrics.f1_score(test_true_cls, test_pred_cls)
            # Invert binary labels to compute precision/recall/F1 for class 0.
            test_true_cls_n = 1 - test_true_cls
            test_pred_cls_n = 1 - test_pred_cls
            test_pre_n = metrics.precision_score(test_true_cls_n, test_pred_cls_n)
            test_rec_n = metrics.recall_score(test_true_cls_n, test_pred_cls_n)
            test_fl_n = metrics.f1_score(test_true_cls_n, test_pred_cls_n)
            outstr = 'Test %d, loss: %.6f, test acc: %.6f, test pre: %.6f, test rec: %.6f, test f1: %.6f, test pre_n: %.6f, test rec_n: %.6f, test f1_n: %.6f' % (
                epoch,
                test_loss * 1.0 / count,
                test_acc,
                test_pre, test_rec, test_fl, test_pre_n, test_rec_n, test_fl_n)
            io.cprint(outstr)
            if test_acc > best_test_acc:
                best_test_acc = copy.deepcopy(test_acc)
                save_checkpoint(model, args.output_dir / 'models' / ('epoch_%d_model.t7' % epoch))

def test(args, io):
    """Evaluate the selected checkpoint, excluding padding from all metrics."""
    test_dataset = TrajectoryDataset(
        path=args.data_path, partition='test', num_points=args.num_points,
        T_D=args.T_D_point_cloud, extra_features=args.extra_features,
        motion_source=args.motion_source)
    test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size,
                             shuffle=True, drop_last=False)
    device = torch.device("cuda" if args.cuda else "cpu")

    # Construct the network and load a checkpoint when requested.
    model = ST_GeoNet(args).to(device)
    model = nn.DataParallel(model)
    load_checkpoint(model, args.model_path, device)
    model = model.eval()
    test_loss = 0.0
    count = 0.0
    test_true_cls = []
    test_pred_cls = []
    for data, seg, mask, adj in test_loader:
        data, seg, mask, adj = data.to(device), seg.to(device), mask.to(device), adj.to(device)
        batch_size = data.size()[0]
        seg_pred = model(data, mask=mask, adj=adj)

        seg_flat = seg.view(-1)
        pred_flat = seg_pred.reshape(-1, 2)
        loss = cal_loss(pred_flat, seg_flat)

        pred = seg_pred.max(dim=2)[1]
        count += batch_size
        test_loss += loss.item() * batch_size
        seg_np = seg.cpu().numpy()
        pred_np = pred.detach().cpu().numpy()
        test_true_cls.append(seg_np[seg_np != -1].reshape(-1))
        test_pred_cls.append(pred_np[seg_np != -1].reshape(-1))
    test_true_cls = np.concatenate(test_true_cls)
    test_pred_cls = np.concatenate(test_pred_cls)
    test_acc = metrics.accuracy_score(test_true_cls, test_pred_cls)
    test_pre = metrics.precision_score(test_true_cls, test_pred_cls)
    test_rec = metrics.recall_score(test_true_cls, test_pred_cls)
    test_fl = metrics.f1_score(test_true_cls, test_pred_cls)
    test_true_cls_n = 1 - test_true_cls
    test_pred_cls_n = 1 - test_pred_cls
    test_pre_n = metrics.precision_score(test_true_cls_n, test_pred_cls_n)
    test_rec_n = metrics.recall_score(test_true_cls_n, test_pred_cls_n)
    test_fl_n = metrics.f1_score(test_true_cls_n, test_pred_cls_n)
    outstr = 'Test: loss: %.6f, test acc: %.6f, test pre: %.6f, test rec: %.6f, test f1: %.6f, test pre_n: %.6f, test rec_n: %.6f, test f1_n: %.6f' % (
        test_loss * 1.0 / count,
        test_acc,
        test_pre, test_rec, test_fl, test_pre_n, test_rec_n, test_fl_n)
    io.cprint(outstr)

if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='ST-GeoNet field-road trajectory segmentation')
    parser.add_argument('--exp_name', type=str, default='ST_GeoNet', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--batch_size', type=int, default=18, metavar='batch_size',
                        help='Number of trajectories per batch')
    parser.add_argument('--test_batch_size', type=int, default=1, metavar='batch_size',
                        help='Number of trajectories per batch')
    parser.add_argument('--epochs', type=int, default=4000, metavar='N',
                        help='Number of training epochs')
    parser.add_argument('--use_sgd', type=parse_bool, default=True,
                        help='Use SGD')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                        help='Base learning rate; SGD uses 100 times this value')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--scheduler', type=str, default='cos', metavar='N',
                        choices=['cos', 'step'],
                        help='Scheduler to use, [cos, step]')
    parser.add_argument('--no_cuda', type=parse_bool, default=False,
                        help='Disable CUDA when True')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--eval', type=parse_bool, default=False,
                        help='evaluate the model')
    parser.add_argument('--num_points', type=int, default=4096,
                        help='num of points to use')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout rate')
    parser.add_argument('--emb_dims', type=int, default=1024, metavar='N',
                        help='Dimension of embeddings')
    parser.add_argument('--graph_feature_dims', type=int, default=32, metavar='N',
                        help='Dimension of graph feature')
    parser.add_argument('--T_D_point_cloud', type=parse_bool, default=False,
                        help='Retained three-channel dataset option; incompatible with the default model')
    parser.add_argument('--k', type=int, default=55, metavar='N',
                        help='Num of nearest neighbors to use')
    parser.add_argument('--model_path', type=str, default=None, metavar='N',
                        help='Checkpoint path; required for evaluation, optional for training')
    parser.add_argument('--data_path', type=str, default=str(PROJECT_ROOT / 'Graph&degree_gen' / 'Degree_data'), metavar='N',
                        help='Dataset path')
    parser.add_argument('--extra_features', nargs='*', default=[], metavar='COLUMN',
                        help='Ordered extra numeric CSV columns to include after the two motion features')
    parser.add_argument('--motion_source', choices=['derived', 'csv'], default='derived',
                        help='Use coordinate-derived distance/angle or the CSV speed/direction columns')
    args = parser.parse_args()
    try:
        feature_config = make_feature_config(args.extra_features, args.motion_source)
    except ValueError as error:
        parser.error(str(error))
    args.extra_features = feature_config['extra_features']
    args.data_path = str(project_path(args.data_path))
    if args.model_path is not None:
        args.model_path = str(project_path(args.model_path))
    args.output_dir = PROJECT_ROOT / 'outputs' / args.exp_name
    if args.T_D_point_cloud:
        parser.error('--T_D_point_cloud must be False for ST-GeoNet.')
    if args.eval and args.model_path is None:
        parser.error('--eval True requires --model_path.')

    _init_()

    io = IOStream(args.output_dir / 'run.log')
    io.cprint(str(args))

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    if args.cuda:
        io.cprint(
            'Using GPU : ' + str(torch.cuda.current_device()) + ' from ' + str(torch.cuda.device_count()) + ' devices')
        torch.cuda.manual_seed(args.seed)
    else:
        io.cprint('Using CPU')

    if not args.eval:
        train(args, io)
    else:
        test(args, io)
