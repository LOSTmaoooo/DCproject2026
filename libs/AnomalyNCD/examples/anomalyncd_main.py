import argparse
import os
import sys
sys.path.append(os.getcwd())
from models.AnomalyNCD import AnomalyNCD
from utils.general_utils import load_yaml

import warnings
warnings.filterwarnings("ignore")


def get_args():
    parser = argparse.ArgumentParser(description='AnomalyNCD', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    # ----------------------
    # dataset setting
    # ----------------------
    parser.add_argument('--dataset', type=str, default='mvtec', help='novel dataset name')
    parser.add_argument('--category', default=None, type=str, help='novel category name')
    parser.add_argument('--dataset_path', type=str, default=None, help='input novel image path')
    parser.add_argument('--anomaly_map_path', type=str, default=None, help='input novel anomaly map path')
    parser.add_argument('--binary_data_path', type=str, default=None, help='output novel binary mask path')
    parser.add_argument('--crop_data_path', type=str, default=None, help='output novel crop data path')
    parser.add_argument('--base_data_path', default=None, type=str, help='input base image path')

    # ----------------------
    # experiment setting
    # ----------------------
    parser.add_argument('--config', type=str, default='./configs/AnomalyNCD.yaml', help='config file path')
    parser.add_argument('--runner_name', default='AnomalyNCD', type=str)
    parser.add_argument('--only_test', type=str, default=None, help='test using the trained checkpoint')
    parser.add_argument('--checkpoint_path', type=str, default=None, help='path of the trained checkpoint')

    args = parser.parse_args()

    return args


def load_args(cfg, args):
    """
    Load args from the config file
    """
    # ----------------------
    # binariation setting
    # ----------------------
    args.sample_rate = cfg['binarization']['sample_rate']
    args.min_interval_len = cfg['binarization']['min_interval_len']
    args.erode = cfg['binarization']['erode']
    # ----------------------
    # model setting
    # ----------------------
    args.grad_from_block = cfg['models']['grad_from_block']
    args.pretrained_backbone = cfg['models']['pretrained_backbone']
    args.mask_layers = cfg['models']['mask_layers']
    args.n_views = cfg['models']['n_views']
    args.n_head = cfg['models']['n_head']
    # ----------------------
    # training setting
    # ----------------------
    args.batch_size = cfg['training']['batch_size']
    args.num_workers = cfg['training']['num_workers']
    args.lr = cfg['training']['lr']
    args.gamma = cfg['training']['gamma']
    args.momentum = cfg['training']['momentum']
    args.weight_decay = cfg['training']['weight_decay']
    args.epochs = cfg['training']['epochs']
    # ----------------------
    # loss setting
    # ----------------------
    args.sup_weight = cfg['loss']['sup_weight']
    args.memax_weight = cfg['loss']['memax_weight']
    args.anomaly_thred = cfg['loss']['anomaly_thred']
    args.teacher_temp = cfg['loss']['teacher_temp']
    args.warmup_teacher_temp = cfg['loss']['warmup_teacher_temp']
    args.warmup_teacher_temp_epochs = cfg['loss']['warmup_teacher_temp_epochs']
    args.repeat_times = cfg['loss']['repeat_times']
    # ----------------------
    # experiment setting
    # ----------------------
    args.seed = cfg['experiment']['seed']
    args.print_freq = cfg['experiment']['print_freq']
    args.table_root = cfg['experiment']['table_root']
    args.exp_name = cfg['experiment']['exp_name']
    args.exp_root = cfg['experiment']['exp_root']

    return args


if __name__ == "__main__":
    args = get_args()
    cfg = load_yaml(args.config)
    args = load_args(cfg, args)
    model = AnomalyNCD(args)
    model.main()