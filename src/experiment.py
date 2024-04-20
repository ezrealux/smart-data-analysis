from model import MultiModel, WeightedSumLoss, SimpleLoss
from dataset import DataProcessor
from argparse import ArgumentParser

from datetime import datetime

import configparser
import torch

def main():
    parser = ArgumentParser()
    parser.add_argument("--start_date", type=str, default='20210101')
    parser.add_argument("--end_date", type=str, default='20211231')
    parser.add_argument("--on_gpu", action='store_true')
    
    start_date = datetime.strptime(args.start_date, '%Y%m%d')
    end_date = datetime.strptime(args.end_date, '%Y%m%d')

    config = configparser.ConfigParser()
    config.read('config.ini')

    print('Start')
    args = parser.parse_args()
    use_gpu = torch.cuda.is_available() & args.on_gpu
    print('Use gpu:', use_gpu)
    device = torch.device(f"cuda:0" if use_gpu else "cpu")

    print('Preprocessing...')
    dp = DataProcessor('../dataset/','2009_2023','TWII.csv', 'prs_dataset_no_fat(clean)')
    dp()
    data_gen = dp.Prepare_test_data(start_date, end_date, args.epochs)
    print('End of data preprocessing.')

    print('Model loading...')
    model_path = config['save_path']['model_path']
    Model = MultiModel().to(device)
    Model.load_state_dict(torch.load(model_path))
    loss_function = SimpleLoss().to(device)

if __name__ == '__main__':
    main()