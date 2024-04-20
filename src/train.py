from model import MultiModel, WeightedSumLoss, SimpleLoss
from dataset import DataProcessor

from argparse import ArgumentParser
import configparser

import torch
from torch import optim
from tqdm import tqdm

import os
import matplotlib.pyplot as plt
from datetime import datetime
import time
import math

def printEpoch(epoch, loss):
    epoch = epoch+1
    if epoch % 10**int(math.log10(epoch)) == 0:
        print('Epoch:', epoch, ', loss:', loss)

def main():
    parser = ArgumentParser()
    parser.add_argument("--train_start_date", type=str, default='20140101')
    parser.add_argument("--train_end_date", type=str, default='20201231')
    parser.add_argument("--test_start_date", type=str, default='20210101')
    parser.add_argument("--test_end_date", type=str, default='20211231')
    parser.add_argument("--on_gpu", action='store_true')
    parser.add_argument("--epochs", type=int, default=20)

    config = configparser.ConfigParser()
    config.read('config.ini')

    print('Start')
    args = parser.parse_args()
    use_gpu = torch.cuda.is_available() & args.on_gpu
    print('Use gpu:', use_gpu)
    device = torch.device(f"cuda:0" if use_gpu else "cpu")

    train_start_date = datetime.strptime(args.train_start_date, '%Y%m%d')
    train_end_date = datetime.strptime(args.train_end_date, '%Y%m%d')
    test_start_date = datetime.strptime(args.test_start_date, '%Y%m%d')
    test_end_date = datetime.strptime(args.test_end_date, '%Y%m%d')
    
    print('Preprocessing...')
    dp = DataProcessor('../dataset/','2009_2023','TWII.csv', 'prs_dataset_no_fat(clean)')
    dp()
    data_gen = dp.Prepare_train_data(train_start_date, train_end_date, args.epochs)
    print('End of data preprocessing.')

    print('Model generating...')
    learning_rate = config['model_sett'].getfloat('learning_rate')
    ensemble_num = config['model_sett'].getint('ensemble_num')
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(42)
    Model = MultiModel(ensemble_num=ensemble_num).to(device)
    #loss_function = WeightedSumLoss()
    loss_function = SimpleLoss().to(device)
    optimizer = optim.AdamW(Model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[i for i in range(500, args.epochs, 500)], gamma=0.7)

    print('Training...')
    running_loss = []
    #for epoch, train_loader, test_loader, c6_train_loader, c6_test_loader, tau_train, logm_train, y_train, tau_c6_train, logm_c6_train in data_gen:
    #for epoch, tau_train, logm_train, y_train, yATM_train, tau_c6_train, logm_c6_train, yATM_c6_train in data_gen:
    for epoch, tau_train, logm_train, y_train, yATM_train in data_gen:
        Model.train()

        tau_train, logm_train, y_train, yATM_train = tau_train.to(device), logm_train.to(device), y_train.to(device), yATM_train.to(device)
        all_output = Model(tau_train, logm_train, yATM_train)
        #all_output_c6 = Model(tau_c6_train, logm_c6_train)
        '''
        tau_train.requires_grad=True
        logm_train.requires_grad=True
        tau_c6_train.requires_grad=True
        logm_c6_train.requires_grad=True
        # 计算梯度
        gradients = torch.autograd.grad(total_y, (tau_train, logm_train), retain_graph=True, create_graph=True)
        grad_tau_1 = gradients[0].clone()  # 保存 x1 的一阶梯度
        grad_logm_1 = gradients[1].clone()  # 保存 x2 的一阶梯度
        # 计算梯度
        gradients = torch.autograd.grad(total_y, (tau_train, logm_train), retain_graph=True, create_graph=True)
        grad_logm_2 = gradients[1].clone()  # 保存 x2 的二阶梯度
        total_y.backward(retain_graph=True)
        tau_train.grad.zero_()
        logm_train.grad.zero_()

        total_y_c6 = torch.sum(all_output_c6)
        total_y_c6.backward(retain_graph=True)
        logm_c6_train.grad.zero_()
        grad_logm_c6_2nd = logm_c6_train.grad.clone()  # 2nd partial derivative of y to logm
        logm_c6_train.grad.zero_()'''
        loss = loss_function(all_output, y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #total_loss, losses = loss_function(all_output, y_train, logm_train, grad_tau_1, grad_logm_1, grad_logm_2, all_output_c6, logm_c6_train, grad_logm_c6_2nd)
        #print(total_loss, losses)
        loss = loss.cpu().item()
        running_loss.append(loss)
        printEpoch(epoch, loss)
    
    print('Evaluating result...')
    Model.eval()
    test_loss = 0
    plot_path = config['save_path']['plot_path']
    model_path = config['save_path']['model_path']
    best_loss = config['model_sett'].getfloat('best_loss')
    if not os.path.exists(plot_path):
        os.makedirs(plot_path)

    with torch.no_grad():
        Model.eval()
        tau_test, logm_test, y_test, yATM_test = dp.Prepare_test_data(test_start_date, test_end_date)
        tau_test, logm_test, y_test, yATM_test = tau_test.to(device), logm_test.to(device), y_test.to(device), yATM_test.to(device)
        output = Model(tau_test, logm_test, yATM_test)
        test_loss = loss_function(output, y_test)
        test_loss = test_loss.cpu().item()

        if test_loss < best_loss:
            config['model_sett']['best_loss'] = str(test_loss)
            with open('config.ini', 'w') as configfile:
                config.write(configfile)

            torch.save(Model.state_dict(), model_path)

    print(f'Test Loss: {test_loss}')
    
    train_start_date = train_start_date.strftime("%Y%m%d")
    train_end_date = train_end_date.strftime("%Y%m%d")
    test_start_date = test_start_date.strftime("%Y%m%d")
    test_end_date = test_end_date.strftime("%Y%m%d")
    plt.plot(running_loss, label='Training Loss')
    plt.xlabel('running')
    plt.ylabel('Loss')
    plt.scatter(args.epochs, test_loss, color='red', label='Test Loss')
    plt.legend()
    plt.savefig(f'{plot_path}/train{train_start_date}to{train_end_date}_test{test_start_date}to{test_end_date}_{epoch}epoch.png')
    print('Finished Training')

if __name__ == '__main__':
    main()