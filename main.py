from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from data import Numbers
from model import Pct
import numpy as np
from torch.utils.data import DataLoader
from util import cal_loss, IOStream
import sklearn.metrics as metrics
from torchvision import transforms
from tqdm import tqdm
import random
import pickle
from collections import defaultdict
import time 
from sklearn.metrics import roc_auc_score

import yaml

from datetime import datetime

BASE_DATA_DIR = './data'


def _init_():
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    if not os.path.exists('checkpoints/'+args.exp_name):
        os.makedirs('checkpoints/'+args.exp_name)
    if not os.path.exists('checkpoints/'+args.exp_name+'/'+'models'):
        os.makedirs('checkpoints/'+args.exp_name+'/'+'models')
    os.system('cp main.py checkpoints'+'/'+args.exp_name+'/'+'main.py.backup')
    os.system('cp model.py checkpoints' + '/' + args.exp_name + '/' + 'model.py.backup')
    os.system('cp util.py checkpoints' + '/' + args.exp_name + '/' + 'util.py.backup')
    os.system('cp data.py checkpoints' + '/' + args.exp_name + '/' + 'data.py.backup')


def seed_everything(seed=42, benchmark=True):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)#as reproducibility docs
    torch.manual_seed(seed)# as reproducibility docs
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = benchmark # as reproducibility docs when False
    torch.backends.cudnn.deterministic = True# as reproducibility docs

def train(args, io):
    
    with open('./config/train_config.yaml') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    
    data_config = config['data']
    TRAIN_MAKE_SELECT = data_config['train_make_select']
    VAL_MAKE_SELECT = data_config['val_make_select']
    TRAIN_NORMAL_PROBABILITY = data_config['train_normal_probability']
    VAL_NORMAL_PROBABILITY = data_config['val_normal_probability']
    NUM_POINTS = data_config['num_points']
    NUM_WORKERS = data_config['num_workers']
    SAMPLING_METHOD = data_config['sampling_method']
    MODEL_SAMPLING = data_config['model_sampling']
    
    # voxelize
    VOXEL_DOWNSAMPLING = data_config['voxelize']['voxel_downsampling']
    VOXEL_SIZE = data_config['voxelize']['voxel_size']
    OUTLIER_REMOVER = data_config['outlier']['outlier_remover']
    NB_POINTS = data_config['outlier']['nb_points']
    RADIUS = data_config['outlier']['radius']
    WEIGHT_METHOD = data_config['weight_method']
    
    model_config = config['model']
    CV = model_config['cv']
    CV_NUM = model_config['cv_num']
    LOAD_SERIAL_NUMBER = model_config['load_serial_number']
    DROPOUT = model_config['dropout']
    
    params = {}
    params['dropout'] = DROPOUT
    
    train_config = config['train']
    OPTIMIZER = train_config['optimizer']
    SCHEDULER = train_config['scheduler']
    EARLY_STOPPING = train_config['early_stopping']
    LR = train_config['lr']
    EPOCH = train_config['epoch']
    BATCH_SIZE = train_config['batch_size']
    WEIGHT_DECAY = train_config['weight_decay']
    BEST_METRIC = train_config['best_metric']
    MOMENTUM = train_config['momentum']
    ETA_MIN = train_config['eta_min']
    
    PATIENCE = train_config['patience']
    THRESHOLD = train_config['threshold']
    FACTOR = train_config['factor']
    MIN_LR = train_config['min_lr']
    EPS = train_config['eps']
    
    
    other_config = config['other']
    RANDOM_SEED = other_config['random_seed']
    BENCH_MARK = other_config['bench_mark']
    
    
    
    label_df = pd.read_csv(BASE_DATA_DIR+'/train.csv')
    labels = np.unique(label_df['label'])
    label_num = len(labels)
    
    if type(LOAD_SERIAL_NUMBER)!=bool or LOAD_SERIAL_NUMBER==None:
        save_path = f'./result/train/{LOAD_SERIAL_NUMBER}'
        
    else:
        now_time = datetime.now().strftime('%Y%m%d%H%M%S')
        save_path = f'./result/train/{now_time}'
        os.makedirs(save_path)
    
    save_path = os.path.join(save_path, str(CV_NUM))
    os.makedirs(save_path)
    
    seed_everything(RANDOM_SEED, BENCH_MARK)
    
    skf = StratifiedKFold(n_splits=CV, shuffle=True, random_state=RANDOM_SEED)
    
    #transform = transforms.Compose([transforms.ToTensor()])
    
    train_index = []
    test_index = []
    idx = 0
    for t_train_index, t_test_index in skf.split(label_df.index.tolist(), label_df['label'].tolist()):
        train_index = t_train_index.tolist()
        test_index = t_test_index.tolist()
        if idx==CV_NUM:
            break
        idx += 1

    #sklearn.model_selection.StratifiedKFold()
    train_dataset = Numbers(partition='train', data_path=BASE_DATA_DIR, num_points=NUM_POINTS, index_list=train_index,
                             make_select=TRAIN_MAKE_SELECT, normal_probability=TRAIN_NORMAL_PROBABILITY,
                             save_path=save_path, sampling_method=SAMPLING_METHOD, 
                            model_sampling=MODEL_SAMPLING, voxel_downsampling=VOXEL_DOWNSAMPLING, 
                            voxel_size=VOXEL_SIZE, outlier_remover=OUTLIER_REMOVER, nb_points=NB_POINTS, 
                            radius=RADIUS, weight_method=WEIGHT_METHOD)
    test_dataset = Numbers(partition='valid', data_path=BASE_DATA_DIR, num_points=NUM_POINTS, index_list=test_index,
                            make_select=VAL_MAKE_SELECT, normal_probability=VAL_NORMAL_PROBABILITY,
                            save_path=save_path, sampling_method=SAMPLING_METHOD, 
                            model_sampling=MODEL_SAMPLING, voxel_downsampling=VOXEL_DOWNSAMPLING, 
                            voxel_size=VOXEL_SIZE, outlier_remover=OUTLIER_REMOVER, nb_points=NB_POINTS, 
                            radius=RADIUS, weight_method=WEIGHT_METHOD)


    train_loader = DataLoader(train_dataset, num_workers=NUM_WORKERS,
                            batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, num_workers=NUM_WORKERS,
                            batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

    cuda_available = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if cuda_available else "cpu")

    model = Pct(params, output_channels=label_num).to(device)
    
    #model.conv1 = nn.Conv1d(1, 64, kernel_size=1, bias=False)
    print(str(model))

    model = nn.DataParallel(model)
    
    opt = None
    if OPTIMIZER=='sgd':
        print("Use SGD")
        opt = optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    elif OPTIMIZER=='adam':
        print("Use Adam")
        opt = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    
    scheduler = None
    if SCHEDULER=='cosine_annealing_lr':
        scheduler = CosineAnnealingLR(opt, EPOCH, eta_min=ETA_MIN)
    elif SCHEDULER=='reduce_lr_onplateau':
        scheduler = ReduceLROnPlateau(opt, patience=PATIENCE, threshold=THRESHOLD,
                                      factor=FACTOR, min_lr=MIN_LR, eps=EPS)
    else:
        raise ValueError('Check scheduler name')

    criterion = cal_loss
    best_test_acc = 0
    
    early_count = 0
    results = defaultdict(list)

    for epoch in tqdm(range(EPOCH)):
        if SCHEDULER!='reduce_lr_onplateau':
            scheduler.step()
        train_loss = 0.0
        count = 0.0
        model.train()
        train_pred = []
        train_true = []
        idx = 0
        total_time = 0.0
        for i, (data, label) in tqdm(enumerate(train_loader)):
            
            data, label = data.to(device), label.to(device).squeeze() 
            data = data.permute(0, 2, 1)

            batch_size = data.size()[0]
            opt.zero_grad()

            start_time = time.time()
            logits = model(data)
            loss = criterion(logits, label)
            loss.backward()
            opt.step()
            end_time = time.time()
            total_time += (end_time - start_time)
            
            preds = logits.max(dim=1)[1]
            count += batch_size
            train_loss += loss.item() * batch_size
            train_true.append(label.cpu().numpy())
            train_pred.append(preds.detach().cpu().numpy())
            idx += 1

    
        print ('train total time is',total_time)
        train_true = np.concatenate(train_true)
        train_pred = np.concatenate(train_pred)
        
        train_loss = train_loss*1.0/count
        train_acc = metrics.accuracy_score(train_true, train_pred)
        train_avg_per_class_acc = metrics.balanced_accuracy_score(train_true, train_pred)
        
        train_true_multi = np.eye(label_num)[train_true]
        train_pred_multi = np.eye(label_num)[train_pred]
        train_roc_auc =roc_auc_score(train_true_multi, train_pred_multi, multi_class='ovr')
        
        
        outstr = 'Train %d, loss: %.6f, train roc_auc: %.6f, train acc: %.6f, train avg acc: %.6f' % (epoch,
                                                                                train_loss,
                                                                                train_roc_auc,
                                                                                train_acc,
                                                                                train_avg_per_class_acc)
        io.cprint(outstr)

        
        results['epoch'].append(epoch+1)
        results['train_acc'].append(train_acc)
        results['train_avg_per_class_acc'].append(train_avg_per_class_acc)
        results['train_roc_auc'].append(train_roc_auc)
        ####################
        # Test
        ####################
        test_loss = 0.0
        count = 0.0
        model.eval()
        test_pred = []
        test_true = []
        total_time = 0.0
        idx = 0
        with torch.no_grad():
            for data, label in test_loader:
                data, label = data.to(device), label.to(device).squeeze()
                
                data = data.permute(0, 2, 1)
                batch_size = data.size()[0]
                start_time = time.time()
                logits = model(data)
                end_time = time.time()
                total_time += (end_time - start_time)
                loss = criterion(logits, label)
                preds = logits.max(dim=1)[1]
                count += batch_size
                test_loss += loss.item() * batch_size
                test_true.append(label.cpu().numpy())
                test_pred.append(preds.detach().cpu().numpy())
                idx +=1 
        
        if SCHEDULER=='reduce_lr_onplateau':
            scheduler.step(test_loss)
        
        print ('test total time is', total_time)
        test_true = np.concatenate(test_true)
        test_pred = np.concatenate(test_pred)
        
        test_true_multi = np.eye(label_num)[test_true]
        test_pred_multi = np.eye(label_num)[test_pred]

        test_roc_auc = roc_auc_score(test_true_multi, test_pred_multi, multi_class='ovr')
        
        test_loss = test_loss*1.0/count
        test_acc = metrics.accuracy_score(test_true, test_pred)
        avg_per_class_acc = metrics.balanced_accuracy_score(test_true, test_pred)
        outstr = 'Test %d, loss: %.6f, test roc_auc: %.6f, test acc: %.6f, test avg acc: %.6f' % (epoch,
                                                                            test_loss,
                                                                            test_roc_auc,
                                                                            test_acc,
                                                                            avg_per_class_acc)
        
        results['test_loss'].append(test_loss)
        results['test_acc'].append(test_acc)
        results['test_avg_per_class_acc'].append(avg_per_class_acc)
        results['test_roc_auc'].append(test_roc_auc)

        io.cprint(outstr)
        if test_acc >= best_test_acc:
            best_test_acc = test_acc
            torch.save(model.state_dict(), f'{save_path}/model.t7')
            print('save model')
            early_stopping_count = 0
        else:
            early_stopping_count += 1
            

        
        if early_stopping_count == EARLY_STOPPING:
            break
        
        with open(f'{save_path}/results.pkl', 'wb') as f:
            pickle.dump(results, f)
        

        #train_loader.dataset.val_auc = test_acc
        print()

    with open(f'{save_path}/train_config.yaml', 'w') as f:
        yaml.dump(config, f)
    


def predict(args, io):
    
    with open('./config/predict_config.yaml', 'r') as f:
        predict_config = yaml.load(f, Loader=yaml.FullLoader)
    
    path_config = predict_config['path']
    TRAIN_SERIAL = path_config['train_serial']
    PRED_CV = path_config['pred_cv']
    if PRED_CV is not False:
        TRAIN_SERIAL = f'{TRAIN_SERIAL}/{PRED_CV}'
    
    test_data_config = predict_config['data']
    TEST_MAKE_SELECT = test_data_config['test_make_select']
    NUM_WORKERS = test_data_config['num_workers']
    
    BATCH_SIZE = predict_config['predict']['batch_size']
    RANDOM_SEED = predict_config['other']['random_seed']
    TRAIN_VAL_PREDICT = predict_config['other']['train_val_predict']
    
    MODEL_PATH = f'./result/train/{TRAIN_SERIAL}/model.t7'
    with open(f'./result/train/{TRAIN_SERIAL}/train_config.yaml', 'r') as f:
        train_set_config = yaml.load(f, Loader=yaml.FullLoader)
    
    data_config = train_set_config['data']
    NUM_POINTS = data_config['num_points']
    SAMPLING_METHOD = data_config['sampling_method']
    MODEL_SAMPLING = data_config['model_sampling']
    
    # voxelize
    VOXEL_DOWNSAMPLING = data_config['voxelize']['voxel_downsampling']
    VOXEL_SIZE = data_config['voxelize']['voxel_size']
    OUTLIER_REMOVER = data_config['outlier']['outlier_remover']
    NB_POINTS = data_config['outlier']['nb_points']
    RADIUS = data_config['outlier']['radius']
    WEIGHT_METHOD = data_config['weight_method']
    
    model_config = train_set_config['model']
    CV = model_config['cv']
    DROPOUT = model_config['dropout']
    
    params = {}
    params['dropout'] = DROPOUT
    
    train_config = train_set_config['train']
    OPTIMIZER = train_config['optimizer']
    SCHEDULER = train_config['scheduler']
    EARLY_STOPPING = train_config['early_stopping']
    LR = train_config['lr']
    EPOCH = train_config['epoch']
    WEIGHT_DECAY = train_config['weight_decay']
    BEST_METRIC = train_config['best_metric']
    MOMENTUM = train_config['momentum']
    ETA_MIN = train_config['eta_min']

    
    other_config = train_set_config['other']

    label_df = pd.read_csv(BASE_DATA_DIR+'/train.csv')
    labels = np.unique(label_df['label'])
    label_num = len(labels)

    now_time = datetime.now().strftime('%Y%m%d%H%M%S')
    save_path = f'./result/predict/{now_time}'
    os.makedirs(save_path)
    
    seed_everything(RANDOM_SEED, False)

    
    #sklearn.model_selection.StratifiedKFold()
    
    test_dataset = Numbers(partition='test', data_path=BASE_DATA_DIR, 
                            num_points=NUM_POINTS, index_list=None, make_select=TEST_MAKE_SELECT,
                            save_path=save_path, label_save=False, sampling_method=SAMPLING_METHOD, 
                            model_sampling=MODEL_SAMPLING, voxel_downsampling=VOXEL_DOWNSAMPLING, 
                            voxel_size=VOXEL_SIZE, outlier_remover=OUTLIER_REMOVER, nb_points=NB_POINTS, 
                            radius=RADIUS, weight_method=WEIGHT_METHOD)
    
    test_loader = DataLoader(test_dataset, num_workers=NUM_WORKERS,
                            batch_size=BATCH_SIZE, shuffle=False, drop_last=False)
    
    cuda_available = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if cuda_available else "cpu")

    model = Pct(params, output_channels=label_num).to(device)
    model = nn.DataParallel(model) 
    model.load_state_dict(torch.load(MODEL_PATH))
    #model.conv1 = nn.Conv1d(1, 64, kernel_size=1, bias=False)

    print(str(model))

    
    opt = None
    if OPTIMIZER=='sgd':
        print("Use SGD")
        opt = optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    elif OPTIMIZER=='adam':
        print("Use Adam")
        opt = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    
    
    criterion = cal_loss
    best_test_acc = 0

    model.eval()
    
    submit = pd.read_csv(f'{BASE_DATA_DIR}/sample_submission.csv')
    submit = submit.drop('label', axis=1)
    temp_df = pd.DataFrame({'ID':[int(i) for i in test_dataset.ids]})
    pred_list = []
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            data = data.permute(0, 2, 1)
    
            logits = model(data)
            preds = logits.max(dim=1)[1]
            pred_list += preds.cpu().tolist()
        
    temp_df['label'] = pred_list
    submit = pd.merge(submit, temp_df, on='ID', how='left')
    #submit = submit.dropna()
    submit['label'] = submit['label'].astype(int)
    submit.to_csv(f'{save_path}/submit.csv', index=False)
    
    # make train, val
    if TRAIN_VAL_PREDICT:
        
        val_dataset = Numbers(partition='valid', data_path=BASE_DATA_DIR, 
                            num_points=NUM_POINTS, index_list=None, make_select=False,
                            train_serial=TRAIN_SERIAL, sampling_method=SAMPLING_METHOD, 
                            model_sampling=MODEL_SAMPLING, voxel_downsampling=VOXEL_DOWNSAMPLING, 
                            voxel_size=VOXEL_SIZE, outlier_remover=OUTLIER_REMOVER, nb_points=NB_POINTS, 
                            radius=RADIUS, weight_method=WEIGHT_METHOD)
        
        val_loader = DataLoader(val_dataset, num_workers=NUM_WORKERS,
                                batch_size=BATCH_SIZE, shuffle=False, drop_last=False)
        
        model.eval()
        
        
        val_true = []
        val_pred = []
        with torch.no_grad():
            for data, label in val_loader:
                data, label = data.to(device), label.to(device).squeeze()
                
                data = data.permute(0, 2, 1)
                batch_size = data.size()[0]
                logits = model(data)                
                preds = logits.max(dim=1)[1]
                
                #test_loss += loss.item() * batch_size
                val_true.append(label.cpu().numpy())
                val_pred.append(preds.detach().cpu().numpy())
        

        val_true = np.concatenate(val_true)
        val_pred = np.concatenate(val_pred)
        
        val_result = {'true':val_true, 'pred':val_pred}
        with open(f'{save_path}/val_result.pkl', 'wb') as f:
            pickle.dump(val_result, f)
    
    with open(f'{save_path}/pred_config.yaml', 'w') as f:
        yaml.dump(predict_config, f)


def cv_predict(args, io):
    
    with open('./config/predict_config.yaml', 'r') as f:
        predict_config = yaml.load(f, Loader=yaml.FullLoader)
    
    path_config = predict_config['path']
    TRAIN_SERIAL = path_config['train_serial']
    
    
    test_data_config = predict_config['data']
    TEST_MAKE_SELECT = test_data_config['test_make_select']
    NUM_WORKERS = test_data_config['num_workers']
    
    BATCH_SIZE = predict_config['predict']['batch_size']
    RANDOM_SEED = predict_config['other']['random_seed']
    TRAIN_VAL_PREDICT = predict_config['other']['train_val_predict']
    
    
    with open(f'./result/train/{TRAIN_SERIAL}/0/train_config.yaml', 'r') as f:
        train_set_config = yaml.load(f, Loader=yaml.FullLoader)
    
    data_config = train_set_config['data']
    NUM_POINTS = data_config['num_points']
    SAMPLING_METHOD = data_config['sampling_method']
    MODEL_SAMPLING = data_config['model_sampling']
    
    # voxelize
    VOXEL_DOWNSAMPLING = data_config['voxelize']['voxel_downsampling']
    VOXEL_SIZE = data_config['voxelize']['voxel_size']
    OUTLIER_REMOVER = data_config['outlier']['outlier_remover']
    NB_POINTS = data_config['outlier']['nb_points']
    RADIUS = data_config['outlier']['radius']
    WEIGHT_METHOD = data_config['weight_method']
    
    model_config = train_set_config['model']
    CV = model_config['cv']
    DROPOUT = model_config['dropout']
    
    params = {}
    params['dropout'] = DROPOUT
    
    train_config = train_set_config['train']
    OPTIMIZER = train_config['optimizer']
    SCHEDULER = train_config['scheduler']
    EARLY_STOPPING = train_config['early_stopping']
    LR = train_config['lr']
    EPOCH = train_config['epoch']
    WEIGHT_DECAY = train_config['weight_decay']
    BEST_METRIC = train_config['best_metric']
    MOMENTUM = train_config['momentum']
    ETA_MIN = train_config['eta_min']

    
    other_config = train_set_config['other']

    label_df = pd.read_csv(BASE_DATA_DIR+'/train.csv')
    labels = np.unique(label_df['label'])
    label_num = len(labels)

    now_time = datetime.now().strftime('%Y%m%d%H%M%S')
    save_path = f'./result/predict/{now_time}'
    os.makedirs(save_path)
    
    seed_everything(RANDOM_SEED, False)

    
    #sklearn.model_selection.StratifiedKFold()
    
    test_dataset = Numbers(partition='test', data_path=BASE_DATA_DIR, 
                            num_points=NUM_POINTS, index_list=None, make_select=TEST_MAKE_SELECT,
                            save_path=save_path, label_save=False, sampling_method=SAMPLING_METHOD, 
                            model_sampling=MODEL_SAMPLING, voxel_downsampling=VOXEL_DOWNSAMPLING, 
                            voxel_size=VOXEL_SIZE, outlier_remover=OUTLIER_REMOVER, nb_points=NB_POINTS, 
                            radius=RADIUS, weight_method=WEIGHT_METHOD)
    
    test_loader = DataLoader(test_dataset, num_workers=NUM_WORKERS,
                            batch_size=BATCH_SIZE, shuffle=False, drop_last=False)
    
    cuda_available = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if cuda_available else "cpu")
    
    for cv in range(CV):
        model = Pct(params, output_channels=label_num).to(device)
        model = nn.DataParallel(model) 
        
        MODEL_PATH = f'./result/train/{TRAIN_SERIAL}/{cv}/model.t7'
        model.load_state_dict(torch.load(MODEL_PATH))
        #model.conv1 = nn.Conv1d(1, 64, kernel_size=1, bias=False)
    
        print(str(model))
    
        
        opt = None
        if OPTIMIZER=='sgd':
            print("Use SGD")
            opt = optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
        elif OPTIMIZER=='adam':
            print("Use Adam")
            opt = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        
        
        criterion = cal_loss
        best_test_acc = 0
    
        model.eval()
        
        submit = pd.read_csv(f'{BASE_DATA_DIR}/sample_submission.csv')
        submit = submit.drop('label', axis=1)
        temp_df = pd.DataFrame({'ID':[int(i) for i in test_dataset.ids]})
        pred_list = []
        with torch.no_grad():
            for data in test_loader:
                data = data.to(device)
                data = data.permute(0, 2, 1)
        
                preds = model(data)
                #preds = logits.max(dim=1)[1]
                pred_list += preds.cpu().tolist()

        temp_df[[f'label_{i}' for i in range(10)]] = pred_list
        submit = pd.merge(submit, temp_df, on='ID', how='left')
        #submit = submit.dropna()
        #submit[[f'label_{i}' for i in range(10)]] = submit[[f'label_{i}' for i in range(10)]].astype(int)
        submit.to_csv(f'{save_path}/submit_{cv}.csv', index=False)
        
        # make train, val
       
    
    with open(f'{save_path}/pred_config.yaml', 'w') as f:
        yaml.dump(predict_config, f)


if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='Point Cloud Recognition')
    parser.add_argument('--execute', type=str, default='', metavar='N',
                        help='function')
    parser.add_argument('--no_cuda', type=bool, default=False,
                        help='enables CUDA training')
    parser.add_argument('--exp_name', type=str, default='exp', metavar='N',
                        help='Name of the experiment')
    args = parser.parse_args()
    
    io = IOStream('checkpoints/' + args.exp_name + '/run.log')
    io.cprint(str(args))

    args.cuda = not args.no_cuda and torch.cuda.is_available()

    if args.cuda:
        io.cprint(
            'Using GPU : ' + str(torch.cuda.current_device()) + ' from ' + str(torch.cuda.device_count()) + ' devices')

    else:
        io.cprint('Using CPU')
    
    if args.execute =='predict':
        predict(args, io)
    elif args.execute == 'cv_predict':
        cv_predict(args, io)
    else:

        train(args, io)

