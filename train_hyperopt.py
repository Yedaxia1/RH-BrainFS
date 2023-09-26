import argparse
import atexit
import os
import pickle
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
import numpy as np
import torch
from train import train_model
from eval import eval_FCSC
from load_data import *
from divide import kfold_split, K_Fold, setup_seed, cross_validate
from transform import *
from bottleneck.loss import SupConLoss
from bottleneck.MBT import Alternately_Attention_Bottlenecks, Attention_Bottlenecks
from dataloader import DataLoader
from torch.utils.data import Subset
from torch_geometric.data import DenseDataLoader
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK, STATUS_FAIL
from munch import DefaultMunch
def load_args():
    parser = argparse.ArgumentParser(
        description='MultiModal hyperopt for MBT',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--seed', type=int, default=777, help='random seed')
    parser.add_argument('--dataset_random_seed', type=int, default=1, help='random seed')
    parser.add_argument('--repetitions', type=int, default=10, help='number of repetitions (default: 10)')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.001, help='weight decay')
    parser.add_argument('--threshold', type=float, default=0.2, help='threshold')
    parser.add_argument('--sc_features', type=int, default=90, help='sc_features')
    parser.add_argument('--fc_features', type=int, default=90, help='fc_features')
    parser.add_argument('--num_classes', type=int, default=2, help='the number of classes (HC/MDD)')
    parser.add_argument('--hidden_dim', type=int, default=128, help='hidden size')
    parser.add_argument('--dropout', type=float, default=0.3, help='dropout ratio')
    parser.add_argument('--num_layers', type=int, default=4, help='the numbers of convolution layers')
    parser.add_argument('--fusion_layers', type=int, default=3, help='the numbers of fusion layers')
    parser.add_argument('--num_bottlenecks', type=int, default=8, help='the numbers of bottlenecks')
    parser.add_argument('--epochs', type=int, default=1000, help='maximum number of epochs')
    parser.add_argument('--dim_feedforward', type=int, default=2048, help='maximum number of epochs')
    parser.add_argument('--patience', type=int, default=400, help='patience for early stopping')
    parser.add_argument('--dataset', type=str, default='HCP_MBT', help="XX_SCFC/ZD_SCFC/HCP_SCFC")
    parser.add_argument('--path', type=str, default='/home/yehongting/yehongting_dataset/multi_zhongda_xinxiang/merge_dataset', help='path of dataset')
    parser.add_argument('--result_path', type=str, default='./result/ZDXX.txt', help='path of dataset')
    parser.add_argument('--use_cuda', type=bool, default=True, help='specify cuda devices')
    parser.add_argument('--temperature', type=float, default=0.03, help='dropout ratio')
    parser.add_argument('--negative_weight', type=float, default=0.8, help='dropout ratio')
    parser.add_argument('--num_atom_type', type=int, default=90, help='value for num_atom_type')
    parser.add_argument('--num_edge_type', type=int, default=90, help='value for num_edge_type')
    parser.add_argument('--num_heads', type=int, default=4, help='value for num_heads')
    parser.add_argument('--in_feat_dropout', type=float, default=0.5, help='value for in_feat_dropout')
    parser.add_argument('--readout', type=str, default='mean', help="mean/sum/max")
    parser.add_argument('--layer_norm', type=bool, default=True, help="Please give a value for layer_norm")
    parser.add_argument('--batch_norm', type=bool, default=False, help="Please give a value for batch_norm")
    parser.add_argument('--residual', type=bool, default=True, help="Please give a value for residual")
    # parser.add_argument('--lap_pos_enc', type=bool, default=True, help="Please give a value for lap_pos_enc")
    # parser.add_argument('--wl_pos_enc', type=bool, default=False, help="Please give a value for wl_pos_enc")
    parser.add_argument('--pos_enc', choices=[None, 'diffusion', 'pstep', 'adj'], default='pstep')
    parser.add_argument('--pos_enc_dim', type=int, default=32, help='hidden size')
    parser.add_argument('--normalization', choices=[None, 'sym', 'rw'], default='sym',
                            help='normalization for Laplacian')
    parser.add_argument('--beta', type=float, default=1.0,
                            help='bandwidth for the diffusion kernel')
    parser.add_argument('--p', type=int, default=2, help='p step random walk kernel')
    parser.add_argument('--zero_diag', action='store_true', help='zero diagonal for PE matrix')
    parser.add_argument('--lappe', action='store_true', help='use laplacian PE',default=True)
    parser.add_argument('--lap_dim', type=int, default=32, help='dimension for laplacian PE')
    parser.add_argument('--h', type=int, default=1, help='dimension for laplacian PE')
    parser.add_argument('--max_nodes_per_hop', type=int, default=5, help='dimension for laplacian PE')
    args = parser.parse_args()
    return args

def main(args):
    args = DefaultMunch.fromDict(args)
    acc = []
    loss = []
    sen = []
    spe = []
    f1 = []
    auc = []
    setup_seed(args.seed)
    
    random_s = np.array([125], dtype=int)
    # random_s = np.array([125], dtype=int)
    print(args)
    for random_seed in random_s:
        # myDataset = FSDataset_GT(args)
        transform = HHopSubgraphs(h=args.h, max_nodes_per_hop=args.max_nodes_per_hop, node_label='hop', use_rd=False, subgraph_pretransform=LapEncoding(dim=4))
        
        args.dataset_random_seed = random_seed
        myDataset = MyOwnDataset("ZDXX_{}".format(args.h), pre_transform=transform, args=args)
        # myDataset = MultiModalDataset(args, pre_transform=transform)

        acc_iter = []
        loss_iter = []
        sen_iter = []
        spe_iter = []
        f1_iter = []
        auc_iter = []
        for i, (train_split, valid_split, test_split) in enumerate(zip(*cross_validate(args.repetitions, myDataset))):
                        
            train_subset, valid_subset, test_subset = myDataset[train_split], myDataset[valid_split], myDataset[test_split]
            
            # train_loader = DenseDataLoader(train_subset, batch_size=args.batch_size, shuffle=True)
            # val_loader = DenseDataLoader(valid_subset, batch_size=args.batch_size, shuffle=False)
            # test_loader = DenseDataLoader(test_subset, batch_size=args.batch_size, shuffle=False)
            train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True)
            val_loader = DataLoader(valid_subset, batch_size=args.batch_size, shuffle=False)
            test_loader = DataLoader(test_subset, batch_size=args.batch_size, shuffle=False)

            # Model initialization
            model = Alternately_Attention_Bottlenecks(args)
            if args.use_cuda:
                model = model.cuda()
            # model = ASAP_multi(args).to(args.device)
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            sup_con_loss = SupConLoss()

            # Model training
            best_model = train_model(args, model, optimizer, sup_con_loss, train_loader, val_loader, test_loader, i)

            # Restore model for testing
            model.load_state_dict(torch.load('ckpt/{}/{}_fold_best_model.pth'.format(args.dataset, i)))
            test_acc, test_loss, test_sen, test_spe, test_f1, test_auc,  y, pred = eval_FCSC(args, model, test_loader)
            acc_iter.append(test_acc)
            loss_iter.append(test_loss)
            sen_iter.append(test_sen)
            spe_iter.append(test_spe)
            f1_iter.append(test_f1)
            auc_iter.append(test_auc)
            print('Test set results, best_epoch = {:.1f}  loss = {:.6f}, accuracy = {:.6f}, sensitivity = {:.6f}, '
                  'specificity = {:.6f}, f1_score = {:.6f}, auc_score = {:.6f}'.format(0, test_loss, test_acc, test_sen, test_spe, test_f1, test_auc))
            # with open(args.result_path, 'a+') as f:
            #             f.write("fold:{:04d}  accuracy:{:.6f}     sensitivity:{:.6f}     specificity:{:.6f}     f1_score:{:.6f}     auc_score:{:.6f}\n".format(
            #                 i,test_acc, test_sen,test_spe,test_f1, test_auc))
            # print(y)
            # print(pred)
            
        return {
                "loss": -np.mean(acc_iter),
                'status': STATUS_OK,
                'params': args
            }
        


if __name__ == '__main__':
    def save_result(result_file,trials):
        print("正在保存结果...")
        with open(result_file, "w+") as f:
            for result in trials.results:
                if 'loss' in result and result['loss'] <= trials.best_trial['result']['loss']:
                    print(result, file=f)
        print("结果已保存 {:s}".format(result_file))
        print(trials.best_trial)
    def initial_hyperopt(trial_file,result_file,max_evals):
        try:
            with open(trial_file, "rb") as f:
                trials = pickle.load(f)
            current_process = len(trials.results)
            print("使用已有的trial记录, 现有进度: {:d}/{:d} {:s}".format(current_process,max_evals,trial_file))
        except:
            trials = Trials()
            print("未找到现有进度, 从0开始训练 0/{:d} {:s}".format(max_evals, trial_file))
        atexit.register(save_result,result_file,trials)
        return trials

    max_evals = 200
    args = vars(load_args())
    # args['pos_enc'] = hp.choice('pos_enc',[None,'diffusion','pstep','adj'])
    # args['lap_pe'] = hp.choice('lap_pe',[True, False])
    # args['batch_size'] = hp.choice('batch_size',[64,128,256])
    # args['fusion_layers'] = hp.choice('fusion_layers',[1,2,3,4])
    
    args['lr'] = hp.choice('lr',[0.001, 0.0005, 0.0001])
    args['num_layers'] = hp.choice('num_layers',[1,3,5,7])
    args['num_bottlenecks'] = hp.choice('num_bottlenecks',[2,4,6,8])
    args['hidden_dim'] = hp.choice('hidden_dim',[128,256,512])
    args['h'] = hp.choice('h', [1,2,4])
    args['dropout'] = hp.choice('dropout',[0.1,0.3,0.5,0.7])

    

    save_root = os.path.join("hyperopt")
    result_file = os.path.join(save_root, f"result.log")
    trial_file = os.path.join(save_root, f"result.trial")

    if not os.path.exists(save_root):
        os.makedirs(save_root)

    trials = initial_hyperopt(trial_file,result_file,max_evals)
    best = fmin(
        fn=main,space=args, algo=tpe.suggest, max_evals=max_evals, 
        trials = trials, trials_save_file=trial_file)
