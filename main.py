import argparse
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
import numpy as np
import torch
from train import train_model
from eval import eval_FCSC
from load_data import *
from divide import kfold_split, K_Fold, setup_seed, cross_validate
from transform import *
from bottleneck.loss import SupConLoss
from bottleneck.MBT import Alternately_Attention_Bottlenecks, Attention_Bottlenecks, GNN_Transformer, SubGraphGNN_Transformer
from dataloader import DataLoader
from torch.utils.data import Subset
from torch_geometric.data import DenseDataLoader
from torch.optim.lr_scheduler import StepLR
import json
parser = argparse.ArgumentParser(description='FC and SC Classification')
parser.add_argument('--seed', type=int, default=777, help='random seed')
parser.add_argument('--dataset_random_seed', type=int, default=1, help='random seed')
parser.add_argument('--repetitions', type=int, default=10, help='number of repetitions (default: 10)')
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=0.001, help='weight decay')
parser.add_argument('--threshold', type=float, default=0.12, help='threshold')
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
parser.add_argument('--path', type=str, default='/multi_zhongda_xinxiang/merge_dataset', help='path of dataset')
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

params = json.load(open("./param.json"))
for key, item in params.items():
    args.__setattr__(key, item)


params_datasets = json.load(open("./param_dataset.json"))
for key, item in params_datasets[args.dataset].items():
    args.__setattr__(key, item)

if __name__ == '__main__':
    acc = []
    loss = []
    sen = []
    spe = []
    f1 = []
    auc = []
    setup_seed(args.seed)
    
    random_s = np.array([25, 50, 100, 125, 150, 175, 200, 225, 250, 275], dtype=int)
    # random_s = np.array([125], dtype=int)
    print(args)
    for random_seed in random_s:
        # myDataset = FSDataset_GT(args)
        transform = HHopSubgraphs(h=args.h, max_nodes_per_hop=args.max_nodes_per_hop, node_label='hop', use_rd=False, subgraph_pretransform=LapEncoding(dim=4))
        
        args.dataset_random_seed = random_seed
        myDataset = MyOwnDataset("{}_{}".format(args.dataset, args.h), pre_transform=transform, args=args)
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
            # model = Alternately_Attention_Bottlenecks(args)
            # model = GNN_Transformer(args)
            model = Alternately_Attention_Bottlenecks(args)
            if args.use_cuda:
                model = model.cuda()
            # model = ASAP_multi(args).to(args.device)
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            scheduler = StepLR(optimizer, step_size=50, gamma=0.8)
            # sup_con_loss = SupConLoss()

            # Model training
            best_model = train_model(args, model, optimizer, scheduler, train_loader, val_loader, test_loader, i)

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
            with open(args.result_path, 'a+') as f:
                        f.write("fold:{:04d}  accuracy:{:.6f}     sensitivity:{:.6f}     specificity:{:.6f}     f1_score:{:.6f}     auc_score:{:.6f}\n".format(
                            i,test_acc, test_sen,test_spe,test_f1, test_auc))
            
            break
            # print(y)
            # print(pred)
        break
        acc.append(np.mean(acc_iter))
        sen.append(np.mean(sen_iter))
        spe.append(np.mean(spe_iter))        
        f1.append(np.mean(f1_iter))
        auc.append(np.mean(auc_iter))

        print('Average test set results, accuracy = {:.2f}±{:.2f}, sen = {:.2f}±{:.2f},'
          'spe = {:.2f}±{:.2f}, f1 = {:.2f}±{:.2f}, auc = {:.2f}±{:.2f}'.format(np.mean(acc_iter)*100, np.std(acc_iter)*100, np.mean(sen_iter)*100, np.std(sen_iter)*100,
                                                       np.mean(spe_iter)*100, np.std(spe_iter)*100, np.mean(f1_iter)*100, np.std(f1_iter)*100, np.mean(auc_iter)*100, np.std(auc_iter)*100))
        with open(args.result_path, 'a+') as f:
                        f.write("seed:{:03d} AVERAGE acc:{:.2f}±{:.2f}, sen:{:.2f}±{:.2f}, spe:{:.2f}±{:.2f}, f1:{:.2f}±{:.2f}, auc:{:.2f}±{:.2f}\n".format(
                            random_seed,
                            np.mean(acc_iter)*100, np.std(acc_iter)*100, np.mean(sen_iter)*100, np.std(sen_iter)*100,
                            np.mean(spe_iter)*100, np.std(spe_iter)*100, np.mean(f1_iter)*100, np.std(f1_iter)*100, 
                            np.mean(auc_iter)*100, np.std(auc_iter)*100))
    
    print(args)
    print('Total test set results, accuracy : {}'.format(acc))
    print('Average test set results, mean accuracy = {:.6f}, std = {:.6f}, mean_sen = {:.6f}, std_sen = {:.6f}, '
          'mean_spe = {:.6f}, std_spe = {:.6f}, mean_f1 = {:.6f}, std_f1 = {:.6f}, mean_auc = {:.6f}, std_auc = {:.6f}'.format(np.mean(acc), np.std(acc), np.mean(sen), np.std(sen),
                                                       np.mean(spe), np.std(spe), np.mean(f1), np.std(f1), np.mean(auc), np.std(auc)))
    with open(args.result_path, 'a+') as f:
                        f.write("Final AVERAGE acc:{:.2f}±{:.2f}, sen:{:.2f}±{:.2f}, spe:{:.2f}±{:.2f}, f1:{:.2f}±{:.2f}, auc:{:.2f}±{:.2f}".format(
                            np.mean(acc)*100, np.std(acc)*100, np.mean(sen)*100, np.std(sen)*100,
                            np.mean(spe)*100, np.std(spe)*100, np.mean(f1)*100, np.std(f1)*100, 
                            np.mean(auc)*100, np.std(auc)*100))