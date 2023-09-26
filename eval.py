import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score,roc_auc_score
import numpy as np

def sensitivity_specificity(Y_test, Y_pred):
    con_mat = confusion_matrix(Y_test, Y_pred)
    tp = con_mat[1][1]
    fp = con_mat[0][1]
    fn = con_mat[1][0]
    tn = con_mat[0][0]
    # print("tn:", tn, "tp:", tp, "fn:", fn, "fp:", fp)
    if tn == 0 and fp == 0:
        specificity = 0
    else:
        specificity = tn / (fp + tn)

    if tp == 0 and fn == 0:
        sensitivity = 0
    else:
        sensitivity = tp / (tp + fn)

    return sensitivity, specificity




def eval_FCSC(args, model, loader):
    model.eval()
    Y_test = []
    Y_pred = []
    correct = 0.
    test_loss = 0.
    for i, data in enumerate(loader):
        # padded_x_sc, padded_x_fc, sc_pos_enc, fc_pos_enc, sc_lap_pos_enc, fc_lap_pos_enc, adjs_sc, adjs_fc, labels = padded_x_sc.cuda(), padded_x_fc.cuda(), sc_pos_enc.cuda(), fc_pos_enc.cuda(), sc_lap_pos_enc.cuda(), fc_lap_pos_enc.cuda(), adjs_sc.cuda(), adjs_fc.cuda(), labels.cuda()
        data = data.cuda()
        labels = data.y.cuda()
        
        output, sc_x, fc_x, bottleneck = model(data)
        
        pred = output.data.argmax(dim=1)
        correct += torch.sum(pred == labels.view(-1)).item()
        test_loss += F.cross_entropy(output, labels.view(-1)).item() * output.shape[0]

        pred_num = pred.cpu().numpy()
        y_num = labels.cpu().numpy()
        for num in range(len(pred)):
            Y_pred.append(pred_num[num])
            Y_test.append(y_num[num])

    test_acc = correct / len(loader.dataset)
    test_loss = test_loss / len(loader.dataset)
    test_sen, test_spe = sensitivity_specificity(Y_test, Y_pred)
    test_f1=f1_score(Y_test, Y_pred)
    test_auc=roc_auc_score(Y_test, Y_pred)
    

    return test_acc, test_loss, test_sen, test_spe, test_f1, test_auc, Y_test, Y_pred

def eval_FCSC_for_tsne(args, model, loader, epoch):
    model.eval()
    Y_test = []
    Y_pred = []
    correct = 0.
    test_loss = 0.
    for i, data in enumerate(loader):
        # padded_x_sc, padded_x_fc, sc_pos_enc, fc_pos_enc, sc_lap_pos_enc, fc_lap_pos_enc, adjs_sc, adjs_fc, labels = padded_x_sc.cuda(), padded_x_fc.cuda(), sc_pos_enc.cuda(), fc_pos_enc.cuda(), sc_lap_pos_enc.cuda(), fc_lap_pos_enc.cuda(), adjs_sc.cuda(), adjs_fc.cuda(), labels.cuda()
        data = data.cuda()
        labels = data.y.cuda()
        
        output, sc_x, fc_x, bottleneck = model(data)
        
        tsne = {
            'y': labels.cpu().numpy(),
            'embedding': bottleneck
        }
        
        np.save('tsne/ZDXX/file_{}.npy'.format(epoch), tsne)
        