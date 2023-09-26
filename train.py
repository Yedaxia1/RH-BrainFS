import time
import torch
import torch.nn.functional as F
from bottleneck.loss import SupConLoss
from eval import eval_FCSC, eval_FCSC_for_tsne


def train_model(args, model, optimizer, scheduler, train_loader, val_loader, test_loader, i_fold):
    """
    :param train_loader:
    :param model: model
    :type optimizer: Optimizer
    """
    min_loss = 1e10
    max_acc = 0.0
    patience = 0
    best_epoch = 0
    t0 = time.time()
    for epoch in range(args.epochs):
        model.train()
        t = time.time()
        train_loss = 0.0
        train_correct = 0
        for i, data in enumerate(train_loader):
            optimizer.zero_grad()
            if args.use_cuda:
                data = data.cuda()
            labels = data.y
            
            # padded_x_sc, padded_x_fc, sc_pos_enc, fc_pos_enc, sc_lap_pos_enc, fc_lap_pos_enc, adjs_sc, adjs_fc, labels = padded_x_sc.cuda(), padded_x_fc.cuda(), sc_pos_enc.cuda(), fc_pos_enc.cuda(), sc_lap_pos_enc.cuda(), fc_lap_pos_enc.cuda(), adjs_sc.cuda(), adjs_fc.cuda(), labels.cuda()

            output, sc_x, fc_x, bottleneck = model(data)
            loss_classifier = F.cross_entropy(output, labels.view(-1))
            # loss_single_modal_con = (sup_con_loss(single_modal_cls_sc, labels.view(-1)) + sup_con_loss(single_modal_cls_fc, labels.view(-1))) / 5
            # loss_multi_modal_con = sup_con_loss(torch.cat((cls_sc, cls_fc), dim=1), labels.view(-1)) / 5

            # loss = loss_classifier + args.lamda_1 * loss_single_modal_con + args.lamda_2 * loss_multi_modal_con
            loss = loss_classifier
            loss.backward()
            optimizer.step()

            pred = output.data.argmax(dim=1)
            train_loss += loss.item() * output.shape[0]
            train_correct += torch.sum(pred == labels.view(-1)).item()
        
        scheduler.step()

        val_acc, val_loss, val_sen, val_spe, val_f1, val_auc, _, _ = eval_FCSC(args, model, val_loader)
        test_acc, test_loss, test_sen, test_spe, test_f1, test_auc, _, _ = eval_FCSC(args, model, test_loader)
        
        if epoch % 30 == 0:
            eval_FCSC_for_tsne(args, model, test_loader, epoch)

        n_sample_train = len(train_loader.dataset)

        print('Epoch: {:04d}'.format(epoch), 'train_loss: {:.6f}'.format(train_loss/n_sample_train), 'train_acc: {:.6f}'.format(train_correct/n_sample_train),
              'val_loss: {:.6f}'.format(val_loss), 'val_acc: {:.6f}'.format(val_acc),
              'test_loss: {:.6f}'.format(test_loss), 'test_acc: {:.6f}'.format(test_acc),
              'time: {:.6f}s'.format(time.time() - t))

        if val_acc > max_acc:
            max_acc = val_acc
            eval_FCSC_for_tsne(args, model, test_loader, epoch)
            torch.save(model.state_dict(), 'ckpt/{}/{}_fold_best_model.pth'.format(args.dataset, i_fold))
            print("Model saved at epoch{}".format(epoch))
            best_epoch = epoch
            patience = 0
        else:
            patience += 1

        # if val_loss < min_loss:
        #     torch.save(model.state_dict(), 'ckpt/{}/{}_fold_best_model.pth'.format(args.dataset, i_fold))
        #     print("Model saved at epoch{}".format(epoch))
        #     best_epoch = epoch
        #     min_loss = val_loss
        #     patience = 0
        #     # else:
        #     #    patience += 1
        # else:
        #     patience += 1

        if patience == args.patience:
            break

    print('Optimization Finished! Total time elapsed: {:.6f}'.format(time.time() - t0))

    return best_epoch


