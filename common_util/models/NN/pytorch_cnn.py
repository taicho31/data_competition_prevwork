# custom metric

## CNN pytorch version
# kaggle titanic pytorch: https://www.kaggle.com/kaerunantoka/titanic-pytorch-nn-tutorial
# https://github.com/pytorch/pytorch/issues/3867#issuecomment-598264120 (conv1d same padding)
# https://stackoverflow.com/questions/42480111/model-summary-in-pytorch (model summary in pytorch)
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import tensorflow as tf
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.metrics import *
from tensorflow.keras.utils import *
from tensorflow.keras.callbacks import *
from sklearn.utils.class_weight import compute_class_weight

# to  keep reproducibility -----
def seed_everything(seed=1234): 
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    
def sk_pr_auc(y_true, y_pred):
    return tf.py_function(average_precision_score, (y_true, y_pred), tf.float64)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class Net(nn.Module):
    def __init__(self, n_filter, filter_size, drop_rate, gauss_rate):
        super(Net, self).__init__()
        self.gauss_rate = gauss_rate
        self.conv1 = nn.Conv1d(in_channels= 2, out_channels= n_filter, kernel_size = filter_size, padding=filter_size//2) # same padding
        self.batch1 = nn.BatchNorm1d(n_filter)
        self.drop1 = nn.Dropout(drop_rate)
        
        self.conv2 = nn.Conv1d(in_channels=n_filter, out_channels= n_filter, kernel_size = filter_size, padding=filter_size//2)
        self.batch2 = nn.BatchNorm1d(n_filter)
        self.drop2 = nn.Dropout(drop_rate)
        
        self.conv3 = nn.Conv1d(in_channels=n_filter, out_channels= n_filter, kernel_size = filter_size, padding=filter_size//2)
        self.batch3 = nn.BatchNorm1d(n_filter)
        self.drop3 = nn.Dropout(drop_rate)
        
        self.drop5 = nn.Dropout(drop_rate)

        self.lr1 = nn.Linear(65,64)
        self.lr2 = nn.Linear(64,1)

    def forward(self, x, x_aux):
        #https://stackoverflow.com/questions/59090533/how-do-i-add-some-gaussian-noise-to-a-tensor-in-pytorch GaussianNoise(gauss_rate)
        x = x + self.gauss_rate * torch.randn(x.shape[0], 2, 511) 
        x = self.drop1(self.batch1(F.relu(self.conv1(x))))
        x = self.drop2(self.batch2(F.relu(self.conv2(x))))
        x = self.drop3(self.batch3(F.relu(self.conv3(x))))
        
        # https://discuss.pytorch.org/t/global-max-pooling/1345 (x = GlobalMaxPool1D(x))
        # https://www.xn--ebkc7kqd.com/entry/pytorch-pooling
        x, _ = torch.max(x, 2)
        
        # https://discuss.pytorch.org/t/concatenate-layer-output-with-additional-input-data/20462 (x = Concatenate()([x, x_aux]))
        x = torch.cat((x, x_aux), dim=1)
        
        x = self.drop5(F.relu(self.lr1(x)))
        _out = self.lr2(x)
        return _out

def modelling_torch(tr, target, te, exc_wl, exc_wl_test,sample_seed):
    seed_everything(seed=sample_seed) 
    X_train = tr.copy()
    y_train = target.copy()
    X_test = te.copy()

    n_folds=5
    skf=StratifiedKFold(n_splits = n_folds, shuffle=True, random_state=2)
    models = []
    
    X_test = torch.tensor(X_test, dtype=torch.float32)
    exc_wl_test = torch.tensor(exc_wl_test, dtype=torch.float32)
    X_test = torch.utils.data.TensorDataset(X_test, exc_wl_test) 
    test_loader = torch.utils.data.DataLoader(X_test, batch_size=batch_size, shuffle=False)
    
    oof = np.array([])
    real = np.array([])
    pred_value = np.zeros(te.shape[0])
    scores = []
    for i , (train_index, valid_index) in enumerate(skf.split(X_train, y_train)):
        print("Fold "+str(i+1))
        # https://stackoverflow.com/questions/56741087/how-to-fix-runtimeerror-expected-object-of-scalar-type-float-but-got-scalar-typ
        X_train2 = torch.tensor(X_train[train_index,:], dtype=torch.float32)
        y_train2 = torch.tensor(y_train[train_index, np.newaxis], dtype=torch.float32)
        exc_wl_train = torch.tensor(exc_wl[train_index], dtype=torch.float32)        

        X_valid2 = torch.tensor(X_train[valid_index,:], dtype=torch.float32)
        y_valid2 = torch.tensor(y_train[valid_index, np.newaxis], dtype=torch.float32)
        exc_wl_valid = torch.tensor(exc_wl[valid_index], dtype=torch.float32)
            
        clf = Net(n_filter=64, filter_size=7, drop_rate=0.2, gauss_rate = 0.07)
        loss_fn = torch.nn.BCEWithLogitsLoss(reduction="mean", weight=class_weight)
        optimizer = optim.Adam(clf.parameters(), lr=0.001)
        # http://katsura-jp.hatenablog.com/entry/2019/01/30/183501#%E8%87%AA%E4%BD%9Cscheduler
        # https://stackoverflow.com/questions/60050586/pytorch-change-the-learning-rate-based-on-number-of-epochs
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
        
        # https://discuss.pytorch.org/t/make-a-tensordataset-and-dataloader-with-multiple-inputs-parameters/26605
        train = torch.utils.data.TensorDataset(X_train2, exc_wl_train, y_train2) # can take arbitrary number of inputs
        valid = torch.utils.data.TensorDataset(X_valid2, exc_wl_valid, y_valid2) #
        
        #clf.to(device)
        
        train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True) 
        valid_loader = torch.utils.data.DataLoader(valid, batch_size=batch_size, shuffle=False)
       
        best_loss = np.inf 
        for epoch in range(train_epochs):
            start_time = time.time()
            clf.train()
            avg_loss = 0.
            for x_batch, exc_batch, y_batch in tqdm(train_loader, disable=True): #
                #x_batch = x_batch.to(device)
                #y_batch = y_batch.to(device)
                y_pred = clf(x_batch, exc_batch) #
                loss = loss_fn(y_pred, y_batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                avg_loss += loss.item() / len(train_loader)        
            
            clf.eval()
            valid = np.zeros((X_valid2.size(0)))
            target_fold = np.zeros((X_valid2.size(0)))
            avg_val_loss = 0.
            for i, (x_batch, exc_batch, y_batch) in enumerate(valid_loader): #
                #x_batch = x_batch.to(device)
                #y_batch = y_batch.to(device)
                y_pred = clf(x_batch, exc_batch).detach() #
                avg_val_loss += loss_fn(y_pred, y_batch).item() / len(valid_loader)
                valid[i * batch_size:(i+1) * batch_size] = sigmoid(y_pred.cpu().numpy())[:, 0]
                target_fold[i * batch_size:(i+1) * batch_size] = y_batch.cpu().numpy().reshape(-1,)
        
            elapsed_time = time.time() - start_time 
            print('Epoch {}/{} \t loss={:.4f} \t val_loss={:.4f} \t time={:.2f}s'.format(
                epoch + 1, train_epochs, avg_loss, avg_val_loss, elapsed_time))
            scheduler.step()
            #print('Epoch-{0} lr: {1}'.format(epoch, optimizer.param_groups[0]['lr']))
           
            if best_loss > avg_val_loss:
                torch.save(clf.state_dict(), 'best-model-parameters.pt')
                best_loss = avg_val_loss
        
        oof = np.concatenate([oof, valid])
        real = np.concatenate([real, target_fold])
            
        # https://discuss.pytorch.org/t/how-to-save-the-best-model/84608 (save best model)
        pred_model = Net(n_filter= 64, filter_size=7, drop_rate=0.2, gauss_rate = 0.07)
        pred_model.load_state_dict(torch.load('best-model-parameters.pt'))
        
        # test predcition --------------
        test_preds = np.zeros(len(X_test))
        for i, (x_batch, exc_batch,) in enumerate(test_loader): #
            y_pred = pred_model(x_batch, exc_batch).detach() #
            test_preds_fold[i * batch_size:(i+1) * batch_size] = sigmoid(y_pred.cpu().numpy())[:, 0]
        pred_value += test_preds / n_folds
        # ------------------------------
        
        print(average_precision_score(target_fold, valid))
        scores.append(average_precision_score(target_fold, valid))

    score = average_precision_score(real, oof)
    print("average precision score = {}".format(score))
    print("scores in folds:", scores)
    print("mean in folds:", np.mean(scores))
    print("std in folds:", np.std(scores))
    
    return score, pred_value

batch_size = 16
train_epochs = 30

# https://teratail.com/questions/170548: 交互に属性を複数チャネル形式に変換する
wave_500 = (np.array(wave_df) * (np.array(wave_df) > 500))
i = range(wave_df.shape[1]) #追加元の配列
wave_tr = np.insert(wave_500,i,np.array(wave_df)[:,i],axis=1).reshape(-1,2,511)

wave_500_test = (np.array(wave_test) * (np.array(wave_test) > 500))
i = range(wave_test.shape[1]) #追加元の配列
wave_te = np.insert(wave_500_test,i,np.array(wave_test)[:,i],axis=1).reshape(-1,2,511)

nn_y_train = new_train.target.values
nn_X_train = np.array(wave_tr).reshape(-1, 2, 511) # axis=0がデータid, axis=1:ch数, axis=2:データ数
nn_X_test = np.array(wave_te).reshape(-1, 2, 511)

# https://stackoverflow.com/questions/57021620/how-to-calculate-unbalanced-weights-for-bcewithlogitsloss-in-pytorch (class weight)
weights = compute_class_weight('balanced', np.unique(new_train.target.copy()), new_train.target.copy())
class_weight = torch.tensor(weights[1] / weights[0], dtype=torch.float32)

exc_wl = np.array(new_train[["exc_wl"]].copy()) 
exc_wl_test = np.array(new_test[["exc_wl"]].copy()) 

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

pytorch_score, pytorch_pred = modelling_torch(nn_X_train, nn_y_train, nn_X_test, exc_wl, exc_wl_test, sample_seed = 13)
