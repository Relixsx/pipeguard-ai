"""Reference CNN-LSTM architecture plus a regularized dynamic predictor.

ARX is an engineering approximation. This implementation does not claim the
consistency or identifiability theorems in the Dankers dynamic-network papers.
"""
import numpy as np
import torch
from torch import nn

class CNNLSTMAutoencoder(nn.Module):
    def __init__(self, input_dim=8, hidden_dim=64, latent_dim=16, seq_len=30, num_layers=2):
        super().__init__()
        self.seq_len = seq_len
        self.cnn = nn.Sequential(nn.Conv1d(input_dim,32,3,padding=1),nn.ReLU(),
                                 nn.Conv1d(32,64,3,padding=1),nn.ReLU(),nn.Dropout(.1))
        self.enc = nn.LSTM(64,hidden_dim,num_layers,batch_first=True,dropout=.2)
        self.latent = nn.Linear(hidden_dim,latent_dim)
        self.expand = nn.Linear(latent_dim,hidden_dim)
        self.dec = nn.LSTM(hidden_dim,hidden_dim,num_layers,batch_first=True,dropout=.2)
        self.out = nn.Linear(hidden_dim,input_dim)
    def forward(self,x):
        _,(h,_) = self.enc(self.cnn(x.transpose(1,2)).transpose(1,2))
        expanded = self.expand(self.latent(h[-1])).unsqueeze(1).repeat(1,self.seq_len,1)
        decoded,_ = self.dec(expanded)
        return self.out(decoded)

class Standardizer:
    def fit(self,arrays):
        x = np.concatenate(arrays)
        if not np.isfinite(x).all():
            raise ValueError('Training data must be finite')
        self.mean = x.mean(axis=0)
        self.scale = x.std(axis=0)
        self.scale[self.scale < 1e-10] = 1.0
        return self
    def transform(self,x):
        return (x-self.mean)/self.scale

def arx_design(z, lag=3, n_outputs=6):
    # Forecast y[t] from y/u history only. No current or future labels used.
    blocks = [z[lag-k:len(z)-k] for k in range(1,lag+1)]
    return np.column_stack(blocks+[np.ones(len(z)-lag)]), z[lag:,:n_outputs]

class DynamicPredictor:
    def __init__(self,lag=3,alpha=1.0):
        self.lag,self.alpha = lag,alpha
    def fit(self,z_runs):
        parts = [arx_design(z,self.lag) for z in z_runs]
        X,Y = np.concatenate([p[0] for p in parts]),np.concatenate([p[1] for p in parts])
        regularizer = np.eye(X.shape[1])*self.alpha
        regularizer[-1,-1] = 0
        self.coef = np.linalg.solve(X.T@X+regularizer,X.T@Y)
        residual = Y-X@self.coef
        self.residual_mean = residual.mean(axis=0)
        cov = np.cov(residual,rowvar=False)
        eigenvalues,eigenvectors = np.linalg.eigh(cov)
        eigenvalues = np.maximum(eigenvalues, max(eigenvalues.max()*1e-5,1e-10))
        self.whitener = eigenvectors@np.diag(1/np.sqrt(eigenvalues))@eigenvectors.T
        self.design_singular_values = np.linalg.svd(X,compute_uv=False)
        return self
    def residuals(self,z):
        X,Y = arx_design(z,self.lag)
        return (Y-X@self.coef-self.residual_mean)@self.whitener

def windows(array, seq_len=30, stride=1):
    if len(array) < seq_len:
        return np.empty((0,seq_len,array.shape[1]),dtype=np.float32)
    return np.stack([array[start:start+seq_len] for start in range(0,len(array)-seq_len+1,stride)]).astype(np.float32)

def train_autoencoder(train,validation,input_dim,seed,epochs=16,batch_size=128):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = CNNLSTMAutoencoder(input_dim)
    opt = torch.optim.Adam(model.parameters(),lr=1e-3)
    best,best_state,patience = np.inf,None,0
    history = []
    validation = torch.from_numpy(validation)
    for epoch in range(epochs):
        model.train()
        sums,count = 0.,0
        for idx in np.array_split(rng.permutation(len(train)),max(1,int(np.ceil(len(train)/batch_size)))):
            x = torch.from_numpy(train[idx])
            opt.zero_grad()
            loss = ((model(x)-x)**2).mean()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),1.0)
            opt.step()
            sums += float(loss.detach())*len(idx)
            count += len(idx)
        model.eval()
        with torch.no_grad():
            values = [float(((model(x)-x)**2).mean())*len(x) for x in validation.split(batch_size)]
        val = sum(values)/len(validation)
        history.append({'epoch':epoch+1,'train_mse':sums/count,'validation_mse':val})
        if val < best-1e-5:
            best,best_state,patience = val,{k:v.detach().clone() for k,v in model.state_dict().items()},0
        else:
            patience += 1
        if patience >= 4:
            break
    model.load_state_dict(best_state)
    model.eval()
    return model,history

def autoencoder_scores(model,array,batch_size=256):
    result = []
    with torch.no_grad():
        for start in range(0,len(array),batch_size):
            x = torch.from_numpy(array[start:start+batch_size])
            result.append(((model(x)-x)**2).mean(dim=(1,2)).numpy())
    return np.concatenate(result) if result else np.array([])
