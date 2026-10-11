"""D9 cluster-refinement framework adapted to causal gas telemetry.

Paper equations 8-19: chunk statistics, DBSCAN centers, balanced minimum-cost
assignment for refinement training, augmented initial forecasts, L1 routing.
Ridge engines replace CatBoost; observed commands replace scheduled-outage text.
Every context chunk ends before the predicted reading. No future modes/labels.
"""
from itertools import permutations

import numpy as np


def forecast_design(z, lag=2):
    return np.column_stack([z[lag-k:len(z)-k] for k in range(1,lag+1)]+
                           [np.ones(len(z)-lag)])


def causal_context(z, window=30):
    # For target t, the context is [t-window, t), not a centered window.
    past = np.lib.stride_tricks.sliding_window_view(z[:-1,6:8],window,axis=0)
    return np.stack([past.mean(axis=2),past.std(axis=2),past.max(axis=2),past.min(axis=2)],
                    axis=2).reshape(len(past),8)


def dbscan_centers(features, eps=.8, min_samples=8):
    """Deterministic Euclidean DBSCAN; -1 noise points do not become centers."""
    x = np.asarray(features,float)
    distance = np.sqrt(np.maximum(0,(x*x).sum(axis=1)[:,None]+(x*x).sum(axis=1)[None,:]-2*x@x.T))
    neighbors = [np.flatnonzero(row<=eps).tolist() for row in distance]
    labels = np.full(len(x),-2,int)
    cluster = 0
    for i in range(len(x)):
        if labels[i]!=-2:continue
        if len(neighbors[i])<min_samples:
            labels[i]=-1;continue
        labels[i]=cluster
        queue,queued=list(neighbors[i]),set(neighbors[i])
        for j in queue:
            if labels[j]==-1:labels[j]=cluster
            if labels[j]!=-2:continue
            labels[j]=cluster
            if len(neighbors[j])>=min_samples:
                for k in neighbors[j]:
                    if k not in queued:queued.add(k);queue.append(k)
        cluster+=1
    centers = np.array([x[labels==j].mean(axis=0) for j in range(cluster)])
    return centers,labels


def balanced_assignment(cost):
    """Exact minimum-cost bijection (D9 Eq.12); small k<=6 permits enumeration.

    Same optimization objective as Hungarian, with a bounded exhaustive solver.
    This implementation is not labelled the Hungarian algorithm itself.
    """
    n=cost.shape[0]
    if cost.shape!=(n,n) or not 1<=n<=6:raise ValueError("Assignment supports square costs of order 1-6")
    return np.array(min(permutations(range(n)),key=lambda p:sum(cost[i,p[i]] for i in range(n))))


def ridge(X,Y,alpha):
    penalty=np.eye(X.shape[1])*alpha;penalty[-1,-1]=0
    return np.linalg.solve(X.T@X+penalty,X.T@Y)


class ContextRefinement:
    def __init__(self,mean,scale,base_coef,context_mean,context_scale,centers,refiners,
                 residual_means,whiteners,window=30):
        self.mean,self.scale,self.base_coef=mean,scale,base_coef
        self.context_mean,self.context_scale,self.centers=context_mean,context_scale,centers
        self.refiners,self.residual_means,self.whiteners=refiners,residual_means,whiteners
        self.window=window

    def residuals(self,values):
        z=(np.asarray(values,float)-self.mean)/self.scale
        context=(causal_context(z,self.window)-self.context_mean)/self.context_scale
        X=forecast_design(z)[self.window-2:]
        base=X@self.base_coef
        augmented=np.column_stack([X[:,:-1],2*base,np.ones(len(X))])
        group=np.argmin(np.abs(context[:,None,:]-self.centers[None,:,:]).sum(axis=2),axis=1)
        predictions=np.einsum('ni,nij->nj',augmented,self.refiners[group])
        error=z[self.window:,:6]-predictions-self.residual_means[group]
        residual=np.einsum('ni,nij->nj',error,self.whiteners[group])
        return residual,group

    def score_series(self,values,timestamps,sample_s=5):
        values=np.asarray(values,float);timestamps=np.asarray(timestamps,float)
        result=np.full(len(values),np.nan)
        start=0
        for i in range(len(values)+1):
            missing=i<len(values) and not np.isfinite(values[i]).all()
            gap=0<i<len(values) and abs(timestamps[i]-timestamps[i-1]-sample_s)>.5
            if i==len(values) or missing or gap:
                if i-start>=2*self.window:
                    residual,_=self.residuals(values[start:i])
                    squares=np.mean(residual**2,axis=1)
                    average=np.convolve(squares,np.ones(self.window)/self.window,mode='valid')
                    result[start+2*self.window-1:i]=average
                start=i+1 if missing else i
        return result

    def save(self,path):
        np.savez_compressed(path,mean=self.mean,scale=self.scale,base_coef=self.base_coef,
                            context_mean=self.context_mean,context_scale=self.context_scale,centers=self.centers,
                            refiners=self.refiners,residual_means=self.residual_means,whiteners=self.whiteners,
                            window=np.array(self.window))

    @classmethod
    def load(cls,path):
        with np.load(path,allow_pickle=False) as data:
            return cls(**{k:data[k] for k in ['mean','scale','base_coef','context_mean','context_scale',
                                            'centers','refiners','residual_means','whiteners']},window=int(data['window']))


def fit_context_refinement(initial_runs,refinement_runs,validation_runs):
    """Independent initial/refinement/validation runs, all healthy."""
    healthy=np.concatenate(initial_runs)
    mean,scale=healthy.mean(axis=0),np.maximum(healthy.std(axis=0),1e-10)
    initial=[(x-mean)/scale for x in initial_runs]
    base_coef=ridge(np.concatenate([forecast_design(z) for z in initial]),
                    np.concatenate([z[2:,:6] for z in initial]),1.)
    contexts=np.concatenate([causal_context(z)[::30] for z in initial])
    context_mean,context_scale=contexts.mean(axis=0),np.maximum(contexts.std(axis=0),1e-8)
    scaled=(contexts-context_mean)/context_scale
    trials=[]
    candidates=[]
    for eps in [.35,.5,.7,.9,1.2,1.5,1.8]:
        centers,labels=dbscan_centers(scaled,eps)
        if not 2<=len(centers)<=6:
            trials.append({'eps':eps,'clusters':len(centers),'accepted':False});continue
        groups=[[] for _ in centers]
        for raw in refinement_runs:
            z=(raw-mean)/scale
            ctx=(causal_context(z)-context_mean)/context_scale
            X=forecast_design(z)[28:]
            augmented=np.column_stack([X[:,:-1],2*(X@base_coef),np.ones(len(X))])
            Y=z[30:,:6]
            starts=list(range(0,len(ctx),30))
            k=len(centers)
            for offset in range(0,len(starts),k):
                batch=starts[offset:offset+k]
                cost=np.abs(ctx[batch,None,:]-centers[None,:,:]).sum(axis=2)
                assignment=balanced_assignment(cost) if len(batch)==k else np.argmin(cost,axis=1)
                for start,group in zip(batch,assignment):
                    groups[int(group)].append((augmented[start:start+30],Y[start:start+30]))
        refiners=np.stack([ridge(np.concatenate([p[0] for p in group]),
                                  np.concatenate([p[1] for p in group]),10.) for group in groups])
        means=np.zeros((len(centers),6));white=np.repeat(np.eye(6)[None],len(centers),axis=0)
        candidate=ContextRefinement(mean,scale,base_coef,context_mean,context_scale,centers,refiners,means,white)
        val_errors=[]
        for raw in validation_runs:
            error,_=candidate.residuals(raw);val_errors.append(error)
        mse=float(np.mean(np.concatenate(val_errors)**2))
        trials.append({'eps':eps,'clusters':len(centers),'healthy_validation_mse':mse,'accepted':True,
                       'noise_fraction':float(np.mean(labels==-1))})
        candidates.append((mse,candidate))
    if not candidates:raise ValueError("No 2-6-cluster DBSCAN solution; inspect healthy operating coverage")
    _,model=min(candidates,key=lambda p:p[0])
    # Use live L1 routing to estimate score scales from refinement training only.
    errors,groups=[],[]
    for raw in refinement_runs:
        e,g=model.residuals(raw);errors.append(e);groups.append(g)
    errors,groups=np.concatenate(errors),np.concatenate(groups)
    for j in range(len(model.centers)):
        block=errors[groups==j]
        if len(block)<64:block=errors
        model.residual_means[j]=block.mean(axis=0)
        eig,vec=np.linalg.eigh(np.cov(block,rowvar=False))
        eig=np.maximum(eig,max(eig.max()*1e-4,1e-10))
        model.whiteners[j]=vec@np.diag(1/np.sqrt(eig))@vec.T
    return model,{'trials':trials,'clusters':len(model.centers),
                  'context':'30 prior readings: mean/std/max/min of two known command channels',
                  'paper':'D9 Equations 8-19; DBSCAN, balanced assignment, augmented forecasts and L1 inference',
                  'adaptations':'Ridge forecasting engines replace CatBoost; causal past chunks replace daily forecast context; no text embeddings',
                  'minimum_readings':60,'fitting_labels':'Healthy runs only'}
