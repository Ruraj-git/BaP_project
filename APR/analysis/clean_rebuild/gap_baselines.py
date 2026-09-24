"""Training-only simple comparators for the registered block/dispersed tasks."""
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from scoring_core import registered_blocks, ORIGIN

def harmonics(d):
    angle=2*np.pi*pd.to_datetime(d.datum).dt.dayofyear.to_numpy()/365.25
    return np.column_stack([np.ones(len(d)),np.sin(angle),np.cos(angle),np.sin(2*angle),np.cos(2*angle)])

def ridge_predict(train,test):
    raw=train[['pm25_mean','pm10_mean']].to_numpy(float)
    med=np.array([np.median(c[np.isfinite(c)]) if np.isfinite(c).any() else 0.0 for c in raw.T])
    def design(d):
        x=d[['pm25_mean','pm10_mean']].to_numpy(float)
        x=np.log1p(np.maximum(0,np.where(np.isfinite(x),x,med)))
        h=harmonics(d)
        return np.column_stack([x,h[:,1:3],x[:,0]*h[:,1],x[:,0]*h[:,2]])
    sc=StandardScaler().fit(design(train))
    stations=sorted(train.eoi.unique())
    def matrix(d):
        return np.column_stack([sc.transform(design(d))]+[d.eoi.eq(s).to_numpy(float) for s in stations])
    fit=Ridge(alpha=1).fit(matrix(train),np.log1p(train.bap))
    return np.maximum(0,np.expm1(fit.predict(matrix(test))))

def interpolation(train,test):
    """Strict bracketing; every bracket belongs to the training partition."""
    linear=np.full(len(test),np.nan); loglinear=linear.copy(); spans=linear.copy()
    for i, row in enumerate(test.itertuples()):
        retained=train[train.eoi.eq(row.eoi)].sort_values('datum')
        before=retained[retained.datum<row.datum]; after=retained[retained.datum>row.datum]
        if before.empty or after.empty: continue
        left,right=before.iloc[-1],after.iloc[0]
        fraction=(row.datum-left.datum)/(right.datum-left.datum)
        linear[i]=left.bap+fraction*(right.bap-left.bap)
        loglinear[i]=np.expm1(np.log1p(left.bap)+fraction*(np.log1p(right.bap)-np.log1p(left.bap)))
        spans[i]=(right.datum-left.datum).days
    return linear,loglinear,spans

def run(frame):
    d=frame.copy(); d.datum=pd.to_datetime(d.datum); parts=[]
    for protocol,folds in [('block30',registered_blocks(d.datum)),('dispersed_mod9',(d.datum-ORIGIN).dt.days%9)]:
        for fold in sorted(folds.unique()):
            train,test=d[folds.ne(fold)],d[folds.eq(fold)].copy()
            harmonic=np.full(len(test),np.nan)
            for station in test.eoi.unique():
                mask=test.eoi.eq(station).to_numpy(); tr=train[train.eoi.eq(station)]
                coef=np.linalg.lstsq(harmonics(tr),tr.bap.to_numpy(float),rcond=None)[0] if len(tr)>=2 else np.array([train.bap.mean(),0,0,0,0])
                harmonic[mask]=np.maximum(0,harmonics(test.loc[mask])@coef)
            linear,loglinear,spans=interpolation(train,test)
            for method,pred in [('PM_RIDGE',ridge_predict(train,test)),('HARMONIC',harmonic),('INTERP_LINEAR',linear),('INTERP_LOGLINEAR',loglinear)]:
                out=test[['eoi','datum','bap']].rename(columns={'bap':'observed'}).copy()
                out['prediction']=pred; out['method']=method; out['protocol']=protocol; out['fold']=fold; out['bracket_days']=spans
                parts.append(out)
    return pd.concat(parts,ignore_index=True)
