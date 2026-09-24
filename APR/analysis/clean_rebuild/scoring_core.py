"""Paired uncertainty and fold-wise reconstruction, independent of fit workers."""
import hashlib
import numpy as np
import pandas as pd

ORIGIN = pd.Timestamp('2023-06-02')
SEED, REPLICATES = 20260921, 2000

def registered_blocks(dates):
    return ((pd.to_datetime(dates) - ORIGIN).dt.days // 30).astype(int)

def scores(observed, prediction):
    y, p = np.asarray(observed, float), np.asarray(prediction, float)
    if not len(y) or not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ValueError('empty or nonfinite scoring support')
    error = p-y
    sst = np.sum((y-y.mean())**2)
    return dict(RMSE=float(np.sqrt(np.mean(error**2))), MAE=float(np.mean(abs(error))),
                bias=float(error.mean()), agreement_R2=float(1-np.sum(error**2)/sst) if sst else np.nan)

def paired(a, b, keys, scheme='station', metric='RMSE', reps=REPLICATES):
    """Paired percentile interval on identical support, never silently inner-join."""
    a, b = a.sort_values(keys).reset_index(drop=True), b.sort_values(keys).reset_index(drop=True)
    if a.duplicated(keys).any() or b.duplicated(keys).any() or not a[keys].equals(b[keys]):
        raise ValueError('paired support differs')
    if not np.array_equal(a.observed.to_numpy(),b.observed.to_numpy()):
        raise ValueError('paired targets differ')
    if a.eoi.nunique()<2:
        raise ValueError('do not bootstrap a single station')
    st, st_names = pd.factorize(a.eoi, sort=True)
    if scheme=='station_x_block':
        block, block_names = pd.factorize(registered_blocks(a.datum), sort=True)
    elif scheme=='station':
        block, block_names = np.zeros(len(a),int), ['all']
    else:
        raise ValueError(scheme)
    ns, nb = len(st_names),len(block_names)
    count=np.zeros((ns,nb)); np.add.at(count,(st,block),1)
    errors=[x.prediction.to_numpy(float)-a.observed.to_numpy(float) for x in (a,b)]
    if not all(np.isfinite(x).all() for x in errors): raise ValueError('nonfinite error')
    matrices=[]
    for error in errors:
        transformed=error**2 if metric=='RMSE' else abs(error) if metric=='MAE' else error
        mat=np.zeros_like(count); np.add.at(mat,(st,block),transformed); matrices.append(mat)
    if metric not in {'RMSE','MAE','bias'}: raise ValueError(metric)
    rng=np.random.default_rng(SEED)
    station_draws=rng.multinomial(ns,np.full(ns,1/ns),size=reps)
    block_draws=rng.multinomial(nb,np.full(nb,1/nb),size=reps) if scheme=='station_x_block' else np.ones((reps,1),int)
    denom=np.einsum('bi,ij,bj->b',station_draws,count,block_draws)
    if (denom==0).any(): raise ValueError('empty bootstrap draw')
    draws=[np.einsum('bi,ij,bj->b',station_draws,m,block_draws)/denom for m in matrices]
    if metric=='RMSE': draws=[np.sqrt(x) for x in draws]
    delta=draws[0]-draws[1]
    point=scores(a.observed,a.prediction)[metric]-scores(b.observed,b.prediction)[metric]
    support_hash=hashlib.sha256(a[keys+['observed']].to_csv(index=False).encode()).hexdigest()
    return dict(metric=metric,units='ng m-3',direction='a_minus_b',estimate=point,
                ci_low=float(np.quantile(delta,.025)),ci_high=float(np.quantile(delta,.975)),
                n=len(a),n_stations=ns,n_blocks=nb,resampling=scheme,seed=SEED,
                replicates=reps,support_sha256=support_hash)

def reconstruct(reference, predictions, folds=range(9)):
    """One reconstruction per station-year-fold; fixed observed denominator.

    predictions is a complete, finite common-support subset of held-out keys.
    Nonbracketable observations remain observed for ALL methods; report counts.
    """
    ref=reference.copy(); ref['year']=pd.to_datetime(ref.datum).dt.year
    pred=predictions.copy(); pred['year']=pd.to_datetime(pred.datum).dt.year
    if ref.duplicated(['eoi','datum']).any() or pred.duplicated(['eoi','datum']).any(): raise ValueError('duplicate reconstruction keys')
    check=pred.merge(ref[['eoi','datum','observed']],on=['eoi','datum'],suffixes=('','_ref'),validate='one_to_one',how='left')
    if check.observed_ref.isna().any() or not np.array_equal(check.observed,check.observed_ref): raise ValueError('reconstruction target mismatch')
    rows=[]
    for (station,year), g in ref.groupby(['eoi','year']):
        if len(g)<20: continue
        s=pred[pred.eoi.eq(station)&pred.year.eq(year)]
        for fold in folds:
            hidden=s[s.fold.astype(str).eq(str(fold))]
            if not len(hidden): continue
            error=(hidden.prediction-hidden.observed).sum()/len(g)
            rows.append(dict(eoi=station,year=year,fold=fold,observed=g.observed.mean(),
                             prediction=g.observed.mean()+error,n_sampled=len(g),
                             n_replaced=len(hidden),fraction_replaced=len(hidden)/len(g)))
    return pd.DataFrame(rows)
