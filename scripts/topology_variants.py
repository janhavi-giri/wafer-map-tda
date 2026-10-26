"""Stage 1 (dev seeds 42, 7): choose the point-cloud threshold. Stage 2 (held-out seeds 2001-2003): evaluate once.
Usage: python scripts/topology_variants.py OUTPUT_DIR   (about 20 minutes on 2 CPU cores)"""
import warnings, io, contextlib, sys, pickle, numpy as np, pandas as pd
from multiprocessing import Pool
warnings.filterwarnings("ignore")
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score as ARI
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from wafer_map_tda import SyntheticWaferMapGenerator, TDAFeatureExtractor
OUT=sys.argv[1]
EX=TDAFeatureExtractor(max_dimension=1)
types=['Center','Donut','Edge-Loc','Edge-Ring','Loc','Random','Scratch','Near-full','None']
levels=["clean","mild","moderate","hard"]; N=60
H0_C=np.linspace(0.25,3.75,8); H1_C=np.linspace(0.5,9.5,10)

def profile(lifetimes, centers, sigma, weight):
    if len(lifetimes)==0: return np.zeros(len(centers))
    k=np.exp(-0.5*((centers[:,None]-lifetimes[None,:])/sigma)**2)
    return (k*(lifetimes[None,:] if weight else 1.0)).sum(1)

def topo(args):
    w,t=args
    pts=EX._extract_point_cloud(w,threshold=t); dg=EX._compute_persistence_diagram(pts)
    f=[]
    for d in range(2): f.extend(EX._persistence_statistics(dg.get(d,np.empty((0,2)))).values())
    b=EX._compute_betti_numbers(dg,np.linspace(0,2,20))
    for d in range(2): c=b[:,d]; f.extend([c.mean(),c.std(),c.max(),c.sum()])
    h0=dg[0][np.isfinite(dg[0][:,1]),1] if len(dg[0]) else np.empty(0)
    h1=(dg[1][:,1]-dg[1][:,0]) if len(dg.get(1,[])) else np.empty(0)
    pi=np.concatenate([profile(h0,H0_C,0.4,False),profile(h1,H1_C,0.7,True)])
    return np.array(f),pi

def feats(M,t,pool):
    r=pool.map(topo,[(w,t) for w in M],chunksize=20)
    return np.array([a for a,_ in r]),np.array([b for _,b in r])

def geo(M): return np.array([EX._extract_geometric_features(w) for w in M])

def gen(seed,lv):
    np.random.seed(seed*10+levels.index(lv)); g=SyntheticWaferMapGenerator(variation=lv)
    s=[g.generate_wafer_map(t) for t in types for _ in range(N)]
    return np.array([x[0] for x in s]),np.array([x[1] for x in s])

def cluster_scores(F,y):
    Z=np.nan_to_num(StandardScaler().fit_transform(F))
    return {"ARI k-means":ARI(y,KMeans(9,n_init=10,random_state=0).fit_predict(Z)),
            "ARI Ward":ARI(y,AgglomerativeClustering(9).fit_predict(Z))}

if __name__=="__main__":
    pool=Pool(2)
    # sanity: mirror equals package features
    w=gen(42,"mild")[0][5]; assert np.allclose(topo((w,0.1))[0],EX.extract_features(w)[:26]), "mirror mismatch"
    # ---- Stage 1: dev seeds, choose threshold by topology-only k-means ARI (mild/moderate/hard)
    dev={}; thr=[0.1,0.2,0.3,0.4,0.5]; rows=[]
    for seed in (42,7):
        for lv in ["mild","moderate","hard"]:
            M,y=gen(seed,lv)
            for t in thr:
                T,_=feats(M,t,pool); rows.append(dict(seed=seed,level=lv,thr=t,ARI=cluster_scores(T,y)["ARI k-means"]))
        print("dev seed",seed,flush=True)
    dv=pd.DataFrame(rows); print(dv.groupby("thr").ARI.mean().round(3)); 
    best=float(dv.groupby("thr").ARI.mean().idxmax()); print("chosen threshold",best,flush=True)
    dv.to_csv(OUT+"/dev_threshold.csv",index=False)
    # ---- Stage 2: held-out seeds, everything fixed
    rows=[]; store={}
    for seed in (2001,2002,2003):
        for lv in levels:
            M,y=gen(seed,lv); G=geo(M)
            T0,P0=feats(M,0.1,pool); T1,P1=feats(M,best,pool)
            store[(seed,lv)]=dict(M=M,y=y,G=G,T0=T0,P0=P0,T1=T1,P1=P1)
        print("heldout seed",seed,flush=True)
    pickle.dump(store,open(OUT+"/heldout_data.pkl","wb"))
    def fsets(d):
        pix=d["M"].reshape(len(d["M"]),-1)
        return {"Raw pixels (PCA-20)":pix,"Geometric (8)":d["G"],
                "Topology, threshold 0.1 (26)":d["T0"],f"Topology, threshold {best} (26)":d["T1"],
                f"Lifetime profiles, threshold {best} (18)":d["P1"],
                "Topology 0.1 + geometry (34)":np.hstack([d["T0"],d["G"]]),
                f"Topology {best} + geometry (34)":np.hstack([d["T1"],d["G"]]),
                f"Lifetime profiles {best} + geometry (26)":np.hstack([d["P1"],d["G"]])}
    def prep(name,F):
        return PCA(20,random_state=0).fit_transform(StandardScaler().fit_transform(F)) if name.startswith("Raw") else F
    for (seed,lv),d in store.items():
        for name,F in fsets(d).items():
            F=prep(name,F)
            for m,v in cluster_scores(F,d["y"]).items(): rows.append(dict(seed=seed,level=lv,features=name,metric=m,value=v))
            rows.append(dict(seed=seed,level=lv,features=name,metric="RF CV acc",value=cross_val_score(RandomForestClassifier(200,random_state=0),F,d["y"],cv=StratifiedKFold(5,shuffle=True,random_state=0)).mean()))
    for seed in (2001,2002,2003):
        tr=store[(seed,"clean")]
        for lv in levels:
            te=store[(seed,lv)]
            for name in fsets(tr):
                Ftr,Fte=fsets(tr)[name],fsets(te)[name]
                if name.startswith("Raw"):
                    sc=StandardScaler().fit(Ftr); p=PCA(20,random_state=0).fit(sc.transform(Ftr)); Ftr,Fte=p.transform(sc.transform(Ftr)),p.transform(sc.transform(Fte))
                rf=RandomForestClassifier(200,random_state=0).fit(Ftr,tr["y"])
                rows.append(dict(seed=seed,level=lv,features=name,metric="Train clean→test acc",value=(rf.predict(Fte)==te["y"]).mean()))
    d=pd.DataFrame(rows); d.to_csv(OUT+"/heldout_results.csv",index=False)
    pd.set_option("display.width",220)
    print(d.groupby(["metric","features","level"]).value.mean().round(3).unstack("level")[levels].to_string())
