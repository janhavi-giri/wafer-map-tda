"""Compare clustering / classification on raw pixels vs geometric vs topology features
as the generator variation level increases.

Usage: python scripts/variation_comparison.py OUTPUT_DIR   (about 5 minutes; writes hard_data.pkl and hard_results.csv)
"""
import warnings, io, contextlib, sys, numpy as np, pandas as pd, pickle
warnings.filterwarnings("ignore")
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score as ARI
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from wafer_map_tda import SyntheticWaferMapGenerator, TDAFeatureExtractor
types=['Center','Donut','Edge-Loc','Edge-Ring','Loc','Random','Scratch','Near-full','None']
levels=["clean","mild","moderate","hard"]; N=60
data={}
for seed in (42,7,123):
    for lv in levels:
        np.random.seed(seed*10+levels.index(lv))
        g=SyntheticWaferMapGenerator(variation=lv)
        smp=[g.generate_wafer_map(t) for t in types for _ in range(N)]
        M=np.array([x[0] for x in smp]); y=np.array([x[1] for x in smp])
        with contextlib.redirect_stdout(io.StringIO()):
            X,_=TDAFeatureExtractor(max_dimension=1).fit_transform(M)
        data[(seed,lv)]=(M,y,X)
    print("seed",seed,"done",flush=True)
pickle.dump(data,open(sys.argv[1]+"/hard_data.pkl","wb"))
def sets(M,X):
    return {"Raw pixels (PCA-20)":("pix",M),"Geometric (8)":("f",X[:,-8:]),"Topology (26)":("f",X[:,:26]),"Topology + geometry (34)":("f",X)}
rows=[]
for (seed,lv),(M,y,X) in data.items():
    for fs,(kind,F) in sets(M,X).items():
        if kind=="pix": F=PCA(20,random_state=0).fit_transform(StandardScaler().fit_transform(F.reshape(len(F),-1)))
        Z=np.nan_to_num(StandardScaler().fit_transform(F))
        for m,l in {"k-means":KMeans(9,n_init=10,random_state=0).fit_predict(Z),"Ward":AgglomerativeClustering(9).fit_predict(Z)}.items():
            rows.append(dict(seed=seed,level=lv,features=fs,metric=f"ARI {m}",value=ARI(y,l)))
        acc=cross_val_score(RandomForestClassifier(200,random_state=0),F,y,cv=StratifiedKFold(5,shuffle=True,random_state=0)).mean()
        rows.append(dict(seed=seed,level=lv,features=fs,metric="RF CV acc",value=acc))
# train on clean, test on each level (same seed)
for seed in (42,7,123):
    Mc,yc,Xc=data[(seed,"clean")]
    for lv in levels:
        Mt,yt,Xt=data[(seed,lv)]
        for fs in ["Raw pixels (PCA-20)","Geometric (8)","Topology (26)","Topology + geometry (34)"]:
            def get(M,X):
                k,F=sets(M,X)[fs]; return F.reshape(len(F),-1) if k=="pix" else F
            Ftr,Fte=get(Mc,Xc),get(Mt,Xt)
            if fs.startswith("Raw"):
                sc=StandardScaler().fit(Ftr); pca=PCA(20,random_state=0).fit(sc.transform(Ftr)); Ftr,Fte=pca.transform(sc.transform(Ftr)),pca.transform(sc.transform(Fte))
            rf=RandomForestClassifier(200,random_state=0).fit(Ftr,yc)
            rows.append(dict(seed=seed,level=lv,features=fs,metric="Train clean→test acc",value=(rf.predict(Fte)==yt).mean()))
d=pd.DataFrame(rows); d.to_csv(sys.argv[1]+"/hard_results.csv",index=False)
t=d.groupby(["metric","features","level"]).value.mean().round(3).unstack("level")[levels]
pd.set_option("display.width",200); print(t.to_string())
