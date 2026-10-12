"""5G-NIDD (Combined.csv from the Kaggle mirror of the official release) -> features for federated runs.
Combined.csv concatenates the two base-station files: its index column restarts at row 728,316, giving two
segments of 728,316 and 487,574 flows, which we treat as base stations BS1 and BS2 (one FL client each).
Steps: drop the row index and the Argus sequence number; one-hot encode Proto, sDSb, dDSb, Cause, State;
missing numeric values -> 0 plus one missing-value indicator per column that has any; drop exact duplicates
(features + label) within each site; drop constant columns; then draw a uniform random sample of 25,000 flows per
site (seed 0), which keeps each site's natural class proportions. Label = 'Attack Type' (benign + 8 attacks)."""
import json, numpy as np, pandas as pd
d = pd.read_csv("Combined.csv", low_memory=False)
n0 = len(d)
u = d.pop("Unnamed: 0").to_numpy(); cut = int(np.where(np.diff(u) < 0)[0][0]) + 1
site = np.r_[np.zeros(cut, int), np.ones(n0 - cut, int)]
lab = d.pop("Attack Type").astype(str); d = d.drop(columns=["Label", "Attack Tool", "Seq"])
cat = ["Proto", "sDSb", "dDSb", "Cause", "State"]
X = pd.get_dummies(d[cat].astype(str), prefix=cat, dtype=np.float32)
num = d.drop(columns=cat).apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
miss = num.columns[num.isna().any()]
ind = num[miss].isna().astype(np.float32).add_suffix("_missing")
F = pd.concat([num.fillna(0).astype(np.float32), ind, X], axis=1)
key = pd.concat([F, lab.rename("__y"), pd.Series(site, name="__s")], axis=1)
dup = key.duplicated().to_numpy()
F, lab, site = F[~dup], lab[~dup], site[~dup]
const = [c for c in F.columns if F[c].nunique() <= 1]; F = F.drop(columns=const)
rng = np.random.default_rng(0)
idx = np.sort(np.concatenate([rng.choice(np.where(site == s)[0], 25000, replace=False) for s in (0, 1)]))
F, lab, site = F.iloc[idx], lab.iloc[idx], site[idx]
names = sorted(lab.unique()); names.remove("Benign"); names = ["Benign"] + names
np.save("X.npy", F.to_numpy(np.float32)); np.save("y.npy", lab.map({n: i for i, n in enumerate(names)}).to_numpy(np.int64))
np.save("day.npy", site.astype(np.int64)); np.save("t.npy", np.arange(len(site))); np.save("conflict_mask.npy", np.zeros(len(site), bool))
info = dict(rows_in=n0, site_rows=[int(cut), int(n0 - cut)], dropped_duplicates=int(dup.sum()), dropped_constant_columns=const,
            n_features=F.shape[1], features=list(F.columns), rows_out=len(lab), label_names=names,
            class_counts=lab.value_counts().reindex(names).astype(int).to_dict(),
            site_class=pd.crosstab(site, lab).reindex(columns=names).to_dict())
json.dump(info, open("prep_info.json", "w"), indent=1, default=int)
print(json.dumps({k: v for k, v in info.items() if k != "features"}, indent=1, default=int))
