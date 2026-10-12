"""Recreate prep.py's row selection and save the flow timestamp of every kept row (t.npy, seconds), plus a
conflicting-label report (identical feature vectors carrying different labels)."""
import json, numpy as np, pandas as pd
df = pd.read_csv("cic2018_sample.csv.gz", low_memory=False)
df.columns = [c.strip() for c in df.columns]
day = df.pop("SourceFile").astype(str)
ts = pd.to_datetime(df["Timestamp"], dayfirst=True, errors="coerce")
df = df.drop(columns=[c for c in ["Flow ID", "Src IP", "Dst IP", "Src Port", "Timestamp"] if c in df.columns])
lab = df.pop("Label").astype(str).str.strip()
X = df.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
ok = X.notna().all(axis=1)
X, lab, day, ts = X[ok], lab[ok], day[ok], ts[ok]
dup = pd.concat([X, lab.rename("__y")], axis=1).duplicated()
X, lab, day, ts = X[~dup], lab[~dup], day[~dup], ts[~dup]
const = [c for c in X.columns if X[c].nunique() <= 1]
X = X.drop(columns=const)
# conflicting labels among the deduplicated records (before the per-class cap)
g = pd.concat([X.reset_index(drop=True), lab.reset_index(drop=True).rename("__y")], axis=1)
key = pd.util.hash_pandas_object(g.drop(columns="__y"), index=False)
nlab = g.groupby(key)["__y"].nunique()
conf_keys = set(nlab[nlab > 1].index)
conf = g[key.isin(conf_keys).values]
rep = {"n_records_dedup": int(len(g)), "n_feature_vectors_with_conflicting_labels": int(len(conf_keys)),
       "n_records_involved": int(len(conf)),
       "records_involved_by_class": conf["__y"].value_counts().to_dict(),
       "class_pairs": pd.Series([" | ".join(sorted(v)) for v in conf.groupby(key[key.isin(conf_keys)].values)["__y"].unique()]).value_counts().to_dict()}
rng = np.random.default_rng(0)
idx = np.concatenate([rng.permutation(np.where(lab.values == c)[0])[:5000] for c in sorted(lab.unique())])
idx.sort()
Xk, tk = X.iloc[idx].to_numpy(np.float32), ts.iloc[idx]
assert np.array_equal(Xk, np.load("X.npy")), "row selection differs from prep.py"
# conflicts among kept rows
kept_key = key.iloc[idx]
rep["kept_records_with_conflicting_labels"] = int(kept_key.isin(conf_keys).sum())
rep["kept_by_class"] = lab.iloc[idx][kept_key.isin(conf_keys).values].value_counts().to_dict()
rep["timestamp_missing"] = int(tk.isna().sum())
np.save("t.npy", tk.to_numpy().astype("datetime64[s]").astype("int64")); np.save("conflict_mask.npy", kept_key.isin(conf_keys).to_numpy())
json.dump(rep, open("conflicts.json", "w"), indent=1, default=int)
print(json.dumps(rep, indent=1, default=int))
