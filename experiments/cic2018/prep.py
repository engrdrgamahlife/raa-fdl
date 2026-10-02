"""CSE-CIC-IDS2018 sample -> features. Documented preprocessing (manuscript Sec. 3.3).

Input: cic2018_sample.csv.gz (uniform sample of <= 20,000 records per class per capture-day file).
Steps: harmonise columns; drop identifiers (Flow ID, Src IP, Dst IP, Src Port, Timestamp); coerce to numeric;
+/-inf -> NaN; drop rows with NaN; drop exact duplicates (features + label); drop constant columns;
cap every class at CAP records overall (uniform, seeded) to fit the compute budget; encode labels in sorted order;
capture day = source file (one of ten).
"""
import sys, json, numpy as np, pandas as pd
CAP = int(sys.argv[2]) if len(sys.argv) > 2 else 5000
df = pd.read_csv(sys.argv[1], low_memory=False)
df.columns = [c.strip() for c in df.columns]
n0 = len(df)
day = df.pop("SourceFile").astype(str)
drop = [c for c in ["Flow ID", "Src IP", "Dst IP", "Src Port", "Timestamp"] if c in df.columns]
df = df.drop(columns=drop)
lab = df.pop("Label").astype(str).str.strip()
X = df.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
ok = X.notna().all(axis=1)
X, lab, day = X[ok], lab[ok], day[ok]
n_nan = int((~ok).sum())
dup = pd.concat([X, lab.rename("__y")], axis=1).duplicated()
X, lab, day = X[~dup], lab[~dup], day[~dup]
n_dup = int(dup.sum())
const = [c for c in X.columns if X[c].nunique() <= 1]
X = X.drop(columns=const)
rng = np.random.default_rng(0)
idx = np.concatenate([rng.permutation(np.where(lab.values == c)[0])[:CAP] for c in sorted(lab.unique())])
idx.sort()
X, lab, day = X.iloc[idx], lab.iloc[idx], day.iloc[idx]
names = sorted(lab.unique()); days = sorted(day.unique())
np.save("X.npy", X.to_numpy(np.float32))
np.save("y.npy", lab.map({n: i for i, n in enumerate(names)}).to_numpy(np.int64))
np.save("day.npy", day.map({d: i for i, d in enumerate(days)}).to_numpy(np.int64))
info = dict(rows_in=n0, dropped_nan_inf=n_nan, dropped_duplicates=n_dup, dropped_constant_columns=const,
            dropped_identifiers=drop, n_features=X.shape[1], features=list(X.columns), cap_per_class=CAP,
            rows_out=len(lab), label_names=names, days=days,
            class_counts=lab.value_counts().reindex(names).astype(int).to_dict(),
            day_class=pd.crosstab(day, lab).to_dict())
json.dump(info, open("prep_info.json", "w"), indent=1, default=int)
print(json.dumps({k: v for k, v in info.items() if k not in ("features", "day_class")}, indent=1, default=int))
