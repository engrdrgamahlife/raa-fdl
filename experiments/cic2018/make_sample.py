"""Create cic2018_sample.csv.gz: a uniform random sample of up to 20,000 records per class from each of the ten
CSE-CIC-IDS2018 'Processed Traffic Data for ML Algorithms' CSV files. Run it in the folder that holds the CSVs,
or set CSV_GLOB. Downloading the CSVs: aws s3 sync --no-sign-request --region ca-central-1
"s3://cse-cic-ids2018/Processed Traffic Data for ML Algorithms/" cic"""
import glob, os, numpy as np, pandas as pd
CAP, rng, out = 20000, np.random.default_rng(0), []
for f in sorted(glob.glob(os.environ.get("CSV_GLOB", "cic/*.csv"))):
    keep = {}
    for ch in pd.read_csv(f, chunksize=500_000, low_memory=False):
        ch.columns = [c.strip() for c in ch.columns]
        ch = ch[ch["Label"] != "Label"].copy()
        ch["_r"] = rng.random(len(ch))
        for lab, g in ch.groupby("Label"):
            g = pd.concat([keep[lab], g]) if lab in keep else g
            keep[lab] = g.nsmallest(CAP, "_r")
    for g in keep.values():
        out.append(g.drop(columns="_r").assign(SourceFile=os.path.basename(f)))
df = pd.concat(out)
df.to_csv("cic2018_sample.csv.gz", index=False, compression="gzip")
print(df.shape)
