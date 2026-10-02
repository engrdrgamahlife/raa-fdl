"""5GAD per-packet feature extraction (follows the dataset authors' labelling convention).

attack = every payload-bearing packet in Attacks/<name>/Attacks_<name>.pcapng
normal = every payload-bearing packet in the Normal-1UE capture(s) used
features: first 128 payload bytes (/255, zero-padded), log payload length, transport one-hot
          (TCP, UDP, SCTP, other), log1p(src port), log1p(dst port)  -> 136 features
group: capture interface (eno1, enp5s0, lo, upfgtp) -> one FL client each
"""
import glob, sys, numpy as np, pandas as pd
from scapy.utils import PcapNgReader
from scapy.all import Raw, TCP, UDP, IP, IPv6
from scapy.layers.sctp import SCTP
NB = 128
def feats(pkt):
    raw = bytes(pkt[Raw].load)
    v = np.zeros(NB + 7, np.float32)
    b = np.frombuffer(raw[:NB], np.uint8)
    v[:len(b)] = b / 255.0
    v[NB] = np.log1p(len(raw))
    if TCP in pkt: v[NB+1] = 1; sp, dp = pkt[TCP].sport, pkt[TCP].dport
    elif UDP in pkt: v[NB+2] = 1; sp, dp = pkt[UDP].sport, pkt[UDP].dport
    elif SCTP in pkt: v[NB+3] = 1; sp, dp = pkt[SCTP].sport, pkt[SCTP].dport
    else: v[NB+4] = 1; sp = dp = 0
    v[NB+5], v[NB+6] = np.log1p(sp), np.log1p(dp)
    return v
def read(path, label, source):
    rows, meta = [], []
    with PcapNgReader(path) as r:
        for pkt in r:
            if Raw not in pkt: continue
            rows.append(feats(pkt)); meta.append((getattr(pkt, "sniffed_on", None) or "unknown", label, source, float(pkt.time)))
    return rows, meta
X, M = [], []
for f in sorted(glob.glob("att/Attacks/*/Attacks_*.pcapng")):
    r, m = read(f, 1, f.split("/")[2]); X += r; M += m; print(f, len(r), flush=True)
for f in sorted(glob.glob("norm/*.pcapng")):
    r, m = read(f, 0, "normal:" + f.split("/")[-1]); X += r; M += m; print(f, len(r), flush=True)
X = np.stack(X)
meta = pd.DataFrame(M, columns=["interface", "label", "source", "time"])
np.save("work/X.npy", X); meta.to_csv("work/meta.csv", index=False)
print(meta.groupby(["interface", "label"]).size().unstack(fill_value=0))
