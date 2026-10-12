---
title: "Which Heterogeneity Remedy Helps Federated Intrusion Detection? A Tuned Comparison of Server-Side and Client-Side Methods on Enterprise and 5G Traffic"
bibliography: refs.bib
csl: springer-basic-brackets.csl
link-citations: true
---

**Amah Gideon Gbaden**

Department of Software Engineering, Federal University of Applied Sciences, Kachia, Kaduna State, Nigeria

E-mail: gideon.amah@fuask.edu.ng · ORCID: 0009-0003-8405-9906

**Abstract.** Sites that train an intrusion detector together by federated learning see different traffic, and many remedies for this heterogeneity exist. We compare them under one tuned protocol: fifteen training strategies covering server optimisers (FedAvgM, FedAdam), server reweighting (FedDisco, FLTrust, a reliability-aware rule), client-side corrections (FedProx, SCAFFOLD, logit adjustment, FedRS) and combinations, plus server-side classifier calibration (CCVR). The settings are CSE-CIC-IDS2018 under five client partitions, a cleaned 5GAD core capture and the two base stations of 5G-NIDD. Hyperparameters were tuned on validation data, and comparisons are paired over five seeds, Holm-corrected and backed by equivalence tests. Tuning mattered most: it raised FedAvg's macro-F1 on CSE-CIC-IDS2018 from 0.52 to 0.70. When sites shared classes, most remedies tied with FedAvg on the well-populated classes; server momentum and server Adam added up to 0.09 macro-F1, mostly by detecting three classes with fewer than 100 records. When one 5G client held every attack, FedAvg collapsed in two of five seeds, while logit adjustment and the reliability-aware rule never did. When clients held different capture days, every training-time remedy stayed below 0.31, whereas recalibrating the classifier at the server raised FedAvg from 0.27 to 0.66. Standard FLTrust was worse than FedAvg in every setting and did not protect against a label-flipping attack. When sites shared classes, benign false-positive rates were high for every method, mainly because benign and Infilteration flows overlap in CSE-CIC-IDS2018. The right remedy depends on how labels are spread across sites, which operators can measure before choosing.

**Keywords:** federated learning; network intrusion detection; 5G; non-IID data; classifier calibration; benchmarking

# Introduction

Operators increasingly want intrusion detectors trained on traffic from many sites. In a 5G network those sites include base stations, User Plane Function (UPF) instances and the network functions that exchange signalling over the service-based interface (SBI) [@chennoufi2024sok]. Moving raw flow records to one place costs backhaul capacity and exposes subscriber metadata. Federated learning (FL) avoids both: each site trains on its own records and sends only model updates to a server, which combines them [@mcmahan2017]. Since Release 18, 3GPP specifies federated learning among Network Data Analytics Function (NWDAF) instances, with one NWDAF acting as FL server and others as FL clients [@3gpp23288].

The difficulty is that sites differ. A UPF under attack sees attack classes that other sites never see, and the attacks seen on one day differ from those seen on another. FedAvg, the standard combination rule, weights each client by its data size, and its accuracy drops when clients hold different class mixtures [@zhao2018noniid; @hsu2019; @hsieh2020quagmire]. Many remedies exist. Some act at the server: server optimisers such as FedAvgM and FedAdam [@hsu2019; @reddi2021fedopt], reweighting rules that score clients by trust, reliability or label discrepancy [@cao2021fltrust; @ye2023feddisco], and post-hoc calibration of the classifier with client feature statistics [@luo2021ccvr]. Others act at the clients: proximal terms [@li2020fedprox], control variates [@karimireddy2020scaffold] and corrections of the output layer for each client's label distribution [@menon2021logit; @li2021fedrs; @zhang2022fedlc].

For a network operator the choice has practical consequences. A server-side method can be deployed at the aggregator, for example in the FL server NWDAF, without touching the sites. A client-side method must be rolled out to every site. Benchmarks on image and tabular data have shown that no federated algorithm dominates under non-IID data [@li2022niidbench], and surveys of federated intrusion detection note that most studies propose one method and compare it with FedAvg on one dataset [@lavaur2022survey; @campos2022]. Such studies cannot tell an operator which family to invest in, nor whether the gains survive proper tuning.

This paper compares the families directly, on enterprise flows and on two 5G datasets, under a protocol in which every method's hyperparameters are tuned on validation data. The contributions are:

1. A tuned comparison of fifteen training strategies, plus server-side calibration, over seven settings: five partitions of CSE-CIC-IDS2018, a cleaned version of the 5GAD core capture and the two base stations of 5G-NIDD (Section 5.2). Every comparison is paired by seed, Holm-corrected, and complemented by equivalence tests for null results.
2. Evidence that tuning changes the conclusions. At one untuned learning rate, every method was under-trained, and remedies that only add effective step size appeared to help or hurt for that reason (Section 5.1).
3. A diagnosis of the failure under label-disjoint sites. The learned features remain usable, and the failure lies in the classifier; calibrating it at the server recovers most of the lost accuracy (Section 5.3).
4. Operating-point, robustness, cost and poisoning analyses aimed at network operators: false-positive rates at a benign-dominated prior, normalisation and model choice, training length, chronological splits, partial participation, number of clients, communication and computation cost, and a label-flipping attack (Sections 5.4 to 5.7).

All code, configurations, tuning runs and per-seed results are public, so every number in the paper can be regenerated.

# Related Work

**Server-side methods.** FedAvg averages client models in proportion to their data size [@mcmahan2017]. FedAvgM adds momentum to the server update [@hsu2019], and FedOpt applies adaptive optimisers such as Adam at the server [@reddi2021fedopt]. FedNova normalises for unequal amounts of local work [@wang2020fednova]. Reweighting rules change each client's weight. FedDisco lowers the weight of clients whose label distribution departs from a uniform target [@ye2023feddisco]. FLTrust trains on a clean root dataset at the server and weights each client by the rectified cosine similarity between its update and the server's update, after rescaling every update to the server update's norm [@cao2021fltrust]; Zeno and Krum score updates for robustness to malicious clients [@xie2019zeno; @blanchard2017krum]. A third group recalibrates the classifier after or during training: CCVR samples virtual features from the class-wise feature statistics reported by clients and retrains the last layer at the server [@luo2021ccvr], and CReFF retrains the classifier on federated features [@shang2022creff].

**Client-side methods.** FedProx adds a proximal penalty that limits how far a client moves from the global model [@li2020fedprox]. SCAFFOLD estimates each client's gradient drift and subtracts it during local training [@karimireddy2020scaffold]. MOON and FedDyn regularise local models towards the global representation or objective [@li2021moon; @acar2021feddyn]. A separate line corrects the output layer for label skew. Logit adjustment and balanced softmax add the logarithm of the class prior to the logits during training [@menon2021logit; @ren2020bms]. FedRS restricts the softmax for classes a client does not hold [@li2021fedrs], and FedLC calibrates each client's logits with a margin that depends on its label counts [@zhang2022fedlc]. Normalisation layers are a known weak point: averaging BatchNorm statistics across clients with different feature distributions can harm the global model, which motivates GroupNorm and FedBN [@hsieh2020quagmire; @li2021fedbn].

**Federated intrusion detection.** FLAD adapts client selection and local training to each client's reported accuracy [@doriguzzi2024flad]. Chennoufi et al. survey federated intrusion detection for 5G and note the shortage of 5G-specific evaluation data [@chennoufi2024sok]. Singh et al. admit only trusted clients to protect 5G network slices against poisoning [@singh2025], and recent work in this journal studies grouped federations for IoT anomaly detection [@zhang2024fedgroup], adversarial robustness of federated recurrent detectors [@rezaei2025fedrnn] and federated detection aligned with the ETSI NFV MANO architecture [@benkhalifa2026v5gids]. Class imbalance is usually handled by resampling, such as stratified sampling, generative augmentation, class-specific dynamic sampling or per-client SMOTE [@weinger2022fl; @youm2025dsfedids; @paray2026smote]. We found fewer studies that correct the class prior in the loss or recalibrate the classifier at the server for intrusion detection, and none that compares these with server optimisers and reweighting rules under one tuned protocol. Evaluation pitfalls in machine learning for security, including duplicated records, label errors and temporal snooping, are well documented [@arp2022dos; @engelen2021; @liu2022error]; Section 4 describes how we handle the ones that apply here.

# Strategies Compared

## Learning task and notation

Each of $K$ clients holds labelled flow or packet records and trains a shared classifier $f_\theta$ over the $C$ classes of the dataset. Client $i$ holds $N_i$ training records, $N_{i,c}$ of them in class $c$. In round $t$ every participating client starts from the global model $\theta^t$, trains locally and returns $\theta_i^t$, with update $\Delta_i^t = \theta_i^t - \theta^t$. Let $\bar{\Delta}^t = \sum_i (N_i/\sum_j N_j)\,\Delta_i^t$ be the size-weighted mean update. All strategies aggregate BatchNorm running statistics with size weights. Table 1 lists the strategies, where each one acts, and what it costs beyond FedAvg. Fig. 1 shows the evaluation pipeline.

![Fig. 1. Evaluation pipeline. Each dataset is split into training, validation and test sets and the training set is partitioned across clients. Clients train locally, with or without a local correction; the server combines the updates by one of the aggregation rules. The validation and test macro-F1 are logged every five rounds (the test log is used only for the sensitivity analysis of Appendix B), and the reported results are those of the model after the last round; server-side calibration (CCVR) is applied after the last round.](fig1_pipeline.png){#fig:pipeline width="70%"}

Table: Table 1. Strategies compared. Extra cost is relative to FedAvg; $P$ is the number of model parameters (about 11,700 here), $C$ the number of classes and $d = 64$ the feature dimension of the classifier input.

| Strategy | Family | What changes | Extra communication | Extra server data or computation |
|:--|:--|:--|:--|:--|
| FedAvg [@mcmahan2017] | Baseline | Size-weighted average | None | None |
| FedAvgM [@hsu2019] | Server optimiser | Momentum 0.9 on $\bar{\Delta}^t$ | None | One state vector |
| FedAdam [@reddi2021fedopt] | Server optimiser | Adam step on $\bar{\Delta}^t$ | None | Two state vectors |
| FedDisco [@ye2023feddisco] | Server reweighting | Weights from size and label discrepancy | Label histogram, once | None |
| FLTrust [@cao2021fltrust] | Server reweighting | Cosine trust, norm rescaling | None | Root data; one epoch per round |
| RAA-FDL (Section 3.4) | Server reweighting | Reliability weights | None | Anchor data; $K$ forward passes per round |
| CCVR [@luo2021ccvr] | Server calibration | Classifier retrained once, after training | $C(d + d(d+1)/2)$ floats, once | Virtual-feature training |
| FedProx [@li2020fedprox] | Client correction | Proximal term | None | None |
| SCAFFOLD [@karimireddy2020scaffold] | Client correction | Control variates; uniform update average | $P$ floats each way per round | One state vector per client |
| SCAFFOLD-N | Client correction | As SCAFFOLD, size-weighted update average | $P$ floats each way per round | As SCAFFOLD |
| FedAvg + LA [@menon2021logit] | Client correction | Logit-adjusted local loss (Eq. 1) | None | None |
| FedRS [@li2021fedrs] | Client correction | Logits of absent classes scaled by 0.5 | None | None |
| SCAFFOLD-N + LA | Combination | Both of the above | As SCAFFOLD | As SCAFFOLD |
| SCAFFOLD-N + LA + RW | Combination | Plus reliability weights on the updates | As SCAFFOLD | Anchor data |

## Client-side corrections

With logit adjustment (LA), client $i$ trains on the loss

$$\ell_i(x, y) = -\log \frac{\exp\bigl(f_\theta(x)_y + \tau \log \pi_{i,y}\bigr)}{\sum_{c=1}^{C}\exp\bigl(f_\theta(x)_c + \tau \log \pi_{i,c}\bigr)}, \qquad \pi_{i,c} = \frac{N_{i,c} + \epsilon}{N_i + C\epsilon}, \tag{1}$$

where $\pi_{i,c}$ is the client's smoothed share of class $c$. The offset is applied only during local training; the global model predicts with the plain logits. We use $\tau = 1$ and $\epsilon = 1$ by default and report a sensitivity analysis over both (Section 5.5). FedRS multiplies the logits of classes absent from the client by $\alpha = 0.5$ during local training [@li2021fedrs]. FedProx adds $\frac{\mu}{2}\lVert\theta - \theta^t\rVert^2$ to the local loss. SCAFFOLD uses Option II control variates and a global step size of 1, so the global model moves by the uniform mean of the updates [@karimireddy2020scaffold]; SCAFFOLD-N weights the updates by data size instead.

## Server-side methods

FedAvgM keeps a momentum buffer $m^t = \beta m^{t-1} + \bar{\Delta}^t$ and sets $\theta^{t+1} = \theta^t + \eta_s m^t$ with $\beta = 0.9$ [@hsu2019]. FedAdam applies an Adam step with $\beta_1 = 0.9$, $\beta_2 = 0.99$ and $\tau = 10^{-3}$ to $\bar{\Delta}^t$ [@reddi2021fedopt]. FedDisco weights client $i$ by $p_i \propto \mathrm{ReLU}(n_i - a\,d_i + b)$, where $n_i$ is the client's data share and $d_i$ the Kullback-Leibler divergence between its label distribution and the uniform distribution, rescaled to sum to one over clients; we use $a = 0.5$ and $b = 0.1$ as in the official code [@ye2023feddisco]. FLTrust uses the anchor set (below) as its root data, trains on it for one epoch per round and rescales every client update to the norm of the server update [@cao2021fltrust]. To separate the effect of the rescaling from that of the root set's size, we also run FLTrust without rescaling and with a 200-record root set.

CCVR is applied once, after the last round [@luo2021ccvr]. Each client computes the mean and covariance of the global model's penultimate features for each class it holds and sends them with its class counts. The server combines them into global class-wise Gaussians, samples 2,000 virtual features per class, and retrains only the final linear layer for 50 epochs (SGD, learning rate 0.01, momentum 0.9, batch 256), starting from the trained classifier. These settings were fixed before any CCVR result was seen. CCVR can be combined with any training strategy; we apply it to the final model of each.

## A reliability-aware rule (RAA-FDL)

We also evaluate a reliability-aware rule, RAA-FDL, which combines four scores in $[0,1]$ for every client $i$ that delivered an update in round $t$: $A_i^t$, the macro-F1 of the client's model on a labelled anchor set held by the server; $S_i^t = 1/(1 + \mathrm{CV}_i^t)$, where $\mathrm{CV}_i^t$ is the coefficient of variation of the norms of the client's last five updates; $C_i^t$, the fraction of its last five scheduled rounds in which the client's update arrived; and $D_i^t = (1 + \cos(\Delta_i^t, \bar{\Delta}^t))/2$. The scores are combined, smoothed and mapped to weights:

$$R_i^t = \beta_1 A_i^t + \beta_2 S_i^t + \beta_3 C_i^t + \beta_4 D_i^t, \qquad \sum_{k}\beta_k = 1, \tag{2}$$

$$\tilde{R}_i^t = \lambda R_i^t + (1-\lambda)\tilde{R}_i^{t-1}, \qquad \tilde{R}_i^0 = 1/K, \tag{3}$$

$$w_i^t = \frac{N_i^{\gamma}\exp(\tilde{R}_i^t/\tau_R)}{\sum_{j} N_j^{\gamma}\exp(\tilde{R}_j^t/\tau_R)}, \tag{4}$$

$$w_i^t \leftarrow \frac{\max(w_i^t, w_{\min})}{\sum_j \max(w_j^t, w_{\min})}, \qquad \theta^{t+1} = \sum_i w_i^t \theta_i^t, \tag{5}$$

with $\beta = (0.4, 0.2, 0.2, 0.2)$, $\lambda = 0.5$, $\tau_R = 0.15$, $\gamma = 0.5$ and $w_{\min} = 0.02$. In SCAFFOLD-N + LA + RW the reliability weights replace the size weights on the SCAFFOLD updates, with $\gamma = 1$ and $\beta = (0.5, 0.25, 0.25, 0)$. Those two settings were chosen by validation macro-F1 on the original 5GAD capture, which is not used for any reported result (Section 4.2).

The anchor set used by RAA-FDL and FLTrust, and the root data of FLTrust, is the validation split. The validation curves of these two strategies are therefore computed on data they used during training, and we do not compare validation scores across strategies; all reported comparisons use the test split.

# Experimental Design

## CSE-CIC-IDS2018

CSE-CIC-IDS2018 [@sharafaldin2018] contains CICFlowMeter flow records from ten capture days, labelled with benign traffic and 14 attack classes. We took a uniform random sample of up to 20,000 records per class from each day's file, giving {prep[rows_in]:,} records. We dropped the identifier columns (flow ID, source and destination address, source port and timestamp), {prep[dropped_nan_inf]:,} records with infinite or missing values, {prep[dropped_duplicates]:,} exact duplicates and {prep[n_const]} constant columns, which left {prep[n_features]} features. To fit the computing budget we kept at most {prep[cap_per_class]:,} records per class, giving {prep[rows_out]:,} records. Each seed splits these 70/15/15 into training, validation and test sets, stratified by class, and fits the standardisation scaler on the training split only.

**Data-quality checks.** Duplicates and label errors are known problems in this dataset family [@engelen2021; @liu2022error]. After deduplication, {conf[n_feature_vectors_with_conflicting_labels]:,} distinct feature vectors still carry more than one label ({conf[n_records_involved]:,} records). Almost all of them pair Benign with Infilteration (sic; the dataset's own spelling, which we keep) ({conf[benign_infil]:,} vectors); the rest involve the small classes, for example DoS-SlowHTTPTest with FTP-BruteForce ({conf[slow_ftp]} vectors). In the capped sample, {conf[kept_records_with_conflicting_labels]} records are involved, including {conf[kept_ftp]} of the 45 FTP-BruteForce and {conf[kept_slow]} of the 47 DoS-SlowHTTPTest records. Three classes (DoS-SlowHTTPTest, FTP-BruteForce and SQL Injection) keep fewer than 100 records, so their test splits hold seven to thirteen records. We therefore report, besides macro-F1 over all 15 classes, macro-F1 over the 12 classes that keep at least 228 records ("learnable-class macro-F1"), and we repeat the core comparison with the conflicting records removed (Section 5.5).

**Leakage and shortcuts.** A random split puts flows from the same attack session into both training and test data. We therefore also run a per-class chronological split, in which the earliest 70% of each class's flows train, the next 15% validate and the latest 15% test, so no test flow precedes a training flow of its class. We keep the destination port as a feature, because operators do use it, and we test whether the conclusions depend on it by repeating the chronological runs without it (Section 5.5).

**Partitions.** The training split is divided among $K = 10$ clients in five ways. The IID partition deals records at random. Three Dirichlet partitions draw each class's share per client from a Dirichlet distribution with $\alpha = 1.0$, $0.5$ or $0.1$. The per-day partition gives each of the ten clients the records of one capture day. Because each day contains benign traffic and a different set of attacks, this partition is label-disjoint across clients except for the benign class; we call it per-day rather than temporal, since training and test data come from the same days. Table 2 measures each partition by the mean pairwise Jensen-Shannon distance between client label distributions and by the mean ROC AUC of a logistic classifier trained to tell one client's features from the others'. Because clients with different classes are separable even without any feature shift, we also report this AUC computed on benign records only. Benign-only AUC is near 0.5 for the Dirichlet partitions, which therefore have label skew without feature skew, and {shift[temporal]} for the per-day partition, which has both.

Table: Table 2. Heterogeneity of the partitions (mean over five seeds). Feature-shift AUC: ROC AUC of a classifier separating one client's features from the others' (0.5 = no shift).

| Setting | Clients | Label distance (JS) | Feature-shift AUC, all records | Feature-shift AUC, benign only |
|:--|:--:|:--:|:--:|:--:|
{T:part_rows}

## 5G datasets

**5GAD, cleaned.** 5GAD [@coldwell2022] contains packet captures from a free5GC core recorded on four interfaces: two host network interfaces (eno1, enp5s0), the loopback interface (lo), which carries the SBI traffic between network functions, and the GTP-U tunnel interface (upfgtp). Following the dataset's own preparation script, payload-bearing packets in the ten attack-isolated captures are labelled *attack* and those in the one-UE normal capture `allcap_00006` *normal*, of which we use a random 20,000. The FakeAMFDelete "isolated" capture is in fact a full capture that also contains background traffic labelled as attack, and it supplied 19,459 of the 24,160 attack packets. We removed it entirely. The remaining 4,701 attack packets all travel over the SBI, so one client (lo) holds every attack and three hold none. Each packet has 135 features: its first 128 payload bytes scaled to $[0,1]$, the logarithm of its payload length, a one-hot transport protocol and its log-scaled ports. Each interface is one client ($K = 4$). Pooled, the task is easy (logistic regression on ports and protocol alone reaches 0.99 macro-F1 on the original labelling), so we treat this setting as a stress test of extreme label skew across co-located functions rather than as representative 5G traffic.

**5G-NIDD.** 5G-NIDD [@samarakoon2022; @siriwardhana2025nidd] contains Argus flow records captured at two base stations of a 5G test network, with benign traffic and eight attacks (ICMP, UDP, SYN and HTTP floods, slow-rate DoS, and SYN, TCP-connect and UDP scans). We used the released combined file, in which the row index restarts once, at row 728,316, splitting the file into the two base-station captures of 728,316 and 487,574 flows; each base station is one client ($K = 2$). We dropped the row index and sequence number, one-hot encoded the categorical fields, set missing values to zero with a missing-value indicator per affected column, removed exact duplicates within each site, and drew a uniform random sample of 25,000 flows per site, which keeps each site's natural class mix. This leaves 96 features. The two sites differ mainly in their class proportions: benign traffic makes up {nidd[b0]} of the first site's sample and {nidd[b1]} of the second's, and UDP floods {nidd[u0]} and {nidd[u1]}. Every class is present at both sites.

## Model, training and tuning

Every client trains the same one-dimensional convolutional network: two convolutional layers (32 and 64 channels, kernel 3, batch normalisation, ReLU, max pooling after the first), global average pooling, and two fully connected layers (64 units, then one output per class). It has about 11,700 parameters. Local training uses SGD with batch size 256 and three epochs per round, for 100 rounds; all clients take part in every round unless stated otherwise. Each configuration runs with five seeds (42 to 46). The seed fixes the data split, the partition, the model initialisation and the order of local batches, and all strategies share it, so results are paired by seed.

**Tuning.** Hyperparameters were selected by the mean validation macro-F1 after 100 rounds, using seeds 42 to 44; test data were never used. On CSE-CIC-IDS2018 we tuned on the Dirichlet $\alpha = 0.5$ and per-day partitions. The local learning rate was searched over $\{0.003, 0.01, 0.03, 0.1, 0.3, 1.0\}$ for FedAvg, FedAvg + LA and SCAFFOLD; the rate with the best mean over these three strategies, 0.3, is used for every strategy. At that rate we tuned FedAvgM's server learning rate over $\{0.1, 0.3, 1.0\}$ (selected 1.0), FedAdam's over $\{0.01, 0.03, 0.1\}$ (selected 0.1) and FedProx's $\mu$ over $\{0.001, 0.01, 0.1\}$ (selected 0.001). 5G-NIDD and 5GAD were tuned the same way on their own validation splits, selecting local learning rates of 0.3 and 0.003; server settings were taken from CSE-CIC-IDS2018. Appendix A lists all tuning results. The submitted version of this study used a single learning rate of 0.01 for every strategy; Section 5.1 compares the two protocols.

**Reference models.** On CSE-CIC-IDS2018 we trained the same network centrally on the pooled training split for 300 epochs (the sample passes of 100 rounds of three local epochs), with the learning rate selected on validation data from {0.01, 0.03, 0.1, 0.3, 1.0}, and did the same for a multilayer perceptron (two hidden layers of 128 and 64 units). Gradient-boosted trees serve as a strong non-neural reference.

## Metrics and statistics

We report macro-F1 over the dataset's classes after the last round. For operators we also report, on CSE-CIC-IDS2018, the benign false-positive rate (the share of benign test flows labelled as any attack) and the detection rate (the share of attack flows labelled as any attack). Because the capped test set is close to class-balanced, we also report macro-F1 at a benign-dominated prior, obtained by reweighting the test flows so that benign traffic makes up 83% of the total, the share in the full dataset [@leevy2020survey], while attacks keep their relative proportions.

Each strategy is compared with FedAvg by a paired $t$-test over the five seeds. We report the mean paired difference, its 95% confidence interval and the number of seeds in which the strategy scored higher. Holm-Bonferroni correction is applied within each setting, over the comparisons in a column of Table 5 (fourteen; eleven for 5G-NIDD, where three strategies were not run); we also report how many results survive correction across all comparisons in the table. To support claims that two strategies perform alike, we use two one-sided tests (TOST) with an equivalence margin of $\pm 0.02$ macro-F1, fixed before the analysis. With five seeds a two-sided Wilcoxon test cannot reach $p < 0.0625$, so we do not use it. Because results at a single round can depend on fluctuations, we also report two alternatives in Appendix B: the mean test macro-F1 over rounds 85 to 100, and the test macro-F1 at the round with the best validation score.

All strategies are implemented in PyTorch in one code base. The experiments ran on a 2-core cloud CPU instance; runs saved a checkpoint every five rounds and we checked that a resumed run reproduces an uninterrupted one exactly.

# Results

## Effect of tuning

Table 3 compares the submitted protocol, with a learning rate of 0.01 for every strategy, with the tuned protocol. Tuning raised FedAvg's macro-F1 by 0.18 to 0.30 under the IID and Dirichlet partitions and from {tune[fedavg_temporal_old]} to {tune[fedavg_temporal_new]} under the per-day partition. The centralised network gained in the same way, from 0.645 to {central[cnn]}, and a centralised multilayer perceptron reached {central[mlp]}, above gradient-boosted trees ({central[hgb]}). At the untuned rate every federated method was therefore under-trained, and differences between methods partly reflected how much each one enlarged the effective step size. Tuning also changed the ranking. At the untuned rate, logit adjustment was the only strategy at or above FedAvg in every CSE-CIC-IDS2018 partition and reliability weights changed nothing. At the tuned rate, logit adjustment no longer helped on CSE-CIC-IDS2018, and the server optimisers gave the largest gains. All conclusions below refer to the tuned protocol.

Table: Table 3. Test macro-F1 on CSE-CIC-IDS2018 under the submitted protocol (one learning rate, 0.01) and the tuned protocol (mean over five seeds).

| Strategy | Protocol | IID | Dir. 1.0 | Dir. 0.5 | Dir. 0.1 | Per-day |
|:--|:--|:--:|:--:|:--:|:--:|:--:|
{T:tune_rows}

## Main comparison

Table 4 gives the test macro-F1 of every strategy in every setting, and Table 5 and Fig. 2 the paired differences from FedAvg.

Table: Table 4. Test macro-F1 after 100 rounds at the tuned settings (mean ± SD over five seeds). Columns 2 to 6: CSE-CIC-IDS2018 partitions ($K = 10$); 5GAD-c: cleaned 5GAD ($K = 4$); 5G-NIDD: two base stations ($K = 2$). FedRS equals FedAvg on 5G-NIDD because every class is present at both sites. "-": not run.

| Strategy | IID | Dir. 1.0 | Dir. 0.5 | Dir. 0.1 | Per-day | 5GAD-c | 5G-NIDD |
|:--|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
{R:main_rows}

Table: Table 5. Difference from FedAvg in test macro-F1 (mean over five paired seeds), with the number of seeds in which the strategy beat FedAvg. \* Holm-adjusted $p < 0.05$ within the column (14 comparisons; 11 for 5G-NIDD); ‡ equivalent to FedAvg within ±0.02 (TOST, $p < 0.05$). Confidence intervals for every cell are in the supplementary file paired_vs_fedavg.csv.

| Strategy | IID | Dir. 1.0 | Dir. 0.5 | Dir. 0.1 | Per-day | 5GAD-c | 5G-NIDD |
|:--|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
{R:vs_rows}

![Fig. 2. Difference in test macro-F1 from FedAvg on CSE-CIC-IDS2018 for nine strategies, with 95% confidence intervals over five paired seeds. Filled markers: Holm-adjusted p < 0.05 within the setting. Grey band: the ±0.02 equivalence margin.](fig2_forest.png){#fig:forest width="100%"}

Four patterns stand out on CSE-CIC-IDS2018. First, the two server optimisers gave the largest gains under IID and Dirichlet skew. FedAvgM beat FedAvg in every seed under the IID and Dirichlet $\alpha = 1.0$ partitions (+0.091 and +0.081, Holm $p = 0.033$ and $0.022$), and FedAdam did the same under IID data (+0.091, Holm $p = 0.041$). Their gains shrank as skew increased and were not significant under Dirichlet $\alpha = 0.5$ and $0.1$, where the seed-to-seed spread was large. Most of these gains come from the three classes with fewer than 100 records. Under IID data, FedAvgM and FedAdam reached a mean F1 of 0.38 and 0.37 on DoS-SlowHTTPTest, FTP-BruteForce and SQL Injection, against 0.01 for FedAvg, whereas on the other 12 classes their advantage was only +0.022 and +0.024 (FedAdam: 95% CI [+0.010, +0.039]); under Dirichlet skew it vanished (between −0.028 and +0.010, no confidence interval excluding zero). Since those three classes have seven to thirteen test records each and partly conflicting labels, this part of the gain is real but fragile. Second, the reweighting rules did not beat FedAvg anywhere. RAA-FDL and FedDisco stayed within 0.04 of FedAvg under IID and Dirichlet skew (RAA-FDL was equivalent to FedAvg within ±0.02 under Dirichlet $\alpha = 0.1$), and FLTrust was 0.10 to 0.25 below FedAvg in all five partitions. Removing FLTrust's norm rescaling recovered about half of that loss and shrinking its root set to 200 records made it worse in every partition except IID (0.517 against 0.448), so both the rescaling and the trust scores contribute to its deficit. Third, no client-side correction improved on FedAvg under tuned training: FedProx, SCAFFOLD, SCAFFOLD-N, logit adjustment and FedRS were within 0.02 of FedAvg under IID and Dirichlet $\alpha \ge 0.5$ (several equivalent within ±0.02), and SCAFFOLD, SCAFFOLD-N and SCAFFOLD-N + LA fell 0.05 to 0.08 below it under Dirichlet $\alpha = 0.1$. Adding reliability weights to SCAFFOLD-N + LA changed macro-F1 by at most 0.018 under IID and Dirichlet $\alpha \ge 0.5$. Fourth, under the per-day partition every strategy stayed between 0.105 and 0.298, and none differed significantly from FedAvg.

On the 12 learnable classes, all strategies except FLTrust and its variants scored between 0.69 and 0.89 under IID and Dirichlet skew, and every strategy stayed below 0.37 under the per-day partition (Appendix B).

Across all 95 comparisons in Table 5, six survive Holm correction within their setting and one survives correction across the whole table (FLTrust with a 200-record root set under IID, which is worse than FedAvg). Most differences between the stronger strategies are therefore within the noise of five seeds, and the equivalence tests confirm that several client-side corrections and reliability weights are practically indistinguishable from FedAvg in the milder settings.

**5G settings.** On the 5G datasets the dominant effect was not the mean but the risk of collapse, so Table 6 reports the median and the number of seeds in which a strategy collapsed. On the cleaned 5GAD, where one client holds every attack, FedAvg exceeded 0.82 in three seeds but predicted only the normal class in the other two (macro-F1 0.447). The server optimisers did not prevent this: FedAvgM collapsed in one seed and had a lower median than FedAvg, FedAdam collapsed in four, and FLTrust in all five, because its trust scores and norm rescaling suppressed the only update that carried attack information. Logit adjustment and RAA-FDL never collapsed (0.944 ± 0.010 and 0.940 ± 0.018), nor did either SCAFFOLD-N + LA variant. Applying logit adjustment only on clients that hold both classes (LA-multi) collapsed in all five seeds: the correction works on 5GAD-c because it stops the three attack-free clients from pushing the attack logit down, not because it rebalances the client that holds the attacks. With the selected $\mu = 0.001$, FedProx produced exactly the same predictions as FedAvg here.

On 5G-NIDD, where both base stations hold every class and differ only in proportions, most strategies reached 0.96 to 0.97. FedAvg and FedAdam each dropped to about 0.6 in one seed (seed 44), while FedProx, FedDisco, RAA-FDL, SCAFFOLD and FedAvgM stayed above 0.92 in every seed. Logit adjustment cost 0.06 to 0.07 in median macro-F1 here, which matches its behaviour on CSE-CIC-IDS2018 when every client holds every class. FLTrust was again the weakest strategy.

Table: Table 6. 5G datasets: median test macro-F1 over five seeds and number of seeds that collapsed (5GAD-c: macro-F1 below 0.5, which is the all-normal predictor's 0.447 plus a margin; 5G-NIDD: below 0.7).

| Strategy | 5GAD-c median | 5GAD-c collapsed | 5G-NIDD median | 5G-NIDD collapsed |
|:--|:--:|:--:|:--:|:--:|
{R:g5_robust_rows}

## Calibrating the classifier

Table 7 and Fig. 3b show the effect of CCVR applied to the final model of each strategy. Under the per-day partition it raised every strategy from 0.11 to 0.30 to between 0.60 and 0.70. For FedAvg the gain was +0.391 (95% CI [+0.336, +0.445], all five seeds, Holm $p = 0.003$), and FedAvgM followed by CCVR reached the highest per-day score, 0.701. The penultimate features learned under the per-day partition therefore separate the classes reasonably well; the failure lies in the final layer, whose class bias the server can correct from client-reported feature statistics. On 5GAD-c, CCVR rescued every collapsed run: all strategies reached 0.94 to 0.99 after calibration, including FLTrust (0.961) and FedAdam (0.983). Under IID data and Dirichlet $\alpha = 1.0$ and $0.5$, CCVR raised the other strategies by 0.004 to 0.044, and under Dirichlet $\alpha = 0.1$ by 0.021 to 0.103; FLTrust gained more (0.12 to 0.26). It lowered FedAvgM and FedAdam under IID data (by 0.030 and 0.041) and under Dirichlet $\alpha = 1.0$ (by 0.023 and 0.009), and changed them by less than 0.01 under $\alpha = 0.5$. On 5G-NIDD it lowered the strategies that were already stable (FedProx, RAA-FDL, FedDisco and SCAFFOLD) by 0.06 to 0.09, presumably because a classifier retrained on Gaussian virtual features loses precision when the trained classifier is already accurate. CCVR is thus a targeted fix for label-disjoint or single-holder settings, not a general improvement. It also has a cost that the other methods do not: each client reports a mean and covariance per class, which reveals which attack classes it has seen and summarises its traffic.

Table: Table 7. Test macro-F1 before → after server-side classifier calibration (CCVR) applied to each strategy's final model (mean over five seeds).

| Strategy | IID | Dir. 1.0 | Dir. 0.5 | Dir. 0.1 | Per-day | 5GAD-c | 5G-NIDD |
|:--|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
{R:ccvr_rows}

![Fig. 3. Per-day partition of CSE-CIC-IDS2018. (a) Validation macro-F1 over 300 rounds (mean ± SD over five seeds; rounds 101 to 300 continue the round-100 checkpoints of the main runs). (b) Test macro-F1 of the trained model and after server-side classifier calibration (CCVR).](fig3_perday.png){#fig:perday width="100%"}

## Operating point

Detection rates on CSE-CIC-IDS2018 were high for the strategies in Table 8 under IID and Dirichlet skew (0.94 to 0.98 across the four partitions; 0.95 to 0.98 for the IID and Dirichlet $\alpha = 0.5$ columns shown in Table 8), but benign false-positive rates were also high, between 0.29 and 0.64 (0.33 to 0.60 in the columns shown; all partitions are in the repository file operating.csv). Almost all of these false positives were benign flows classified as Infilteration: for FedAvg under IID data, 0.415 of the benign test flows were labelled Infilteration out of a false-positive rate of 0.461, and for FedAvgM 0.443 out of 0.458. The centralised network shows the same pattern (benign false-positive rate {central[cnn_fpr]}). This matches the {conf[benign_infil]:,} benign and Infilteration flows with identical features noted in Section 4.1, and the analysis of Liu et al. [@liu2022error]: in this dataset Infilteration is largely indistinguishable from benign traffic. Excluding the Infilteration confusion, FedAvg's benign false-positive rate under IID data is 0.046. At a benign-dominated prior (83% benign), macro-F1 falls to 0.44 to 0.71 under IID and Dirichlet skew (0.55 to 0.71 in the columns of Table 8), with FedAvgM and FedAdam still ahead. Operators who use this dataset family should therefore treat the Infilteration label with caution and should calibrate the decision threshold for the benign class on their own traffic. Under the per-day partition the operating point varies widely by strategy (benign false-positive rate 0.03 to 0.92, detection rate 0.63 to 1.00 across the twelve strategies analysed), and FedAvg still flagged 0.83 of attack flows as attacks while reaching a macro-F1 of only 0.27: the per-day failure is mainly confusion between attack classes rather than missed attacks.

Table: Table 8. Operating point on CSE-CIC-IDS2018 (mean over five seeds): benign false-positive rate (FPR), attack detection rate (DR) and macro-F1 at a benign share of 83%.

| Strategy | Setting | Benign FPR | DR | Macro-F1, 83% benign | After CCVR: FPR / DR / Macro-F1 |
|:--|:--|:--:|:--:|:--:|:--:|
{R:op_rows}

## Robustness of the conclusions

**Model and normalisation.** The per-day collapse is largely a property of the convolutional network. Replacing BatchNorm with GroupNorm raised FedAvg's per-day macro-F1 from 0.266 to 0.353 (+0.087, 95% CI [+0.044, +0.130], all five seeds) and left the Dirichlet $\alpha = 0.1$ result unchanged within noise (Table 9). A plain multilayer perceptron without normalisation layers did much better: FedAvg reached 0.804 under IID data, 0.784 under Dirichlet $\alpha = 0.1$ and 0.678 under the per-day partition, against 0.697, 0.655 and 0.266 for the network. With the MLP, the server optimisers again led once their server learning rates were re-tuned on validation data for this model (FedAvgM 0.3, FedAdam 0.01): FedAdam reached 0.821, 0.792 and 0.722, the best result in all three settings, while FedAvgM led under IID data and the per-day partition but was unstable under Dirichlet $\alpha = 0.1$ (0.638 ± 0.349). With the network's server settings, both diverged under skew, which shows that their gains depend on tuning. RAA-FDL and FedDisco stayed within 0.025 of FedAvg, logit adjustment was 0.03 to 0.08 below it, and SCAFFOLD diverged under Dirichlet $\alpha = 0.1$ and the per-day partition at the shared learning rate. CCVR on the MLP raised per-day macro-F1 only slightly (FedAvg 0.678 to 0.716) and lowered it under IID data, consistent with the classifier being less biased in the first place. The comparison between families therefore holds for both models, but the size of the per-day failure, and the benefit of calibration, depend on the model.

Table: Table 9. Model and normalisation (test macro-F1, mean ± SD over five seeds; the convolutional network with BatchNorm from Table 4 in brackets).

| Strategy | Setting | Result |
|:--|:--|:--|
{R:gn_mlp_rows}

**Training length.** {R:ext300 prose}

**Data protocol.** {R:chrono prose}

**Logit-adjustment settings.** On the per-day partition, scaling the offset by $\tau = 2$ cut FedAvg + LA's macro-F1 from 0.235 to 0.098, $\tau = 0.5$ gave 0.257, and changing the smoothing constant gave 0.180 ($\epsilon = 0.1$) and 0.276 ($\epsilon = 10$). Larger offsets for absent classes therefore hurt, and none of the settings beat FedAvg (0.266) by more than noise.

**Partial participation and number of clients.** When every client took part in each round with probability 0.5, FedAvg fell from 0.699 to 0.672 under Dirichlet $\alpha = 0.5$ and FedAvgM kept its lead (0.755). Under the per-day partition, however, FedAvgM collapsed to 0.032 with partial participation, against 0.174 for FedAvg and 0.207 for FedAvg + LA: server momentum amplifies the swings caused by label-disjoint clients dropping in and out. SCAFFOLD, whose control variates are known to suffer from partial participation, reached 0.130. With 20 and 50 clients under Dirichlet $\alpha = 0.5$, FedAvg fell to 0.680 and 0.629, logit adjustment and SCAFFOLD stayed within 0.015 of it, and FedAvgM kept a lead of about 0.08 to 0.12 (0.762 and 0.749).

## Poisoning

All results so far assume honest clients. To expose the trade-off between robustness and accuracy, we let 1 or 3 of the 10 clients relabel all their attack records as benign under Dirichlet $\alpha = 0.5$, which is what an attacker who wants traffic to pass unnoticed would do (Table 10). FedAvg lost 0.055 and 0.136 macro-F1, and its detection rate fell from 0.963 to 0.872 and 0.795, while its benign false-positive rate fell from 0.460 to 0.078 and 0.014, as expected when attack records are relabelled benign. FLTrust, designed for this threat, did not protect against it here: its macro-F1 fell further, to 0.476 and 0.274, because with heterogeneous honest clients its trust scores already discard much useful information. Logit adjustment was the most robust strategy (0.714 and 0.703, detection rate 0.957 and 0.931), presumably because the per-client prior offset limits how far a client whose labels are almost all benign can push the attack logits down. FedAvgM kept its macro-F1 (0.724 and 0.721) but lost detection rate (0.919 and 0.880). These results come from one attack type at one partition and do not show that any of these methods is a defence; they do show that operators cannot assume trust-based aggregation protects a heterogeneous federation, and that the choice of remedy interacts with the threat model.

Table: Table 10. Label-flipping attack under Dirichlet α = 0.5: test macro-F1 (mean ± SD), benign false-positive rate and detection rate (means over five seeds).

| Strategy | No attack | 1 of 10 clients | 3 of 10 clients | Benign FPR, 1 / 3 clients | DR, 1 / 3 clients |
|:--|:--:|:--:|:--:|:--:|:--:|
{R:poison_rows}

## Cost

Table 1 lists the extra communication of each strategy. The model has about 11,700 parameters, so one upload or download is about 46 KB; SCAFFOLD and its variants double this in both directions. CCVR adds a one-time upload of $C(d + d(d+1)/2) = 15 \times 2{,}144$ floats, about 126 KB per client on CSE-CIC-IDS2018, and FedDisco a one-time label histogram. In our simulation one round of FedAvg took 6.6 s on one CPU core on CSE-CIC-IDS2018, including the clients' training; FLTrust took 7.2 s because of the server's training epoch, and RAA-FDL 9.5 s because it evaluates every client model on the anchor set, which grows with the number of clients. FedAvgM, FedAdam, FedDisco, logit adjustment and FedRS added no measurable time.

# Discussion

**What the comparison shows.** No remedy won in every setting, which agrees with earlier benchmarks on other domains [@li2022niidbench]. The pattern is nonetheless clear enough to guide practice. When clients share most classes and differ in proportions (IID, Dirichlet skew and 5G-NIDD), most remedies tied with FedAvg on the well-populated classes. The server optimisers added a further gain, at no communication cost, by learning the rarest classes, but only when they were tuned, and server momentum became unstable under partial participation with label-disjoint clients. Reweighting rules did not help, and the trust-based rule hurt. When one client holds a class that the others lack (5GAD-c), stability rather than the mean decides: a prior correction on the attack-free clients, or a reliability rule that keeps the attack holder's weight, prevented the collapses that affected FedAvg and the server optimisers. When clients hold disjoint attack sets (the per-day partition), no training-time remedy helped, and only calibrating the classifier at the server, or a model whose classifier is less biased in the first place, recovered usable accuracy.

**Why tuning matters so much.** At the submitted learning rate of 0.01 every method was under-trained, so any change that increased the effective step size looked like an improvement. FedAvgM with momentum 0.9 multiplies the effective server step by up to ten, and logit adjustment changes the gradient scale of the frequent classes. Conclusions drawn at an untuned learning rate, including those of the submitted version of this paper, can therefore be artefacts of under-training. We recommend that federated intrusion detection studies report their tuning grid and select hyperparameters on validation data for every compared method.

**Deployment in a 5G core.** In the Release 18 architecture, an FL server NWDAF containing a Model Training Logical Function (MTLF) aggregates the updates of FL client NWDAFs that also contain an MTLF [@3gpp23288]. Server optimisers, reweighting rules and CCVR's calibration step all run in the FL server NWDAF and require no change at the clients beyond, for CCVR, the one-time upload of class-wise feature statistics. Logit adjustment and FedRS change only the clients' loss and need only local label counts. SCAFFOLD needs a control variate in each client MTLF and doubles the model traffic. Because label-disjoint sites are plausible in a 5G network, for example base stations or slices that see different attack campaigns, an operator can test for this condition cheaply: the label histograms FedDisco already collects, or a held-out validation set at the server, show whether calibration is needed.

**Security.** Recommending FedAvg-based aggregation for an intrusion detector is risky because FedAvg offers no protection against compromised sites. Our poisoning experiment shows that trust-based aggregation is not a safe default either in a heterogeneous federation: FLTrust lost more accuracy than FedAvg, with or without the attack. Operators facing a poisoning threat need defences designed and evaluated for heterogeneous clients, and should evaluate them on their own partition; our results do not identify one.

**Recommendations.** On this evidence, an operator should (1) tune the local learning rate and, if used, the server optimiser's learning rate on validation data; (2) when sites share most classes, expect little from any remedy on well-populated classes, and consider server momentum or server Adam to learn rare classes, checking their stability if participation is partial; (3) check whether some sites lack classes that others hold, and if so add a client-side prior correction or calibrate the classifier at the server; (4) avoid trust-based reweighting of honest heterogeneous clients; and (5) report false-positive rates at the deployment's own benign share, since macro-F1 on a balanced test set hides them.

# Limitations

First, CSE-CIC-IDS2018 is a capped and deduplicated sample with known label problems (Section 4.1); we report learnable-class macro-F1 and a run with conflicting records removed, but we did not relabel Infilteration. Second, all results use two small models; larger models or other optimisers could change the ranking. Third, five seeds detect large effects only; differences of 0.02 to 0.04 between the stronger strategies are not resolved, and we used equivalence tests rather than more seeds to support null results. Fourth, the 5G evidence comes from two datasets with few clients: the cleaned 5GAD has four co-located interfaces and a task that is easy when pooled, and 5G-NIDD has two base stations whose combined file had to be split by its row index. Fifth, server settings for the 5G datasets were taken from CSE-CIC-IDS2018 rather than tuned on each dataset, which may disadvantage the server optimisers there. Sixth, we did not evaluate MOON, FedDyn, FedLC, CReFF or FedBN, nor secure aggregation or differential privacy. Seventh, the poisoning experiment covers one attack at one partition. Finally, all experiments simulate federated training on recorded data; we did not run a live 5G core.

# Conclusion

We compared fourteen remedies for data heterogeneity, plus server-side calibration, in federated intrusion detection on enterprise traffic and two 5G datasets, with every method tuned on validation data and every comparison paired, corrected and complemented by equivalence tests. Tuning changed the conclusions of our own earlier analysis, and it is the first thing an operator should do. Once tuned, most remedies tied with FedAvg on well-populated classes when sites share classes, and server momentum and server Adam added gains mainly on the rarest classes; a client-side prior correction prevented collapse when one site held all attacks; and only classifier calibration at the server recovered accuracy when sites held disjoint attack sets. On enterprise traffic, reweighting by trust, reliability or label discrepancy did not improve on FedAvg; on the 5G datasets the reliability and label-discrepancy rules helped mainly by avoiding FedAvg's collapses, while standard FLTrust stayed the weakest method. Trust-based reweighting was not a defence against a stealthy label-flipping attack in a heterogeneous federation. The right remedy depends on how labels are spread across sites, which operators can measure before choosing.

# Appendix A. Tuning results {.unnumbered}

Table: Table A1. Validation macro-F1 after 100 rounds used for selection (mean over seeds 42 to 44). CSE-CIC-IDS2018: mean of the Dirichlet α = 0.5 and per-day partitions.

{R:tuning_tables}

# Appendix B. Alternative evaluation choices {.unnumbered}

{R:robust_eval prose and learnable-class table}


# Statements and Declarations {.unnumbered}

**Funding.** The author received no specific funding for this work.

**Competing interests.** The author has no relevant financial or non-financial interests to disclose.

**Author contributions.** A. G. Gbaden conceived the study, implemented the methods, ran the experiments, analysed the results and wrote the manuscript.

**Data availability.** CSE-CIC-IDS2018 [@sharafaldin2018] is available from the Canadian Institute for Cybersecurity (<https://www.unb.ca/cic/datasets/ids-2018.html>) and the AWS Open Data Registry (<https://registry.opendata.aws/cse-cic-ids2018/>). 5GAD [@coldwell2022] is available at <https://github.com/IdahoLabResearch/5GAD> under the MIT licence. 5G-NIDD [@samarakoon2022] is available from IEEE DataPort (<https://ieee-dataport.org/documents/5g-nidd-comprehensive-network-intrusion-detection-dataset-generated-over-5g-wireless>).

**Code availability.** The implementation of all strategies, the preprocessing scripts, the configuration of every run (including all tuning runs), the per-seed results, the per-round logs and the analysis scripts that produce every table and figure are available at <https://github.com/engrdrgamahlife/raa-fdl>.

**Ethics approval.** Not applicable. The study uses only public datasets and involves no human participants.

# References {.unnumbered}

::: {#refs}
:::
