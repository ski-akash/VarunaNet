# Ratio-fix sweep — partial results (paused)

Retrain of the ablation sweep on the **fixed** `VV_VH_ratio` channel (`vv_vh_ratio =
vv_db - vh_db`, subtraction rather than division — see commit 5534b89). Every CNN number
in `cnn_results.md` except the three rescored primary U-Net seeds was trained on the
broken channel, so this sweep is what supersedes them.

**Status: paused after 8 of 22 jobs.** Stopped deliberately, not by failure — the
drip-feeder was killed and 12 scripts were never submitted. Nothing here is final.

## The numbers below are NOT comparable to `RESULTS.md` or `cnn_results.md`

These are **best per-chip mean IoU on the validation split**, read from each job's
training log. They are the early-stopping selection metric, nothing more.

The headline numbers everywhere else in this project are **pooled IoU on the test
split**. The two differ enormously — on this dataset the gap is roughly **+0.17 in
favour of pooled** (Otsu scores 0.304 per-chip and 0.479 pooled on the *same*
predictions), because pooled weights every pixel equally while per-chip lets one small
bad chip drag the mean down as hard as one large good chip lifts it.

Concretely, for the primary U-Net seed 1: per-chip 0.4198, pooled **0.6761** — same
checkpoint, same split. So a 0.41 in this table does not mean "worse than Otsu's 0.479".
Comparing a number from this file against a pooled number is a category error.

Real test-split figures require rescoring each `best.pt` through
`training/evaluate_checkpoints.py`, which has not been run for this sweep yet.

## Completed jobs

Checkpoints are at `~/job_results/<jobid>/checkpoints/best.pt` on the cluster (the
cluster's own epilog copies `$TMPDIR/results` there; the job scripts do not do it).

| Job | Model | State | Elapsed | Best val mean IoU | Epochs run |
|---|---|---|---|---|---|
| 2221 | SegFormer-B2 | COMPLETED | 00:11:32 | 0.4345 | 28 |
| 2218 | U-Net++ (ResNet-34) | COMPLETED | 00:11:25 | 0.4142 | 22 |
| 2222 | U-Net (ResNet-50) | COMPLETED | 00:10:17 | 0.4121 | 27 |
| 2220 | SegFormer-B0 | COMPLETED | 00:08:20 | 0.3977 | 19 |
| 2225 | ChangeAwareUNet seed 1 | COMPLETED | 00:09:05 | 0.3930 | 21 |
| 2223 | U-Net (ResNet-18) | COMPLETED | 00:07:47 | 0.3881 | 18 |
| 2224 | U-Net (MobileNetV3-Large) | COMPLETED | 00:07:54 | 0.3780 | 17 |
| 2219 | DeepLabV3+ (ResNet-34) | COMPLETED | 00:12:14 | 0.3776 | 30 |

Two more (2226 ChangeAwareUNet s2, 2227 ChangeAwareUNet s3) were still running when the
sweep was paused and should have completed; their logs are `2226.out` / `2227.out`.

**Caveat on DeepLabV3+:** it ran all 30 epochs without early-stopping, so it may simply
be under-trained rather than genuinely last. Don't read its position as a result.

## Not yet submitted (12)

Listed in `sweep_pending.txt` on the cluster, in this order:

```
train_speckle_looks1_ratio_fix_a100.sh
train_speckle_looks4_s1_ratio_fix_a100.sh
train_speckle_looks4_s2_ratio_fix_a100.sh
train_speckle_looks4_s3_ratio_fix_a100.sh
train_speckle_looks10_ratio_fix_a100.sh
train_no_hand_ratio_fix_a100.sh
train_no_slope_ratio_fix_a100.sh
train_focal_loss_ratio_fix_a100.sh
train_resnet18_tversky_a100.sh
train_resnet18_lovasz_a100.sh
train_densenet121_tversky_a100.sh
train_densenet121_lovasz_a100.sh
```

## How to resume

The `mtech` QOS caps this account at **2 jobs in the system at once**
(`MaxSubmitJobsPerUser=2`), so submitting the remainder in one go gets all but two
rejected with `QOSMaxSubmitJobPerUserLimit`. `drip_submit.sh` on the cluster polls and
feeds them in two at a time:

```
ssh csecluster
cd ~/VarunaNet && setsid nohup ./drip_submit.sh > drip_submit.log 2>&1 < /dev/null &
```

Runtimes are ~8–12 minutes per job, so the remaining 12 is roughly an hour of wall clock.

## Then

Rescore every checkpoint through the pooled/per-chip/OA/kappa path before any of it goes
into `cnn_results.md`:

```
python -m training.evaluate_checkpoints
```
