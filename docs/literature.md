# Literature reference — SAR flood segmentation

Reference notes supplied by the project owner, kept here so benchmark numbers can be
placed against published work instead of being read in isolation.

**Status of these figures: second-hand and unverified by this project.** They were
compiled from a literature search, not reproduced here. The owner's own caveat applies —
some entries (DAM-Net's venue metadata, and a very recent explainable-segmentation
preprint) are recent enough that details may shift, so pull the arXiv/GitHub sources
directly before implementing against any of them. Nothing below should be quoted as a
project result.

## Dataset

**Sen1Floods11** (Bonafilia et al. 2020, CVPRW) — 4,831 chips of 512×512 at 10 m,
across 11 distinct flood events, covering 120,406 km². Sentinel-1 SAR (VV/VH) paired
with aligned Sentinel-2 multispectral. **446 chips are hand-corrected by experts**; the
standard split is 60-20-20 train/val/test over that hand-labelled subset. This is the
dataset this project trains and benchmarks on.

Related: **S1GFloods** (companion/larger, used by DAM-Net) and **ETCI-2021** (NASA), both
competing benchmarks not currently used here.

## Assam / Brahmaputra coverage

Directly relevant to this project's motivation:

- A **Sen1Floods11 test scene is the severe Assam monsoon flooding of 12 August 2016**,
  used in published work to visualise SegFormer vs U-Net predictions near Majuli.
- **DeepSARFlood** (ViT ensemble) was demonstrated on the 2022 Pakistan floods and the
  2020 Assam floods, and validated against the **2022 Brahmaputra floods (10 Aug 2022)**
  using Sentinel-2 as reference labels.
- **Non-DL Assam baselines exist in the literature**: Otsu-threshold SAR mapping for
  Assam over July–September 2018–2020 validated against Sentinel-2 at ≥87% accuracy; a
  Random Forest / CART / SVM flood-hazard study combining weather, soil and terrain; and
  a classifier comparison on Nagaon district (Index Approach, EM clustering, K-means,
  Random Forest, Maximum Likelihood, GLCM) across VV/VH polarisations.

The Otsu and Random Forest baselines this project already implements therefore have
direct Assam-specific precedent, which is useful when justifying why they are included
rather than skipped.

## Reported results

| Approach | Setting | Reported |
|---|---|---|
| U-Net (CNN baseline) | Encoder-decoder on SAR | Widely adopted; skip connections preserve fine structures such as river channels |
| U-Net / U-Net++ / DeepLabV3(ResNet-34) vs SegFormer b0–b2 | Sen1Floods11, no TTA | SegFormer **IoU 0.583–0.595** vs CNN **0.516–0.570**. **On the flood class specifically: IoU 0.38–0.42, F1 0.55–0.59**, SegFormer-b2 highest at flood IoU **0.418**, only marginally ahead of the CNNs |
| DeepSARFlood (ViT + CNN-ViT hybrid ensemble) | Multitask, model soups, ensemble uncertainty | State-of-the-art **IoU 0.72** on Sen1Floods11 |
| DAM-Net (differential-attention Siamese ViT) | S1GFloods | 97.8% OA, 96.5% F1, **93.2% IoU** on S1GFloods (not Sen1Floods11 — different dataset, not comparable) |
| Prithvi (IBM-NASA geospatial foundation model) | Sentinel-2, 6 aligned bands | Compared against SegFormer and a from-scratch U-Net; strong transfer to unseen regions |
| Prithvi-CAFE (adapter fusion) | Sentinel-2 | **83.41 IoU water**, over Prithvi-600M 82.50, TerraMind 82.90, DOFA 81.54, DeepSAR 72.22, MM U-Net 73.84; beats baseline U-Net by 10.8 IoU on the held-out Bolivia site |
| GFM benchmark, 12 foundation models | Frozen-encoder GFMs vs plain U-Net, Sentinel-2 | **A conventional U-Net (mIoU 91.42) beat all 12 geo-foundation models**, with CROMA 90.89 and TerraMind 90.78 competitive; U-Net held its edge even at 10% and 50% data |
| CNN-LSTM fusion | Sentinel-1 flood fraction + MODIS time series, Bangladesh | Beats CNN-only; used to reconstruct 20 years of inundation extent |

## Why the metric definition matters when reading the table above

This project reports **two** IoU figures that differ by roughly 0.17 on identical
predictions (see `benchmarks/RESULTS.md`): pooled IoU, which sums confusion counts across
the whole test set, and per-chip mean IoU, which scores each chip and averages.

The published numbers above do not all state which convention they use, so **matching a
project number to a paper number requires knowing both**. Two anchors are worth keeping
in mind:

- The **0.72 SOTA** (DeepSARFlood) is the figure the spec treats as the pooled-style
  headline target. This project's primary U-Net sits at **0.676 pooled**.
- The **flood-class IoU 0.38–0.42** range, with SegFormer-b2 at 0.418, is strikingly
  close to this project's *per-chip* numbers (SegFormer-B2 measured at 0.4344 per-chip on
  the test split). Whether that range is per-chip or pooled in the source is **not
  established**, so the resemblance is suggestive, not a validated like-for-like
  comparison. Confirm the convention in the paper before making the claim anywhere.

## Implications for the build

- The **U-Net-beats-foundation-models** finding supports this project's ordering: a
  well-trained U-Net or DeepLabV3 baseline first, with SegFormer or a Prithvi variant as
  the comparison arm, rather than starting from a foundation model.
- A sensible target shape for the Assam work is: fine-tune on Sen1Floods11, then
  transfer/fine-tune to local Assam Sentinel-1/2 scenes — which matches this project's
  Track A / Track B data strategy.
- Architectures named as worth comparing: U-Net, U-Net++, DeepLabV3, SegFormer b0–b2
  (all already implemented here), plus DAM-Net-style Siamese ViT and Prithvi /
  Prithvi-CAFE fine-tuning as stretch work.
