# Suspicious Photo Detection — Write-up

## The problem

Each outlet folder has photos from many visits, no timestamps, no labels. I need to flag the images that don't belong to that outlet. Folders are small (5 to 40 images), and clean folders must return zero flags. So this is per-folder unsupervised outlier detection on tiny samples.

## What I built, and how I got there

I started by measuring the data instead of assuming things about it. All 2,042 images turned out to be the same size (960x1280 JPEG, no EXIF), so preprocessing could stay minimal.

**First version: single embedding per image.** Frozen DINOv2 ViT-S/14, center crop at 224, distance to the folder's robust center. Why DINOv2? No labels means no training, so I need features that already answer "is this the same place?" out of the box. DINOv2's frozen features beat CLIP-style features by about 34% mAP on landmark retrieval, which is basically this task. The small variant (21M params, 384-d) runs fine on my CPU.

**Then I tested my own preprocessing choice.** I compared center-crop against squashing the full image to 224x224 (no crop) on real folders. Crop won on separation in most folders, but the bigger finding came later: the no-crop version over-flagged badly. Background clutter around the shop kept triggering flags on genuine photos.

**Then I hit a real failure and changed the design.** The big brand banner in the middle of these shop photos dominates a single global embedding. A fake shop with the same banner can look "close enough" to the real outlet, and the real outlet at a new angle can look far. So I embed each image three times: standard center crop, top half (roof and structure above the banner), bottom half (ground and counter below it), and concatenate them into one 1152-d descriptor. A fake location can fool one region. It can't fool all three at once.

**Scoring.** L2-normalize, then Euclidean distance to the folder's geometric median. I use the geometric median, not the mean, because it has a 50% breakdown point. The mean has zero: one fake image drags the center toward itself and hides. Fraud can't do that to the median. On top of the distances I use the MAD modified z-score (0.6745 x (d - median) / MAD, the Iglewicz-Hoaglin form) with an absolute cutoff. I rejected LOF and Isolation Forest early: LOF wants ~20 neighbors and many of my folders have 5 images. I also rejected any fixed contamination fraction, because it force-flags something in every folder and the spec requires clean folders to return empty. The output suspicion_score is a sigmoid of the z-score centered at the cutoff, so 0.5 sits exactly on the flag line. It's folder-relative, not comparable across outlets.

**Choosing the cutoff.** I ran the multiregion pipeline at four cutoffs (2.00, 2.35, 2.50, 2.75) plus the single-region and no-crop variants, all on the full dataset. Then I ran a structured visual verification over a sample of outlets: every flagged image plus reference photos was reviewed and cross-checked against every variant's verdict, and I spot-checked the conclusions. Results: multiregion made the fewest false positives (roughly 7-9 in the sample, vs ~14 for single-region and ~17 for no-crop). The clearest case was one outlet shot from six angles: multiregion flagged only the one genuinely fraudulent image, the other two variants flagged 5 and 6. On the cutoff: going 2.35 to 2.50 removed a verified false positive (same shop, an autorickshaw parked in front pushed the score to 2.49) and cost one borderline case. Going to 2.75 dropped two verified real fraud cases (different businesses, scores 2.52 and 2.66). So 2.75 is out. Clear fraud scores 4 and above; some real fraud sits at 2.5-2.7. I shipped 2.50. Honestly, 2.35 and 2.50 differ by only 2 flags, and I tuned on the same outlets I checked, so I'd call 2.35-2.50 the operating range and 2.50 my pick, not a proven optimum.

## Trade-offs

Multiregion costs 3 forward passes per image instead of 1. For fraud review, I'll take precision over speed. Median + MAD needs no training and every flag is explainable, but a supervised model with labels would catch subtler fakes. There are no labels, so that's for later.

## Scalability

Embedding is the only heavy step and I cache embeddings to disk, so changing the cutoff re-scores the whole dataset in seconds. Folders are independent, so this shards trivially across workers, and GPU batching gives roughly 10x. Scoring itself is O(N x D) per folder, negligible.

## Known limitations

Small folders (N of 5-6) make MAD noisy, so the detector goes conservative there and can miss a subtle fake. There's one shared blind spot I found and couldn't fix in time: in one outlet, three clearly wrong shops among the "_1" secondary images were missed by every variant, and "_1" images carry most of the confirmed fraud overall, so I'd route them to manual review and add an OCR check on shop-name text next. Quarterly promo banner changes on the same shop can score borderline; those go to review, not auto-reject. Many coordinated fakes in one folder could shift even the median. And with no ground truth, validation is visual verification of flagged cases plus cross-variant comparison, not ROC-AUC.