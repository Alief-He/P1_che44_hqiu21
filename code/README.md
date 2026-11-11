# EN.520.665 Project 1: What Does Your Vision Model See?

Team: Chongxu He (che44@jh.edu), Hanyu Qiu (hqiu21@jh.edu)
Classes for Part B: bicycle, bottle, chair, car, cat. Third B3 factor: distance (near, far).

## What is in this ZIP

```
code/
  P1_A_*.ipynb        Part A: data split, models A1-A6, Table 1, learning curves (writes checkpoints/)
  P1_V_*.ipynb        V1-V5 visualizations
  P1_B_*.ipynb        Part B: B1 baseline, B3 series, B4 minimal pairs, B5 repair
  P1_665_*.ipynb      665 extension: frozen ViT features + linear head vs. fine-tuned A6-H
  p1_utils.py         course helper functions (data loading, photo loading, CSV writers)
  my_models.py        our model classes and preprocessing (A1Linear, A2Linear, A4ResNet, A5ViT, A6ViT)
  README.md           this file
photos/               all our photos, upright, longest side 1024 px
  B1/<class>/<object>_<n>.jpg
  B3/<class>/<object>/{baseline, occlusion_quarter, occlusion_half, distance_far}.jpg
  B4/                 the photos of the three minimal pairs (pairK_<class>_<object>_...)
  B5/adapt/<class>/<object>.jpg      10 adaptation photos (distance = far)
  B5/heldout/<class>/{ordinary, challenge}.jpg
b3_results.csv        one row per B3 image and model (A2, A4, A6-H, and A6-frozen for rotation)
b5_results.csv        held-out photos and B3 far photos, original vs. repaired A6-H
```

## How to run

All notebooks run on Google Colab (we used a Tesla T4) with a Drive folder `MyDrive/p1/` that holds
`p1_utils.py`, `my_models.py`, `data/`, `checkpoints/`, `results/`, and `photos/`.
Run them in this order, each with Runtime > Run all:

1. `P1_A` trains A1-A6 and saves the checkpoints and Table 1 (A6 downloads
   `vit_tiny_patch16_224.augreg_in21k_ft_in1k` from timm).
2. `P1_V` makes the V1-V5 figures from the checkpoints.
3. `P1_B` loads A2 (concatenated features), A4, and A6-H, runs B1, B3, B4, and the B5 repair
   (saves `a6_h_repaired.pt`), and writes the B-part CSVs and figures to `results/`.
4. `P1_665` extracts frozen class-token features, trains the linear head, and adds the
   `A6-frozen` rotation rows to `b3_results.csv`. Run it after `P1_B`.

Seeds: data split seed 0 (split by image), model heads seeded with `torch.manual_seed(1)`.

Software (from this session): python 3.13.18, torch 2.14.1+cpu, torchvision 0.29.1+cpu, timm 1.0.31, numpy 2.4.6, scikit-learn 1.6.1, pandas 2.2.3, matplotlib 3.10.0, pillow 11.3.0, pillow-heif 1.8.1, opencv-python 5.0.0.93

## Photos and objects

- B1: 52 photos, 10 per class except cat (12; two extra cat photos stayed in the folder when we added the
  third cat). Objects: bicycle lime, redbike1, redbike2; bottle fiji, poland, smart; car darkcar, redcar,
  whitesuv; cat browncat, siamese, tabby; chair pinkchair, whitechair, woodchair.
- B3: two objects per class: redbike1, redbike2; poland, fiji; woodchair, whitechair; darkcar, whitesuv;
  siamese, tabby. Rotations (30, 90, 180 degrees) are made in code from the baseline.
  Known issues: the quarter occlusion covers only about 10-15% of both chairs and redbike1, and the far photo of
  redbike1 has another bike in the foreground.
- B5 (repaired condition: distance = far):
  - Held-out objects (never used in the repair): bicycle = lime, bottle = smart, chair = pinkchair,
    car = redcar, cat = tabby. For each, `ordinary.jpg` is a normal photo and `challenge.jpg` a far photo
    taken at the same spot (for the tabby these are its B3 baseline and far photos).
  - Adaptation photos: we did not have time to take new objects, so these are the far photos of the B3 objects
    (redbike1, redbike2, poland, fiji, woodchair, whitechair, darkcar, whitesuv, siamese) plus a second far
    photo of the siamese cat (`siamese2.jpg`). This breaks the rule that adaptation objects are new; the
    report explains why the held-out row is our main evidence and the B3/B1 rows are not a fair test.

## Results files

- `b3_results.csv`: columns object, class, factor, condition, model, pred, correct.
- `b5_results.csv`: columns object, class, condition, model, pred, correct
  (models `A6-H` and `A6-H-repaired`).
