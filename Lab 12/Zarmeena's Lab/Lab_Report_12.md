# Lab 12 — GANs (MNIST Digits)

**Student:** Zarmeena Jawad  
**Roll No:** B23F0115AI125  
**Section:** B.S AI - Red  
**Date:** April 26, 2026

## Task 1 — Data Preparation
- Source used: `torchvision`
- Normalization: `[-1, 1]`
- Real grid: `outputs/plots/task1_real_grid.png`

## Task 2 — Model Design
- Vanilla GAN (MLP)
- Generator: `z -> 784` pixels, final activation `tanh`
- Discriminator: `784 -> logits`, loss `BCEWithLogitsLoss`

## Task 3 — Training
- Epochs: `15`
- Batch size: `128`
- Latent dim: `100`
- Adam lr: `0.0002` (betas `(0.5, 0.999)`)
- Final epoch losses: D=`1.3052`, G=`0.8467`
- Loss plot: `outputs/plots/task3_epoch_losses.png`
- Loss CSV: `outputs/loss_history.csv`
- Samples: `outputs/samples/epoch_###.png`

## Task 4 — Visualization
- Comparison: `outputs/plots/task4_real_vs_fake.png`

## Task 5 — Experimentation
Suggested:
- latent_dim: 50 / 100 / 200
- lr: 1e-4 / 2e-4 / 5e-4

## Lab Questions
1. Adversarial: G vs D game (fool vs detect).
2. Too-strong D: G gets weak gradients, can’t improve.
3. Mode collapse: low diversity outputs.
4. Noise input: provides randomness to generate variety.
5. GAN vs CNN: framework vs architecture.
