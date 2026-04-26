# Lab 12: Introduction to GANs (MNIST)

**Course:** COMP-341L - Artificial Neural Networks Lab  
**Student:** Ali Hamza  
**Roll Number:** B23F0063AI106  
**Section:** B.S AI - Red  
**Date:** April 26, 2026

## Problem Context (Why GANs?)
In real-world domains like medical imaging, labeled data can be limited. GANs can generate realistic synthetic samples to augment training data and reduce overfitting.

## Task 1: Data Preparation
- Data source used: `torchvision`
- Normalization: scaled to `[-1, 1]` (matches `tanh` output)
- Real samples plot: `plots/task1_real_samples.png`

## Task 2: GAN Implementation
- Generator: DCGAN-style conv transpose network (noise → 28×28 image)
- Discriminator: conv classifier (image → probability real/fake)

## Task 3: Training
- Epochs: `15`
- Batch size: `128`
- Latent dim: `100`
- Optimizer: Adam (lr=0.0002, betas=(0.5,0.999))
- Loss: BCE
- Loss curves: `plots/task3_loss_curves.png`
- Loss CSV: `plots/task3_losses.csv`
- Final losses: D=0.8483, G=2.1725
- Generated samples per epoch: `samples/epoch_###.png`

## Task 4: Visualization
- Real vs Fake comparison: `plots/task4_real_vs_fake.png`

## Task 5: Experimentation
Suggested experiments:
- Change latent dimension: 100 → 50 → 200
- Change learning rate
- Add layers / feature maps
(Optional experiment runner included in notebook.)

## Lab Questions
1. **Why are GANs called “adversarial”?**  
   Generator vs Discriminator compete in a minimax game (fooling vs detecting).
2. **What happens if discriminator becomes too strong?**  
   Generator gradients become weak; training can stall.
3. **What is mode collapse?**  
   Generator outputs lack diversity; repeats a few patterns.
4. **Why do we use random noise as input?**  
   It enables controllable randomness and diverse outputs via mapping z→x.
5. **Difference between GAN and CNN?**  
   GAN is a generative training setup with 2 networks; CNN is an architecture often used inside either network.

## Notes on GAN Training Behavior
GAN training is unstable: losses may oscillate, and generated images can improve gradually. If samples look repetitive, it may indicate mode collapse.
