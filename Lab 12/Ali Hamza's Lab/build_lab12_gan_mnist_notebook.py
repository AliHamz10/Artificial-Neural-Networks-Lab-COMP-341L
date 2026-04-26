import json
from pathlib import Path
from textwrap import dedent


ROOT = Path(__file__).resolve().parent
NOTEBOOK_PATH = ROOT / "lab12_gan_mnist_colab.ipynb"


def lines(text: str):
    return dedent(text).lstrip("\n").splitlines(keepends=True)


def md_cell(text: str):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": lines(text),
    }


def code_cell(text: str):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": lines(text),
    }


cells = [
    md_cell(
        """
        # Lab 12: Introduction to Generative Adversarial Networks (GANs)

        **Course:** COMP-341L - Artificial Neural Networks Lab  
        **Student:** Ali Hamza  
        **Roll Number:** B23F0063AI106  
        **Section:** B.S AI - Red  
        **Execution Environment:** Google Colab

        ## Learning Objectives
        - Understand generative models and why they matter (data scarcity)
        - Explain GAN training (Generator vs Discriminator, adversarial game)
        - Implement a basic GAN using PyTorch
        - Train a GAN on MNIST handwritten digits
        - Visualize generated digits and analyze GAN behavior

        ## Lab Tasks (Summary)
        1. **Data Preparation:** Load MNIST-from-CSV dataset and normalize images  
        2. **GAN Implementation:** Build Generator + Discriminator (PyTorch)  
        3. **Training:** Train for ~10–20 epochs and observe losses  
        4. **Visualization:** Generate fake images and compare with real images  
        5. **Experimentation:** Change latent dimension / learning rate / layers

        ## Expected Output
        - Generated handwritten digits
        - Loss graphs (Generator vs Discriminator)
        - Visual comparison (Real vs Fake)
        """
    ),
    code_cell(
        """
        import os
        from datetime import datetime

        try:
            from google.colab import drive  # type: ignore
            IN_COLAB = True
        except Exception:
            IN_COLAB = False

        STUDENT_NAME = "Ali Hamza"
        STUDENT_ROLL = "B23F0063AI106"
        STUDENT_SECTION = "B.S AI - Red"
        STUDENT_FOLDER_NAME = "Ali Hamza's Lab"
        USE_GOOGLE_DRIVE = True

        if IN_COLAB:
            if not USE_GOOGLE_DRIVE:
                raise RuntimeError("Set USE_GOOGLE_DRIVE=True to save everything on Google Drive.")

            # Requirement: everything saved on Google Drive
            drive.mount("/content/drive", force_remount=True)
            BASE_DIR = f"/content/drive/MyDrive/COMP-341L/Lab 12/{STUDENT_FOLDER_NAME}"
            print("Google Drive mounted successfully.")
        else:
            BASE_DIR = os.environ.get("LAB12_BASE_DIR", ".")

        DATA_DIR = os.path.join(BASE_DIR, "data")
        PLOTS_DIR = os.path.join(BASE_DIR, "plots")
        SAMPLES_DIR = os.path.join(BASE_DIR, "samples")

        os.makedirs(BASE_DIR, exist_ok=True)
        os.makedirs(DATA_DIR, exist_ok=True)
        os.makedirs(PLOTS_DIR, exist_ok=True)
        os.makedirs(SAMPLES_DIR, exist_ok=True)

        print("IN_COLAB :", IN_COLAB)
        print("USE_GOOGLE_DRIVE:", USE_GOOGLE_DRIVE)
        print("BASE_DIR   :", os.path.abspath(BASE_DIR))
        print("DATA_DIR   :", os.path.abspath(DATA_DIR))
        print("PLOTS_DIR  :", os.path.abspath(PLOTS_DIR))
        print("SAMPLES_DIR:", os.path.abspath(SAMPLES_DIR))
        """
    ),
    md_cell(
        """
        ## Task 1: Data Preparation (MNIST-from-CSV)

        **Dataset (Kaggle):** MNIST in CSV  
        - `oddrationale/mnist-in-csv`

        **Important (Colab):** To download from Kaggle, upload your `kaggle.json` API token to the Colab session.
        This notebook will:
        1. Try Kaggle download (if token is available)
        2. Otherwise, fallback to `torchvision.datasets.MNIST` so the lab still runs end-to-end.
        """
    ),
    code_cell(
        """
        import math
        import os
        import random
        import shutil
        import subprocess
        import sys
        import zipfile
        from dataclasses import dataclass
        from pathlib import Path

        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        import seaborn as sns
        import torch
        import torch.nn as nn
        import torchvision
        import torchvision.transforms as T
        from IPython.display import display
        from torch.utils.data import DataLoader, Dataset
        from tqdm.auto import tqdm

        sns.set_style("whitegrid")

        SEED = 42
        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)

        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("DEVICE:", DEVICE)

        # Training hyperparameters (you can change these in Task 5)
        latent_dim = 100
        batch_size = 128
        lr = 2e-4
        beta1, beta2 = 0.5, 0.999
        num_epochs = 15  # (10–20 recommended)

        print(
            {
                "latent_dim": latent_dim,
                "batch_size": batch_size,
                "lr": lr,
                "betas": (beta1, beta2),
                "num_epochs": num_epochs,
            }
        )


        def _pip_install(pkg: str):
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "-q", pkg],
                check=True,
            )


        def _try_kaggle_download(dataset: str, out_dir: str):
            \"\"\"Download a Kaggle dataset into out_dir if kaggle.json is present.\"\"\"
            kaggle_json_candidates = [
                "/content/kaggle.json",  # common Colab upload location
                os.path.expanduser("~/.kaggle/kaggle.json"),
            ]
            kaggle_json_path = next(
                (p for p in kaggle_json_candidates if os.path.exists(p)), None
            )
            if kaggle_json_path is None:
                return False, "kaggle.json not found (upload it in Colab to enable Kaggle download)."

            _pip_install("kaggle")

            kaggle_dir = os.path.expanduser("~/.kaggle")
            os.makedirs(kaggle_dir, exist_ok=True)
            dst = os.path.join(kaggle_dir, "kaggle.json")
            if os.path.abspath(kaggle_json_path) != os.path.abspath(dst):
                shutil.copyfile(kaggle_json_path, dst)
            os.chmod(dst, 0o600)

            subprocess.run(
                [
                    "kaggle",
                    "datasets",
                    "download",
                    "-d",
                    dataset,
                    "-p",
                    out_dir,
                    "--unzip",
                ],
                check=True,
            )

            # If --unzip is unavailable or fails silently, try extracting any zip files we see.
            zip_files = list(Path(out_dir).glob("*.zip"))
            for zp in zip_files:
                try:
                    with zipfile.ZipFile(str(zp), "r") as zf:
                        zf.extractall(out_dir)
                except zipfile.BadZipFile:
                    pass

            return True, f"Downloaded Kaggle dataset: {dataset}"


        def _find_first_csv(search_dir: str):
            candidates = list(Path(search_dir).rglob("*.csv"))
            if not candidates:
                return None
            # prefer common MNIST CSV naming patterns
            preferred = []
            for p in candidates:
                name = p.name.lower()
                if "train" in name:
                    preferred.append((0, p))
                elif "mnist" in name:
                    preferred.append((1, p))
                else:
                    preferred.append((2, p))
            preferred.sort(key=lambda x: x[0])
            return str(preferred[0][1])


        def load_mnist_from_csv(csv_path: str):
            df = pd.read_csv(csv_path)
            if "label" in df.columns:
                labels = df["label"].to_numpy().astype("int64")
                pixels = df.drop(columns=["label"]).to_numpy().astype("float32")
            else:
                labels = df.iloc[:, 0].to_numpy().astype("int64")
                pixels = df.iloc[:, 1:].to_numpy().astype("float32")

            if pixels.shape[1] != 28 * 28:
                raise ValueError(
                    f"Expected 784 pixel columns, got {pixels.shape[1]} columns from {csv_path}"
                )

            images = pixels.reshape(-1, 28, 28)
            return images, labels


        class MNISTTensorDataset(Dataset):
            def __init__(self, images_01: np.ndarray):
                self.images_01 = images_01

            def __len__(self):
                return int(self.images_01.shape[0])

            def __getitem__(self, idx):
                x = self.images_01[idx]
                # x in [0,1] -> normalize to [-1, 1] for Tanh generator output
                x = (x - 0.5) / 0.5
                x = torch.from_numpy(x).float().unsqueeze(0)  # (1, 28, 28)
                return x


        DATA_SOURCE = "kaggle"
        kaggle_dataset = "oddrationale/mnist-in-csv"

        ok, msg = _try_kaggle_download(kaggle_dataset, DATA_DIR)
        print("Kaggle:", ok, "-", msg)
        if not ok:
            DATA_SOURCE = "torchvision"

        if DATA_SOURCE == "kaggle":
            csv_path = _find_first_csv(DATA_DIR)
            if csv_path is None:
                raise FileNotFoundError(f"No CSV found after Kaggle extraction in: {DATA_DIR}")
            images, labels = load_mnist_from_csv(csv_path)
            print("Loaded CSV:", csv_path)
            print("images:", images.shape, "labels:", labels.shape)
            images_01 = (images / 255.0).astype("float32")
            dataset = MNISTTensorDataset(images_01)
        else:
            print("Fallback: using torchvision MNIST (still valid for completing the lab).")
            tfm = T.Compose([T.ToTensor(), T.Normalize((0.5,), (0.5,))])  # -> [-1, 1]
            tv = torchvision.datasets.MNIST(root=DATA_DIR, train=True, download=True, transform=tfm)

            class _TorchvisionOnlyImages(Dataset):
                def __init__(self, ds):
                    self.ds = ds

                def __len__(self):
                    return len(self.ds)

                def __getitem__(self, idx):
                    x, _y = self.ds[idx]
                    return x

            dataset = _TorchvisionOnlyImages(tv)

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=torch.cuda.is_available(),
            drop_last=True,
        )

        # Visualize some real samples
        real_batch = next(iter(dataloader))
        grid = torchvision.utils.make_grid(real_batch[:32], nrow=8, normalize=True, value_range=(-1, 1))

        plt.figure(figsize=(8, 8))
        plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
        plt.title("Task 1: Real MNIST Samples")
        plt.axis("off")
        plt.tight_layout()
        path = os.path.join(PLOTS_DIR, "task1_real_samples.png")
        plt.savefig(path, dpi=160, bbox_inches="tight")
        plt.show()
        print("Saved:", path)
        """
    ),
    md_cell(
        """
        ## Task 2: GAN Implementation (Generator + Discriminator)

        We implement a **DCGAN-style** architecture for **28×28 grayscale** digits.

        - **Generator**: noise `z` → fake image (shape: `1×28×28`)
        - **Discriminator**: image → probability real/fake (scalar)
        """
    ),
    code_cell(
        """
        def weights_init(m):
            name = m.__class__.__name__
            if name.find("Conv") != -1:
                nn.init.normal_(m.weight.data, 0.0, 0.02)
            elif name.find("BatchNorm") != -1:
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0)


        class Generator(nn.Module):
            def __init__(self, z_dim: int):
                super().__init__()
                self.net = nn.Sequential(
                    # (N, z_dim, 1, 1) -> (N, 256, 7, 7)
                    nn.ConvTranspose2d(z_dim, 256, kernel_size=7, stride=1, padding=0, bias=False),
                    nn.BatchNorm2d(256),
                    nn.ReLU(True),
                    # 7 -> 14
                    nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(128),
                    nn.ReLU(True),
                    # 14 -> 28
                    nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(64),
                    nn.ReLU(True),
                    # refine channels, keep 28x28
                    nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1, bias=True),
                    nn.Tanh(),
                )

            def forward(self, z):
                return self.net(z)


        class Discriminator(nn.Module):
            def __init__(self):
                super().__init__()
                self.net = nn.Sequential(
                    # (N, 1, 28, 28) -> (N, 64, 14, 14)
                    nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1, bias=True),
                    nn.LeakyReLU(0.2, inplace=True),
                    # 14 -> 7
                    nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(128),
                    nn.LeakyReLU(0.2, inplace=True),
                    # 7 -> 4
                    nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(256),
                    nn.LeakyReLU(0.2, inplace=True),
                    # 4 -> 1
                    nn.Conv2d(256, 1, kernel_size=4, stride=1, padding=0, bias=True),
                    nn.Sigmoid(),
                )

            def forward(self, x):
                out = self.net(x)
                return out.view(-1, 1)


        G = Generator(latent_dim).to(DEVICE)
        D = Discriminator().to(DEVICE)
        G.apply(weights_init)
        D.apply(weights_init)

        def count_params(m):
            return sum(p.numel() for p in m.parameters() if p.requires_grad)

        print("Generator params:", count_params(G))
        print("Discriminator params:", count_params(D))
        """
    ),
    md_cell(
        """
        ## Task 3: Training (Adversarial Learning)

        Training loop (per batch):
        1. Train **Discriminator** on real images (label=1) and fake images (label=0)
        2. Train **Generator** to fool discriminator (wants label=1 for fake images)

        We track:
        - Discriminator loss `D_loss`
        - Generator loss `G_loss`
        """
    ),
    code_cell(
        """
        adversarial_loss = nn.BCELoss()

        opt_G = torch.optim.Adam(G.parameters(), lr=lr, betas=(beta1, beta2))
        opt_D = torch.optim.Adam(D.parameters(), lr=lr, betas=(beta1, beta2))

        fixed_noise = torch.randn(64, latent_dim, 1, 1, device=DEVICE)

        g_losses = []
        d_losses = []

        step = 0
        for epoch in range(1, num_epochs + 1):
            pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{num_epochs}")
            for real_images in pbar:
                real_images = real_images.to(DEVICE)
                bsz = real_images.size(0)

                # ---------------------
                #  Train Discriminator
                # ---------------------
                D.zero_grad(set_to_none=True)

                valid = torch.ones(bsz, 1, device=DEVICE)
                fake = torch.zeros(bsz, 1, device=DEVICE)

                real_pred = D(real_images)
                d_real = adversarial_loss(real_pred, valid)

                z = torch.randn(bsz, latent_dim, 1, 1, device=DEVICE)
                gen_images = G(z)
                fake_pred = D(gen_images.detach())
                d_fake = adversarial_loss(fake_pred, fake)

                d_loss = d_real + d_fake
                d_loss.backward()
                opt_D.step()

                # -----------------
                #  Train Generator
                # -----------------
                G.zero_grad(set_to_none=True)
                z = torch.randn(bsz, latent_dim, 1, 1, device=DEVICE)
                gen_images = G(z)
                g_loss = adversarial_loss(D(gen_images), valid)
                g_loss.backward()
                opt_G.step()

                g_losses.append(float(g_loss.item()))
                d_losses.append(float(d_loss.item()))

                if step % 50 == 0:
                    pbar.set_postfix({"D_loss": f"{d_loss.item():.4f}", "G_loss": f"{g_loss.item():.4f}"})

                step += 1

            # Save sample grid each epoch
            with torch.no_grad():
                fake_grid = G(fixed_noise).detach().cpu()
            grid = torchvision.utils.make_grid(
                fake_grid, nrow=8, normalize=True, value_range=(-1, 1)
            )
            plt.figure(figsize=(8, 8))
            plt.imshow(grid.permute(1, 2, 0).numpy())
            plt.title(f"Generated Digits — Epoch {epoch}")
            plt.axis("off")
            plt.tight_layout()
            sample_path = os.path.join(SAMPLES_DIR, f"epoch_{epoch:03d}.png")
            plt.savefig(sample_path, dpi=160, bbox_inches="tight")
            plt.show()
            print("Saved:", sample_path)

        # Plot loss curves
        plt.figure(figsize=(10, 4))
        plt.plot(d_losses, label="Discriminator Loss", linewidth=1)
        plt.plot(g_losses, label="Generator Loss", linewidth=1)
        plt.title("Task 3: GAN Losses (per iteration)")
        plt.xlabel("Iteration")
        plt.ylabel("BCE Loss")
        plt.legend()
        plt.tight_layout()
        loss_path = os.path.join(PLOTS_DIR, "task3_loss_curves.png")
        plt.savefig(loss_path, dpi=160, bbox_inches="tight")
        plt.show()
        print("Saved:", loss_path)

        # Save loss values to CSV for the report
        loss_csv_path = os.path.join(PLOTS_DIR, "task3_losses.csv")
        pd.DataFrame({"d_loss": d_losses, "g_loss": g_losses}).to_csv(loss_csv_path, index=False)
        print("Saved:", loss_csv_path)
        """
    ),
    md_cell(
        """
        ## Task 4: Visualization (Real vs Fake)

        We compare:
        - A grid of **real** MNIST digits (from dataset)
        - A grid of **fake** digits (from generator)
        """
    ),
    code_cell(
        """
        # Prepare a real grid (one batch)
        real_batch = next(iter(dataloader))
        real_grid = torchvision.utils.make_grid(
            real_batch[:64], nrow=8, normalize=True, value_range=(-1, 1)
        ).cpu()

        with torch.no_grad():
            fake_batch = G(fixed_noise).detach().cpu()
        fake_grid = torchvision.utils.make_grid(
            fake_batch, nrow=8, normalize=True, value_range=(-1, 1)
        )

        fig, axes = plt.subplots(1, 2, figsize=(14, 7))
        axes[0].imshow(real_grid.permute(1, 2, 0).numpy())
        axes[0].set_title("Real MNIST")
        axes[0].axis("off")
        axes[1].imshow(fake_grid.permute(1, 2, 0).numpy())
        axes[1].set_title("Fake (Generated)")
        axes[1].axis("off")
        plt.tight_layout()
        path = os.path.join(PLOTS_DIR, "task4_real_vs_fake.png")
        plt.savefig(path, dpi=160, bbox_inches="tight")
        plt.show()
        print("Saved:", path)
        """
    ),
    md_cell(
        """
        ## Task 5: Experimentation (Optional)

        Try changing:
        - **latent_dim**: 100 → 50 → 200
        - **learning rate**: e.g., `2e-4` → `1e-4` or `5e-4`
        - Add more layers / feature maps

        The cell below provides a small experiment runner. By default it is **OFF** to avoid long runtimes.
        """
    ),
    code_cell(
        """
        RUN_EXPERIMENTS = False  # set True if you want to run quick comparisons


        @dataclass
        class ExperimentConfig:
            name: str
            latent_dim: int
            lr: float
            epochs: int = 3
            max_batches_per_epoch: int = 200


        def train_quick(config: ExperimentConfig):
            g = Generator(config.latent_dim).to(DEVICE).apply(weights_init)
            d = Discriminator().to(DEVICE).apply(weights_init)
            og = torch.optim.Adam(g.parameters(), lr=config.lr, betas=(beta1, beta2))
            od = torch.optim.Adam(d.parameters(), lr=config.lr, betas=(beta1, beta2))

            g_loss_hist = []
            d_loss_hist = []

            for _epoch in range(config.epochs):
                for bi, real_images in enumerate(dataloader):
                    if bi >= config.max_batches_per_epoch:
                        break
                    real_images = real_images.to(DEVICE)
                    bsz = real_images.size(0)
                    valid = torch.ones(bsz, 1, device=DEVICE)
                    fake = torch.zeros(bsz, 1, device=DEVICE)

                    # D
                    d.zero_grad(set_to_none=True)
                    d_real = adversarial_loss(d(real_images), valid)
                    z = torch.randn(bsz, config.latent_dim, 1, 1, device=DEVICE)
                    gen = g(z)
                    d_fake = adversarial_loss(d(gen.detach()), fake)
                    d_loss = d_real + d_fake
                    d_loss.backward()
                    od.step()

                    # G
                    g.zero_grad(set_to_none=True)
                    z = torch.randn(bsz, config.latent_dim, 1, 1, device=DEVICE)
                    gen = g(z)
                    g_loss = adversarial_loss(d(gen), valid)
                    g_loss.backward()
                    og.step()

                    g_loss_hist.append(float(g_loss.item()))
                    d_loss_hist.append(float(d_loss.item()))

            return {
                "name": config.name,
                "latent_dim": config.latent_dim,
                "lr": config.lr,
                "g_loss_last": float(g_loss_hist[-1]) if g_loss_hist else float("nan"),
                "d_loss_last": float(d_loss_hist[-1]) if d_loss_hist else float("nan"),
            }


        if RUN_EXPERIMENTS:
            configs = [
                ExperimentConfig(name="z100_lr2e-4", latent_dim=100, lr=2e-4),
                ExperimentConfig(name="z50_lr2e-4", latent_dim=50, lr=2e-4),
                ExperimentConfig(name="z200_lr2e-4", latent_dim=200, lr=2e-4),
                ExperimentConfig(name="z100_lr1e-4", latent_dim=100, lr=1e-4),
            ]

            rows = []
            for cfg in configs:
                print("Running:", cfg)
                rows.append(train_quick(cfg))

            exp_df = pd.DataFrame(rows).sort_values(["latent_dim", "lr"])
            display(exp_df)

            exp_path = os.path.join(PLOTS_DIR, "task5_experiments.csv")
            exp_df.to_csv(exp_path, index=False)
            print("Saved:", exp_path)
        """
    ),
    md_cell(
        """
        ## Lab Questions (Answer in Notebook)

        1. **Why are GANs called “adversarial”?**  
           Because the generator and discriminator play a competitive (minimax) game: the generator tries to fool the discriminator, while the discriminator tries to detect fake samples.

        2. **What happens if the discriminator becomes too strong?**  
           If the discriminator classifies real/fake perfectly, the generator receives near-zero gradients (no useful learning signal), so generator training may stall.

        3. **What is mode collapse?**  
           When the generator produces a limited variety of outputs (e.g., only a few digit styles) and ignores other modes of the data distribution.

        4. **Why do we use random noise as input?**  
           Noise provides a simple source of randomness; the generator learns to map different noise vectors to diverse outputs, covering the data distribution.

        5. **Difference between GAN and CNN?**  
           A CNN is a neural network architecture often used for feature extraction/classification. A GAN is a training framework with two networks (often CNN-based) that learns to generate new data rather than only classify.
        """
    ),
    code_cell(
        """
        # Export Lab Report (Markdown + HTML)

        final_g_loss = float(g_losses[-1]) if g_losses else float("nan")
        final_d_loss = float(d_losses[-1]) if d_losses else float("nan")
        iters = int(len(d_losses))
        d_min = float(np.min(d_losses)) if d_losses else float("nan")
        g_min = float(np.min(g_losses)) if g_losses else float("nan")

        # Best-effort: include the latest saved sample image from SAMPLES_DIR
        try:
            sample_files = sorted([f for f in os.listdir(SAMPLES_DIR) if f.lower().endswith(".png")])
        except Exception:
            sample_files = []
        last_sample = sample_files[-1] if sample_files else None
        mid_sample = sample_files[len(sample_files) // 2] if sample_files else None
        first_sample = sample_files[0] if sample_files else None

        def _safe_int(x, default=0):
            try:
                return int(x)
            except Exception:
                return default

        dataset_size = _safe_int(len(dataset), default=0)
        g_params = sum(p.numel() for p in G.parameters() if p.requires_grad)
        d_params = sum(p.numel() for p in D.parameters() if p.requires_grad)

        report_md = f\"\"\"# Lab 12: Introduction to GANs (MNIST)

        **Course:** COMP-341L - Artificial Neural Networks Lab  
        **Student:** {STUDENT_NAME}  
        **Roll Number:** {STUDENT_ROLL}  
        **Section:** {STUDENT_SECTION}  
        **Date:** {datetime.now().strftime('%B %d, %Y')}

        ## Problem Context (Why GANs?)
        In real-world domains like medical imaging, labeled data can be limited. GANs can generate realistic synthetic samples to augment training data and reduce overfitting.

        ## Task 1: Data Preparation
        - Data source used: `{DATA_SOURCE}`
        - Normalization: scaled to `[-1, 1]` (matches `tanh` output)
        - Real samples plot: `plots/task1_real_samples.png`

        ## Task 2: GAN Implementation
        - Generator: DCGAN-style conv transpose network (noise → 28×28 image)
        - Discriminator: conv classifier (image → probability real/fake)

        ## Task 3: Training
        - Epochs: `{num_epochs}`
        - Batch size: `{batch_size}`
        - Latent dim: `{latent_dim}`
        - Optimizer: Adam (lr={lr}, betas=({beta1},{beta2}))
        - Loss: BCE
        - Loss curves: `plots/task3_loss_curves.png`
        - Loss CSV: `plots/task3_losses.csv`
        - Final losses: D={final_d_loss:.4f}, G={final_g_loss:.4f}
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
        \"\"\"

        html = f\"\"\"<!doctype html>
        <html lang="en">
          <head>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <title>COMP-341L — Lab 12 Report — {STUDENT_NAME}</title>
            <style>
              :root {{
                --bg: #0b1220;
                --surface: rgba(255,255,255,0.03);
                --surface-2: rgba(255,255,255,0.05);
                --border: rgba(255,255,255,0.12);
                --text: #e5e7eb;
                --muted: #c7d2fe;
                --muted-2: #94a3b8;
                --accent: #a78bfa;
                --accent-2: #22c55e;
                --danger: #fb7185;
              }}
              * {{ box-sizing: border-box; }}
              html, body {{ background: var(--bg); color: var(--text); }}
              body {{
                margin: 0;
                font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial;
                line-height: 1.62;
                background:
                  radial-gradient(1000px 600px at 15% 0%, rgba(167,139,250,0.22), transparent 55%),
                  radial-gradient(900px 500px at 85% 10%, rgba(34,197,94,0.15), transparent 60%),
                  var(--bg);
              }}
              .page {{ max-width: 1040px; margin: 0 auto; padding: 34px 20px 70px; }}
              .hero {{
                border: 1px solid var(--border);
                border-radius: 18px;
                padding: 26px 22px;
                background: linear-gradient(180deg, rgba(255,255,255,0.05), rgba(255,255,255,0.02));
                backdrop-filter: blur(6px);
              }}
              .kicker {{
                font-weight: 800;
                letter-spacing: 0.12em;
                text-transform: uppercase;
                color: var(--muted);
                font-size: 12px;
                margin: 0 0 10px;
              }}
              h1 {{ margin: 0 0 8px; font-size: 30px; line-height: 1.2; }}
              .subtitle {{ margin: 0 0 14px; color: var(--muted-2); font-size: 15px; }}
              .meta-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-top: 12px; }}
              .meta-card {{
                background: var(--surface);
                border: 1px solid var(--border);
                border-radius: 14px;
                padding: 12px 14px;
              }}
              .meta-label {{ margin: 0; font-size: 12px; color: var(--muted); }}
              .meta-value {{ margin: 2px 0 0; font-weight: 700; }}
              .chip {{
                display: inline-block;
                padding: 6px 10px;
                border-radius: 999px;
                border: 1px solid var(--border);
                background: var(--surface);
                color: var(--muted);
                font-size: 12px;
                margin-right: 8px;
              }}
              .row {{ display: grid; grid-template-columns: 1.15fr 0.85fr; gap: 14px; margin-top: 14px; }}
              @media (max-width: 900px) {{
                .row {{ grid-template-columns: 1fr; }}
                .meta-grid {{ grid-template-columns: 1fr; }}
              }}
              .card {{
                border: 1px solid var(--border);
                border-radius: 16px;
                padding: 14px 14px;
                background: var(--surface);
              }}
              h2 {{ font-size: 20px; margin: 24px 0 10px; }}
              h3 {{ font-size: 15px; margin: 14px 0 8px; color: var(--muted); }}
              .callout {{
                border: 1px solid var(--border);
                background: linear-gradient(180deg, rgba(167,139,250,0.12), rgba(255,255,255,0.03));
                padding: 12px 12px;
                border-radius: 14px;
                margin: 12px 0;
              }}
              .callout strong {{ color: var(--text); }}
              .figure {{
                border: 1px solid var(--border);
                border-radius: 16px;
                padding: 12px;
                background: var(--surface-2);
                margin: 12px 0;
              }}
              .figure img {{ max-width: 100%; border-radius: 10px; display: block; margin: 0 auto; }}
              .figcap {{ color: var(--muted-2); font-size: 13px; margin-top: 8px; text-align: center; }}
              ul, ol {{ margin: 8px 0 12px 20px; }}
              code {{
                background: rgba(255,255,255,0.08);
                padding: 2px 6px;
                border-radius: 8px;
                border: 1px solid rgba(255,255,255,0.10);
              }}
              pre {{
                background: rgba(255,255,255,0.06);
                border: 1px solid var(--border);
                padding: 12px;
                border-radius: 14px;
                overflow-x: auto;
              }}
              a {{ color: var(--accent); text-decoration: none; }}
              a:hover {{ text-decoration: underline; }}
              .toc a {{ color: var(--muted); }}
              .toc {{
                border: 1px solid var(--border);
                border-radius: 16px;
                padding: 12px 14px;
                background: var(--surface);
                margin-top: 14px;
              }}
              .grid-2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }}
              @media (max-width: 900px) {{
                .grid-2 {{ grid-template-columns: 1fr; }}
              }}
              table {{
                width: 100%;
                border-collapse: collapse;
                overflow: hidden;
                border-radius: 14px;
                border: 1px solid var(--border);
                background: rgba(255,255,255,0.02);
              }}
              th, td {{
                padding: 10px 10px;
                border-bottom: 1px solid rgba(255,255,255,0.08);
                font-size: 13px;
                vertical-align: top;
              }}
              th {{ text-align: left; color: var(--muted); font-weight: 800; }}
              .footer {{
                margin-top: 22px;
                color: var(--muted-2);
                font-size: 12px;
                border-top: 1px solid rgba(255,255,255,0.10);
                padding-top: 12px;
              }}
            </style>
          </head>
          <body>
            <div class="page">
              <div class="hero" id="top">
                <p class="kicker">COMP-341L • Artificial Neural Networks Lab</p>
                <h1>Lab 12 — Generative Adversarial Networks (GANs) for Synthetic Digits</h1>
                <p class="subtitle">
                  Academic Lab Report • PyTorch Implementation • MNIST Handwritten Digits
                </p>
                <div>
                  <span class="chip">Generator vs Discriminator</span>
                  <span class="chip">Normalization: [-1, 1]</span>
                  <span class="chip">Device: {DEVICE}</span>
                </div>
                <div class="meta-grid">
                  <div class="meta-card"><p class="meta-label">Student</p><p class="meta-value">{STUDENT_NAME}</p></div>
                  <div class="meta-card"><p class="meta-label">Roll Number</p><p class="meta-value">{STUDENT_ROLL}</p></div>
                  <div class="meta-card"><p class="meta-label">Section</p><p class="meta-value">{STUDENT_SECTION}</p></div>
                  <div class="meta-card"><p class="meta-label">Date</p><p class="meta-value">{datetime.now().strftime('%B %d, %Y')}</p></div>
                </div>

                <div class="toc">
                  <strong style="color: var(--text);">Contents</strong>
                  <ul class="toc">
                    <li><a href="#abstract">Abstract</a></li>
                    <li><a href="#objectives">Objectives</a></li>
                    <li><a href="#dataset">Dataset & Preprocessing</a></li>
                    <li><a href="#theory">GAN Theory (Loss Functions)</a></li>
                    <li><a href="#method">Methodology (Architecture)</a></li>
                    <li><a href="#training">Training Setup</a></li>
                    <li><a href="#results">Results & Visualizations</a></li>
                    <li><a href="#discussion">Discussion</a></li>
                    <li><a href="#questions">Lab Questions</a></li>
                    <li><a href="#conclusion">Conclusion</a></li>
                  </ul>
                </div>
              </div>

              <h2 id="abstract">Abstract</h2>
              <div class="card">
                This lab implements and trains a Generative Adversarial Network (GAN) to generate synthetic handwritten digits.
                The generator learns to transform random noise vectors into 28×28 grayscale digit images, while the discriminator learns
                to classify images as real or fake. The trained GAN demonstrates adversarial learning behavior through oscillating losses and
                progressively improved generated samples.
              </div>

              <h2 id="objectives">Objectives</h2>
              <div class="card">
                <ul>
                  <li>Understand the concept of generative models and why data generation can help in data-scarce domains.</li>
                  <li>Implement a basic GAN in PyTorch using a generator and discriminator trained in a minimax game.</li>
                  <li>Train a GAN on MNIST and analyze losses and visual outputs over epochs.</li>
                </ul>
              </div>

              <h2 id="dataset">Dataset & Preprocessing</h2>
              <div class="row">
                <div class="card">
                  <h3>Dataset</h3>
                  <ul>
                    <li>Source used: <code>{DATA_SOURCE}</code></li>
                    <li>Samples used (train split): <code>{dataset_size}</code></li>
                    <li>Image shape: <code>1×28×28</code> grayscale</li>
                  </ul>
                  <h3>Normalization</h3>
                  <div class="callout">
                    Pixel values are scaled to <code>[-1, 1]</code> to match the generator’s <code>tanh</code> output range,
                    improving stability during training.
                  </div>
                </div>
                <div class="figure">
                  <img src="plots/task1_real_samples.png" alt="Real MNIST samples">
                  <div class="figcap">Figure 1: Real MNIST samples (normalized for display).</div>
                </div>
              </div>

              <h2 id="theory">GAN Theory (Loss Functions)</h2>
              <div class="card">
                <p>
                  A GAN trains two neural networks simultaneously:
                  the <strong>Generator (G)</strong> and the <strong>Discriminator (D)</strong>.
                  The discriminator attempts to maximize correct classification, while the generator attempts to fool the discriminator.
                </p>
                <div class="grid-2">
                  <div class="card" style="background: rgba(167,139,250,0.08);">
                    <h3>Discriminator objective</h3>
                    <p style="margin: 0;">
                      Maximize: <code>log D(x) + log(1 − D(G(z)))</code><br>
                      Learns to assign high probability to real samples and low probability to generated samples.
                    </p>
                  </div>
                  <div class="card" style="background: rgba(34,197,94,0.08);">
                    <h3>Generator objective</h3>
                    <p style="margin: 0;">
                      Minimize: <code>−log D(G(z))</code><br>
                      Learns to generate samples that the discriminator labels as real.
                    </p>
                  </div>
                </div>
                <p style="color: var(--muted-2); margin-top: 10px;">
                  In this implementation, Binary Cross-Entropy (BCE) loss is used to train both networks.
                </p>
              </div>

              <h2 id="method">Methodology (Architecture)</h2>
              <div class="card">
                <div class="callout">
                  <strong>Design choice:</strong> A DCGAN-style convolutional generator/discriminator is used for better spatial learning on images.
                </div>
                <table>
                  <tr><th>Component</th><th>Input</th><th>Output</th><th>Notes</th></tr>
                  <tr>
                    <td><strong>Generator (G)</strong></td>
                    <td><code>z ∈ R^{latent_dim}</code></td>
                    <td><code>1×28×28</code></td>
                    <td>ConvTranspose + BatchNorm + ReLU, final <code>tanh</code></td>
                  </tr>
                  <tr>
                    <td><strong>Discriminator (D)</strong></td>
                    <td><code>1×28×28</code></td>
                    <td><code>p(real)</code></td>
                    <td>Conv + BatchNorm + LeakyReLU, final <code>sigmoid</code></td>
                  </tr>
                </table>
                <p style="color: var(--muted-2); margin-top: 10px;">
                  Trainable parameters — G: <code>{g_params}</code>, D: <code>{d_params}</code>.
                </p>
              </div>

              <h2 id="training">Training Setup</h2>
              <div class="card">
                <table>
                  <tr><th>Hyperparameter</th><th>Value</th></tr>
                  <tr><td>Epochs</td><td><code>{num_epochs}</code></td></tr>
                  <tr><td>Batch size</td><td><code>{batch_size}</code></td></tr>
                  <tr><td>Latent dimension</td><td><code>{latent_dim}</code></td></tr>
                  <tr><td>Optimizer</td><td><code>Adam</code></td></tr>
                  <tr><td>Learning rate</td><td><code>{lr}</code></td></tr>
                  <tr><td>Betas</td><td><code>({beta1}, {beta2})</code></td></tr>
                  <tr><td>Loss</td><td><code>BCE</code></td></tr>
                </table>
                <div class="callout">
                  <strong>Loss summary (iterations={iters}):</strong>
                  Final D=<code>{final_d_loss:.4f}</code>, Final G=<code>{final_g_loss:.4f}</code> •
                  Min D=<code>{d_min:.4f}</code>, Min G=<code>{g_min:.4f}</code>
                </div>
              </div>

              <h2 id="results">Results & Visualizations</h2>
              <div class="grid-2">
                <div class="figure">
                  <img src="plots/task3_loss_curves.png" alt="GAN loss curves">
                  <div class="figcap">Figure 2: Discriminator vs Generator loss across iterations (training behavior may oscillate).</div>
                </div>
                <div class="figure">
                  <img src="plots/task4_real_vs_fake.png" alt="Real vs Fake">
                  <div class="figcap">Figure 3: Real digits vs generated digits after training.</div>
                </div>
              </div>

              <div class="card">
                <h3>Generated samples over epochs</h3>
                <p style="color: var(--muted-2); margin-top: 0;">
                  The notebook saves one image grid per epoch in <code>samples/</code>. The report highlights three checkpoints when available.
                </p>
                <div class="grid-2">
                  {f'<div class=\"figure\"><img src=\"samples/{first_sample}\" alt=\"Epoch samples (start)\"><div class=\"figcap\">Figure 4: Early epoch samples ({first_sample}).</div></div>' if first_sample else '<div class=\"card\">No sample images found in <code>samples/</code>.</div>'}
                  {f'<div class=\"figure\"><img src=\"samples/{mid_sample}\" alt=\"Epoch samples (mid)\"><div class=\"figcap\">Figure 5: Mid-training samples ({mid_sample}).</div></div>' if mid_sample else '<div class=\"card\">Mid checkpoint not available.</div>'}
                  {f'<div class=\"figure\"><img src=\"samples/{last_sample}\" alt=\"Epoch samples (final)\"><div class=\"figcap\">Figure 6: Final epoch samples ({last_sample}).</div></div>' if last_sample else '<div class=\"card\">Final checkpoint not available.</div>'}
                  <div class="card">
                    <h3>Files saved</h3>
                    <ul>
                      <li><code>plots/task1_real_samples.png</code></li>
                      <li><code>plots/task3_loss_curves.png</code></li>
                      <li><code>plots/task4_real_vs_fake.png</code></li>
                      <li><code>samples/epoch_###.png</code></li>
                    </ul>
                  </div>
                </div>
              </div>

              <h2 id="discussion">Discussion</h2>
              <div class="card">
                <ul>
                  <li><strong>Adversarial dynamics:</strong> Unlike standard supervised learning, GAN losses can oscillate; visual quality is often a better indicator than loss alone.</li>
                  <li><strong>Too-strong discriminator:</strong> If D becomes near-perfect early, G gradients weaken and learning slows. Balancing model capacity and learning rates helps.</li>
                  <li><strong>Mode collapse:</strong> If generated digits become repetitive, it indicates low diversity. Common fixes include architecture changes, modified losses, or regularization.</li>
                  <li><strong>Practical note:</strong> More epochs typically improve sample quality, but overtraining can destabilize if one player dominates.</li>
                </ul>
              </div>

              <h2 id="questions">Lab Questions</h2>
              <div class="card">
                <ol>
                  <li><strong>Why are GANs called adversarial?</strong> Because G and D have opposing objectives: G tries to fool D, D tries to detect fakes.</li>
                  <li><strong>What happens if the discriminator becomes too strong?</strong> The generator receives weak gradients and may stop improving.</li>
                  <li><strong>What is mode collapse?</strong> The generator produces low-diversity outputs, repeating a few patterns.</li>
                  <li><strong>Why random noise as input?</strong> Noise provides controllable randomness; different z vectors map to different outputs.</li>
                  <li><strong>Difference between GAN and CNN?</strong> GAN is a two-network generative framework; CNN is a neural architecture commonly used inside GAN components.</li>
                </ol>
              </div>

              <h2 id="conclusion">Conclusion</h2>
              <div class="card">
                This lab demonstrates how GANs can learn a data distribution and generate new samples that resemble real handwritten digits.
                The trained generator produces synthetic digits from noise, which can support data augmentation in scenarios where real labeled data is limited.
                Future work can include conditional GANs (class-guided digits) and improved training objectives to reduce instability and mode collapse.
              </div>

              <h2 id="references">References</h2>
              <div class="card">
                <ul>
                  <li>Goodfellow, I., Pouget-Abadie, J., Mirza, M., Xu, B., Warde-Farley, D., Ozair, S., Courville, A., & Bengio, Y. (2014). <em>Generative Adversarial Nets</em>. NeurIPS.</li>
                  <li>LeCun, Y. (quoted). GANs described as a highly influential idea in modern machine learning (commonly cited in GAN introductions).</li>
                </ul>
              </div>

              <div class="footer">
                Generated by notebook export cell • Saved to <code>{BASE_DIR}</code> • Plots in <code>{PLOTS_DIR}</code> • Samples in <code>{SAMPLES_DIR}</code> • <a href="#top">Back to top</a>
              </div>
            </div>
          </body>
        </html>
        \"\"\"

        md_path = os.path.join(BASE_DIR, "Lab_Report_12.md")
        html_path = os.path.join(BASE_DIR, "Lab_Report_12.html")

        with open(md_path, "w", encoding="utf-8") as f:
            f.write(report_md)
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html)

        print("Saved:", os.path.abspath(md_path))
        print("Saved:", os.path.abspath(html_path))
        print("Plots currently saved:")
        for filename in sorted(os.listdir(PLOTS_DIR)):
            print(" -", filename)
        print("Sample images saved (last 5):")
        for filename in sorted(os.listdir(SAMPLES_DIR))[-5:]:
            print(" -", filename)
        """
    ),
]


notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.x"},
        "colab": {"name": "lab12_gan_mnist_colab.ipynb"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

NOTEBOOK_PATH.write_text(json.dumps(notebook, indent=2), encoding="utf-8")
print("Wrote:", NOTEBOOK_PATH)
