import json
import textwrap
from pathlib import Path


ROOT = Path(__file__).resolve().parent
NOTEBOOK_PATH = ROOT / "zarmeena_open_ended_waste_lab.ipynb"
README_PATH = ROOT / "README.md"


def md(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": textwrap.dedent(source).strip("\n").splitlines(keepends=True),
    }


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": textwrap.dedent(source).strip("\n").splitlines(keepends=True),
    }


cells = [
    md(
        """
        # Open Ended Lab - Waste Sorting Model

        **Student:** Zarmeena Jawad  
        **Roll No:** B23F0115AI125  
        **Section:** AI Red  
        **Platform:** Google Colab

        This notebook solves the open-ended lab in four parts:
        - baseline CNN check
        - parameter tuning
        - MobileNetV2 transfer learning
        - fine-tuning
        """
    ),
    code(
        """
        import os
        from datetime import datetime

        STUDENT_NAME = "Zarmeena Jawad"
        STUDENT_ROLL = "B23F0115AI125"
        STUDENT_SECTION = "AI Red"
        STUDENT_FOLDER = "Zarmeena's Lab"

        try:
            from google.colab import drive  # type: ignore
            IN_COLAB = True
        except Exception:
            IN_COLAB = False

        if IN_COLAB:
            drive.mount("/content/drive")
            default_base = f"/content/drive/MyDrive/COMP-341L/Open Ended Lab/{STUDENT_FOLDER}"
        else:
            default_base = "."

        BASE_DIR = os.environ.get("OPEN_ENDED_ZARMEENA_BASE_DIR", default_base)
        FIG_DIR = os.path.join(BASE_DIR, "figures")

        os.makedirs(BASE_DIR, exist_ok=True)
        os.makedirs(FIG_DIR, exist_ok=True)

        print("BASE_DIR:", os.path.abspath(BASE_DIR))
        print("FIG_DIR :", os.path.abspath(FIG_DIR))
        """
    ),
    md(
        """
        ## 1. Imports
        """
    ),
    code(
        """
        import json
        import random
        import numpy as np
        import matplotlib.pyplot as plt
        import tensorflow as tf

        from sklearn.metrics import confusion_matrix, classification_report

        SEED = 17
        random.seed(SEED)
        np.random.seed(SEED)
        tf.random.set_seed(SEED)

        BATCH_SIZE = 32
        BASE_IMG = (32, 32)
        TL_IMG = (96, 96)
        NUM_CLASSES = 3
        AUTOTUNE = tf.data.AUTOTUNE

        print("TensorFlow:", tf.__version__)
        """
    ),
    md(
        """
        ## 2. Data Setup

        The manual allows a CIFAR-10 subset, so this notebook builds a 3-class proxy dataset:
        - `truck -> Plastic`
        - `deer -> Paper`
        - `ship -> Metal`
        """
    ),
    code(
        """
        CLASS_MAP = {
            9: 0,  # truck -> Plastic
            4: 1,  # deer  -> Paper
            8: 2,  # ship  -> Metal
        }
        LABELS = ["Plastic", "Paper", "Metal"]

        (x_train_full, y_train_full), (x_test_full, y_test_full) = tf.keras.datasets.cifar10.load_data()
        x_all = np.concatenate([x_train_full, x_test_full], axis=0)
        y_all = np.concatenate([y_train_full, y_test_full], axis=0).reshape(-1)

        keep = np.isin(y_all, list(CLASS_MAP.keys()))
        x_pool = x_all[keep]
        y_pool = y_all[keep]
        y_pool = np.array([CLASS_MAP[int(v)] for v in y_pool], dtype=np.int32)

        rng = np.random.default_rng(SEED)
        chosen = []
        per_class = 650
        for class_id in range(NUM_CLASSES):
            ids = np.where(y_pool == class_id)[0]
            chosen.extend(rng.choice(ids, size=per_class, replace=False).tolist())

        chosen = np.array(chosen)
        rng.shuffle(chosen)

        x_data = x_pool[chosen]
        y_data = y_pool[chosen]

        print("x_data:", x_data.shape)
        print("y_data:", y_data.shape)
        print({LABELS[i]: int((y_data == i).sum()) for i in range(NUM_CLASSES)})
        """
    ),
    code(
        """
        total = len(x_data)
        train_end = int(0.70 * total)
        val_end = int(0.85 * total)

        x_train, y_train = x_data[:train_end], y_data[:train_end]
        x_val, y_val = x_data[train_end:val_end], y_data[train_end:val_end]
        x_test, y_test = x_data[val_end:], y_data[val_end:]

        print("Train:", x_train.shape, y_train.shape)
        print("Val  :", x_val.shape, y_val.shape)
        print("Test :", x_test.shape, y_test.shape)
        """
    ),
    code(
        """
        fig, axes = plt.subplots(3, 3, figsize=(8, 8))
        for ax, idx in zip(axes.flat, range(9)):
            ax.imshow(x_train[idx])
            ax.set_title(LABELS[int(y_train[idx])], fontsize=10)
            ax.axis("off")
        plt.tight_layout()
        sample_file = os.path.join(FIG_DIR, "samples.png")
        plt.savefig(sample_file, dpi=130, bbox_inches="tight")
        plt.show()
        print("Saved:", sample_file)
        """
    ),
    code(
        """
        def prep_small(images, labels):
            images = tf.cast(images, tf.float32) / 255.0
            return images, tf.cast(labels, tf.int32)


        def prep_transfer(images, labels):
            images = tf.cast(images, tf.float32)
            images = tf.image.resize(images, TL_IMG)
            images = tf.keras.applications.mobilenet_v2.preprocess_input(images)
            return images, tf.cast(labels, tf.int32)


        def make_ds(x, y, fn, training=False):
            ds = tf.data.Dataset.from_tensor_slices((x, y))
            if training:
                ds = ds.shuffle(len(x), seed=SEED)
            ds = ds.map(fn, num_parallel_calls=AUTOTUNE)
            return ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)


        train_ds = make_ds(x_train, y_train, prep_small, training=True)
        val_ds = make_ds(x_val, y_val, prep_small)
        test_ds = make_ds(x_test, y_test, prep_small)

        train_tl_ds = make_ds(x_train, y_train, prep_transfer, training=True)
        val_tl_ds = make_ds(x_val, y_val, prep_transfer)
        test_tl_ds = make_ds(x_test, y_test, prep_transfer)
        """
    ),
    md(
        """
        ## 3. Task 1: Baseline CNN
        Train the given CNN for 10 epochs and check train vs validation accuracy.
        """
    ),
    code(
        """
        def baseline_cnn():
            model = tf.keras.Sequential([
                tf.keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=(32, 32, 3)),
                tf.keras.layers.MaxPooling2D(2, 2),
                tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
                tf.keras.layers.MaxPooling2D(2, 2),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(128, activation="relu"),
                tf.keras.layers.Dense(3, activation="softmax"),
            ])
            model.compile(
                optimizer="adam",
                loss="sparse_categorical_crossentropy",
                metrics=["accuracy"],
            )
            return model


        model_a = baseline_cnn()
        model_a.summary()
        """
    ),
    code(
        """
        hist_a = model_a.fit(
            train_ds,
            validation_data=val_ds,
            epochs=10,
            verbose=1,
        )

        a_train = float(hist_a.history["accuracy"][-1])
        a_val = float(hist_a.history["val_accuracy"][-1])
        a_gap = a_train - a_val

        print("train_acc:", a_train)
        print("val_acc  :", a_val)
        print("gap      :", a_gap)
        """
    ),
    code(
        """
        plt.figure(figsize=(7, 4.5))
        plt.plot(hist_a.history["accuracy"], label="train", linewidth=2)
        plt.plot(hist_a.history["val_accuracy"], label="val", linewidth=2)
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("Baseline CNN")
        plt.legend()
        plt.tight_layout()
        curve_a = os.path.join(FIG_DIR, "baseline_accuracy.png")
        plt.savefig(curve_a, dpi=130, bbox_inches="tight")
        plt.show()
        print("Saved:", curve_a)
        """
    ),
    code(
        """
        baseline_note = (
            f"The baseline shows overfitting because train accuracy is {a_train:.4f} "
            f"while validation accuracy is {a_val:.4f}."
            if a_gap > 0.05 else
            f"The baseline does not show strong overfitting yet because the gap is only {a_gap:.4f}."
        )
        print(baseline_note)
        """
    ),
    md(
        """
        ## 4. Task 2: Modified CNN
        This version changes:
        - optimizer: `RMSprop`
        - learning rate: `0.0005`
        - dropout added
        - dense units reduced to `96`
        """
    ),
    code(
        """
        def tuned_cnn():
            model = tf.keras.Sequential([
                tf.keras.layers.Input(shape=(32, 32, 3)),
                tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu"),
                tf.keras.layers.MaxPooling2D(2),
                tf.keras.layers.Dropout(0.20),

                tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu"),
                tf.keras.layers.MaxPooling2D(2),
                tf.keras.layers.Dropout(0.30),

                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(96, activation="relu"),
                tf.keras.layers.Dropout(0.35),
                tf.keras.layers.Dense(NUM_CLASSES, activation="softmax"),
            ])
            model.compile(
                optimizer=tf.keras.optimizers.RMSprop(learning_rate=5e-4),
                loss="sparse_categorical_crossentropy",
                metrics=["accuracy"],
            )
            return model


        model_b = tuned_cnn()
        model_b.summary()
        """
    ),
    code(
        """
        hist_b = model_b.fit(
            train_ds,
            validation_data=val_ds,
            epochs=10,
            verbose=1,
        )

        b_train = float(hist_b.history["accuracy"][-1])
        b_val = float(hist_b.history["val_accuracy"][-1])

        print("modified_train:", b_train)
        print("modified_val  :", b_val)
        """
    ),
    code(
        """
        plt.figure(figsize=(7, 4.5))
        plt.plot(hist_a.history["val_accuracy"], label="original val", linewidth=2)
        plt.plot(hist_b.history["val_accuracy"], label="modified val", linewidth=2)
        plt.xlabel("Epoch")
        plt.ylabel("Validation Accuracy")
        plt.title("Original vs Modified")
        plt.legend()
        plt.tight_layout()
        curve_b = os.path.join(FIG_DIR, "modified_compare.png")
        plt.savefig(curve_b, dpi=130, bbox_inches="tight")
        plt.show()
        print("Saved:", curve_b)
        """
    ),
    md(
        """
        ## 5. Task 3: MobileNetV2
        Freeze the base model, add a small head, and train for 5 epochs.
        """
    ),
    code(
        """
        def transfer_model():
            augment = tf.keras.Sequential([
                tf.keras.layers.RandomFlip("horizontal"),
                tf.keras.layers.RandomZoom(0.10),
            ])

            base = tf.keras.applications.MobileNetV2(
                input_shape=TL_IMG + (3,),
                include_top=False,
                weights="imagenet",
            )
            base.trainable = False

            inp = tf.keras.Input(shape=TL_IMG + (3,))
            x = augment(inp)
            x = base(x, training=False)
            x = tf.keras.layers.GlobalAveragePooling2D()(x)
            x = tf.keras.layers.Dense(128, activation="relu")(x)
            x = tf.keras.layers.Dropout(0.25)(x)
            out = tf.keras.layers.Dense(NUM_CLASSES, activation="softmax")(x)

            model = tf.keras.Model(inp, out)
            model.compile(
                optimizer=tf.keras.optimizers.Adam(1e-3),
                loss="sparse_categorical_crossentropy",
                metrics=["accuracy"],
            )
            return model, base


        model_c, base_c = transfer_model()
        model_c.summary()
        """
    ),
    code(
        """
        hist_c = model_c.fit(
            train_tl_ds,
            validation_data=val_tl_ds,
            epochs=5,
            verbose=1,
        )

        c_train = float(hist_c.history["accuracy"][-1])
        c_val = float(hist_c.history["val_accuracy"][-1])

        print("transfer_train:", c_train)
        print("transfer_val  :", c_val)
        """
    ),
    code(
        """
        plt.figure(figsize=(7, 4.5))
        plt.plot(hist_c.history["accuracy"], label="train", linewidth=2)
        plt.plot(hist_c.history["val_accuracy"], label="val", linewidth=2)
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("MobileNetV2 (Frozen Base)")
        plt.legend()
        plt.tight_layout()
        curve_c = os.path.join(FIG_DIR, "transfer_accuracy.png")
        plt.savefig(curve_c, dpi=130, bbox_inches="tight")
        plt.show()
        print("Saved:", curve_c)
        """
    ),
    md(
        """
        ## 6. Task 4: Fine-Tuning
        Unfreeze the last 10 layers, lower the learning rate, and train for 3 more epochs.
        """
    ),
    code(
        """
        base_c.trainable = True
        for layer in base_c.layers[:-10]:
            layer.trainable = False

        model_c.compile(
            optimizer=tf.keras.optimizers.Adam(1e-5),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        hist_d = model_c.fit(
            train_tl_ds,
            validation_data=val_tl_ds,
            epochs=3,
            verbose=1,
        )

        d_train = float(hist_d.history["accuracy"][-1])
        d_val = float(hist_d.history["val_accuracy"][-1])

        print("finetune_train:", d_train)
        print("finetune_val  :", d_val)
        """
    ),
    code(
        """
        plt.figure(figsize=(7, 4.5))
        plt.plot(hist_d.history["accuracy"], label="train", linewidth=2)
        plt.plot(hist_d.history["val_accuracy"], label="val", linewidth=2)
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("Fine-Tuning")
        plt.legend()
        plt.tight_layout()
        curve_d = os.path.join(FIG_DIR, "finetune_accuracy.png")
        plt.savefig(curve_d, dpi=130, bbox_inches="tight")
        plt.show()
        print("Saved:", curve_d)
        """
    ),
    code(
        """
        test_loss, test_acc = model_c.evaluate(test_tl_ds, verbose=0)
        pred_prob = model_c.predict(test_tl_ds, verbose=0)
        pred_cls = np.argmax(pred_prob, axis=1)

        cm = confusion_matrix(y_test, pred_cls)
        report = classification_report(y_test, pred_cls, target_names=LABELS, digits=4)

        print("test_acc :", test_acc)
        print("test_loss:", test_loss)
        print(report)

        fig, ax = plt.subplots(figsize=(5.5, 5))
        im = ax.imshow(cm, cmap="Oranges")
        ax.set_xticks(range(NUM_CLASSES))
        ax.set_yticks(range(NUM_CLASSES))
        ax.set_xticklabels(LABELS)
        ax.set_yticklabels(LABELS)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        for i in range(NUM_CLASSES):
            for j in range(NUM_CLASSES):
                ax.text(j, i, int(cm[i, j]), ha="center", va="center")
        plt.colorbar(im)
        plt.tight_layout()
        cm_file = os.path.join(FIG_DIR, "confusion_matrix.png")
        plt.savefig(cm_file, dpi=130, bbox_inches="tight")
        plt.show()
        print("Saved:", cm_file)
        """
    ),
    code(
        """
        answer_1 = (
            f"Yes. Validation accuracy moved from {c_val:.4f} to {d_val:.4f} after fine-tuning."
            if d_val > c_val else
            f"No clear gain. Validation accuracy changed from {c_val:.4f} to {d_val:.4f}."
        )

        answer_2 = (
            "A smaller learning rate is used in fine-tuning so pretrained weights change slowly. "
            "This protects useful ImageNet features from being disturbed too aggressively."
        )

        print(answer_1)
        print(answer_2)
        """
    ),
    md(
        """
        ## 7. Export
        The next cell writes a short report in Markdown and HTML.
        """
    ),
    code(
        """
        summary = {
            "baseline_train": a_train,
            "baseline_val": a_val,
            "modified_train": b_train,
            "modified_val": b_val,
            "transfer_train": c_train,
            "transfer_val": c_val,
            "finetune_train": d_train,
            "finetune_val": d_val,
            "test_acc": float(test_acc),
            "test_loss": float(test_loss),
        }

        with open(os.path.join(BASE_DIR, "summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        report_md = f\"\"\"# Open Ended Lab Report

        **Name:** {STUDENT_NAME}  
        **Roll No:** {STUDENT_ROLL}  
        **Section:** {STUDENT_SECTION}  
        **Date:** {datetime.now().strftime('%B %d, %Y')}

        ## Dataset
        CIFAR-10 subset used as allowed by the manual:
        - truck -> Plastic
        - deer -> Paper
        - ship -> Metal

        ![Samples](figures/samples.png)

        ## Task 1
        - Train Acc: **{a_train:.4f}**
        - Val Acc: **{a_val:.4f}**
        - Gap: **{a_gap:.4f}**

        {baseline_note}

        ![Baseline](figures/baseline_accuracy.png)

        ## Task 2
        Changes applied:
        - RMSprop optimizer
        - learning rate 0.0005
        - dropout layers
        - dense units reduced to 96

        | Version | Train Acc | Val Acc |
        |---|---:|---:|
        | Original | {a_train:.4f} | {a_val:.4f} |
        | Modified | {b_train:.4f} | {b_val:.4f} |

        ![Modified](figures/modified_compare.png)

        ## Task 3
        - Transfer Train Acc: **{c_train:.4f}**
        - Transfer Val Acc: **{c_val:.4f}**

        ![Transfer](figures/transfer_accuracy.png)

        ## Task 4
        - Fine-tune Train Acc: **{d_train:.4f}**
        - Fine-tune Val Acc: **{d_val:.4f}**
        - Test Acc: **{test_acc:.4f}**
        - Test Loss: **{test_loss:.4f}**

        **Did validation improve?**  
        {answer_1}

        **Why reduce learning rate?**  
        {answer_2}

        ![Fine-tuning](figures/finetune_accuracy.png)
        ![Confusion Matrix](figures/confusion_matrix.png)

        ## Classification Report
        ```text
        {report.strip()}
        ```

        ## Conclusion
        The scratch CNN gave a useful starting point but generalization was limited. Regularization and optimizer changes improved stability, while MobileNetV2 provided stronger image features. Fine-tuning with a low learning rate allowed safer adaptation of the pretrained network.
        \"\"\"

        report_html = f\"\"\"<!DOCTYPE html>
        <html lang="en">
        <head>
          <meta charset="utf-8">
          <title>Zarmeena Open Ended Lab</title>
          <style>
            body {{ font-family: Georgia, serif; max-width: 820px; margin: 32px auto; line-height: 1.55; color: #202020; padding: 0 18px; }}
            h1, h2 {{ color: #111; }}
            table {{ border-collapse: collapse; width: 100%; margin: 14px 0; }}
            th, td {{ border: 1px solid #bbb; padding: 8px; }}
            th {{ background: #f4efe7; }}
            pre {{ background: #f7f7f7; padding: 10px; overflow-x: auto; }}
            img {{ max-width: 100%; border: 1px solid #ddd; margin: 12px 0 20px; }}
          </style>
        </head>
        <body>
          <h1>Open Ended Lab Report</h1>
          <p><strong>Name:</strong> {STUDENT_NAME}<br>
          <strong>Roll No:</strong> {STUDENT_ROLL}<br>
          <strong>Section:</strong> {STUDENT_SECTION}<br>
          <strong>Date:</strong> {datetime.now().strftime('%B %d, %Y')}</p>

          <h2>Dataset</h2>
          <p>CIFAR-10 subset used as allowed by the manual: truck -&gt; Plastic, deer -&gt; Paper, ship -&gt; Metal.</p>
          <img src="figures/samples.png" alt="Samples">

          <h2>Task 1</h2>
          <p>Train Acc: {a_train:.4f}<br>Val Acc: {a_val:.4f}<br>Gap: {a_gap:.4f}</p>
          <p>{baseline_note}</p>
          <img src="figures/baseline_accuracy.png" alt="Baseline">

          <h2>Task 2</h2>
          <table>
            <tr><th>Version</th><th>Train Acc</th><th>Val Acc</th></tr>
            <tr><td>Original</td><td>{a_train:.4f}</td><td>{a_val:.4f}</td></tr>
            <tr><td>Modified</td><td>{b_train:.4f}</td><td>{b_val:.4f}</td></tr>
          </table>
          <img src="figures/modified_compare.png" alt="Modified">

          <h2>Task 3</h2>
          <p>Transfer Train Acc: {c_train:.4f}<br>Transfer Val Acc: {c_val:.4f}</p>
          <img src="figures/transfer_accuracy.png" alt="Transfer">

          <h2>Task 4</h2>
          <p>Fine-tune Train Acc: {d_train:.4f}<br>Fine-tune Val Acc: {d_val:.4f}<br>Test Acc: {test_acc:.4f}<br>Test Loss: {test_loss:.4f}</p>
          <p><strong>Did validation improve?</strong> {answer_1}</p>
          <p><strong>Why reduce learning rate?</strong> {answer_2}</p>
          <img src="figures/finetune_accuracy.png" alt="Fine-tuning">
          <img src="figures/confusion_matrix.png" alt="Confusion Matrix">

          <h2>Classification Report</h2>
          <pre>{report.strip()}</pre>
        </body>
        </html>
        \"\"\"

        md_file = os.path.join(BASE_DIR, "OpenEnded_Report.md")
        html_file = os.path.join(BASE_DIR, "OpenEnded_Report.html")

        with open(md_file, "w", encoding="utf-8") as f:
            f.write(report_md)
        with open(html_file, "w", encoding="utf-8") as f:
            f.write(report_html)

        print("Saved:", md_file)
        print("Saved:", html_file)
        """
    ),
]


notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.10",
        },
        "colab": {
            "name": NOTEBOOK_PATH.name,
            "provenance": [],
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}


NOTEBOOK_PATH.write_text(json.dumps(notebook, indent=2), encoding="utf-8")

README_PATH.write_text(
    textwrap.dedent(
        """
        # Zarmeena Open Ended Lab

        Files in this folder:
        - `zarmeena_open_ended_waste_lab.ipynb`
        - `build_zarmeena_open_ended_notebook.py`

        Run the notebook in Google Colab. It will save outputs in:
        `/content/drive/MyDrive/COMP-341L/Open Ended Lab/Zarmeena's Lab`

        Generated items:
        - figures
        - `summary.json`
        - `OpenEnded_Report.md`
        - `OpenEnded_Report.html`
        """
    ).strip()
    + "\n",
    encoding="utf-8",
)

print(f"Created {NOTEBOOK_PATH}")
print(f"Created {README_PATH}")
