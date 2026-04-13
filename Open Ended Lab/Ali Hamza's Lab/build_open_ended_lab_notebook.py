import json
import textwrap
from pathlib import Path


ROOT = Path(__file__).resolve().parent
NOTEBOOK_PATH = ROOT / "open_ended_lab_waste_classification_colab.ipynb"
README_PATH = ROOT / "README.md"


def md_cell(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": textwrap.dedent(source).strip("\n").splitlines(keepends=True),
    }


def code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": textwrap.dedent(source).strip("\n").splitlines(keepends=True),
    }


cells = [
    md_cell(
        """
        # Open Ended Lab: Real-Time Waste Classification (Ali Hamza)

        **Course:** COMP-341L - Artificial Neural Networks Lab  
        **Student:** Ali Hamza  
        **Roll Number:** B23F0063AI106  
        **Section:** B.S AI - Red  
        **Environment:** Google Colab

        This notebook completes the open-ended lab step by step:
        1. Train the given CNN for 10 epochs and diagnose overfitting
        2. Modify at least two network parameters and compare validation performance
        3. Replace the CNN with a pretrained MobileNetV2 model
        4. Fine-tune the last 10 layers with a smaller learning rate
        5. Auto-generate a Markdown and HTML report in the lab folder

        ## Reproducibility note
        The lab manual allows either a **CIFAR-10 subset** or a **custom waste dataset**.  
        To keep this notebook fully reproducible in Colab without external downloads, it uses a **3-class CIFAR-10 subset as a proxy dataset** and maps the classes to:
        - `Plastic`
        - `Paper`
        - `Metal`

        If you later want to replace the proxy dataset with a real waste dataset, you only need to change the dataset-loading cell; the rest of the workflow stays the same.
        """
    ),
    code_cell(
        """
        import os
        from datetime import datetime

        STUDENT_NAME = "Ali Hamza"
        STUDENT_ROLL = "B23F0063AI106"
        STUDENT_SECTION = "B.S AI - Red"
        STUDENT_FOLDER_NAME = "Ali Hamza's Lab"

        try:
            from google.colab import drive  # type: ignore
            IN_COLAB = True
        except Exception:
            IN_COLAB = False

        if IN_COLAB:
            drive.mount('/content/drive')
            default_base_dir = f"/content/drive/MyDrive/COMP-341L/Open Ended Lab/{STUDENT_FOLDER_NAME}"
        else:
            default_base_dir = "."

        BASE_DIR = os.environ.get("OPEN_ENDED_LAB_BASE_DIR", default_base_dir)
        PLOTS_DIR = os.path.join(BASE_DIR, "plots")

        os.makedirs(BASE_DIR, exist_ok=True)
        os.makedirs(PLOTS_DIR, exist_ok=True)

        print("IN_COLAB :", IN_COLAB)
        print("BASE_DIR :", os.path.abspath(BASE_DIR))
        print("PLOTS_DIR:", os.path.abspath(PLOTS_DIR))
        """
    ),
    md_cell(
        """
        ## Task 0: Imports and Configuration

        We keep the training setup lightweight so it runs comfortably in Google Colab.
        """
    ),
    code_cell(
        """
        import json
        import random
        import numpy as np
        import matplotlib.pyplot as plt
        import tensorflow as tf

        from sklearn.metrics import confusion_matrix, classification_report

        SEED = 42
        random.seed(SEED)
        np.random.seed(SEED)
        tf.random.set_seed(SEED)

        IMG_SIZE = (32, 32)
        TRANSFER_IMG_SIZE = (96, 96)
        BATCH_SIZE = 32
        NUM_CLASSES = 3
        AUTOTUNE = tf.data.AUTOTUNE

        tf.__version__
        """
    ),
    md_cell(
        """
        ## Task 1: Dataset Loading

        We use a **CIFAR-10 subset** with three source classes and map them to the required waste labels:

        - CIFAR-10 `airplane` -> `Plastic` (proxy)
        - CIFAR-10 `automobile` -> `Paper` (proxy)
        - CIFAR-10 `ship` -> `Metal` (proxy)

        This proxy setup is acceptable because the manual explicitly allows a CIFAR-10 subset.
        """
    ),
    code_cell(
        """
        WASTE_LABELS = ["Plastic", "Paper", "Metal"]
        CIFAR_PROXY_MAP = {
            0: 0,  # airplane   -> Plastic
            1: 1,  # automobile -> Paper
            8: 2,  # ship       -> Metal
        }
        CIFAR_CLASS_NAMES = [
            "airplane", "automobile", "bird", "cat", "deer",
            "dog", "frog", "horse", "ship", "truck"
        ]

        (x_train_full, y_train_full), (x_test_full, y_test_full) = tf.keras.datasets.cifar10.load_data()
        x_all = np.concatenate([x_train_full, x_test_full], axis=0)
        y_all = np.concatenate([y_train_full, y_test_full], axis=0).reshape(-1)

        mask = np.isin(y_all, list(CIFAR_PROXY_MAP.keys()))
        x_subset = x_all[mask]
        y_subset_original = y_all[mask]
        y_subset = np.array([CIFAR_PROXY_MAP[int(label)] for label in y_subset_original], dtype=np.int64)

        # Keep the dataset intentionally small to expose overfitting in the baseline model.
        samples_per_class = 700
        selected_indices = []
        rng = np.random.default_rng(SEED)
        for class_id in range(NUM_CLASSES):
            class_indices = np.where(y_subset == class_id)[0]
            chosen = rng.choice(class_indices, size=samples_per_class, replace=False)
            selected_indices.extend(chosen.tolist())

        selected_indices = np.array(selected_indices)
        rng.shuffle(selected_indices)

        x_data = x_subset[selected_indices]
        y_data = y_subset[selected_indices]

        print("Dataset shape:", x_data.shape, y_data.shape)
        print("Class counts:", {WASTE_LABELS[i]: int((y_data == i).sum()) for i in range(NUM_CLASSES)})
        """
    ),
    code_cell(
        """
        def train_val_test_split(x, y, train_ratio=0.70, val_ratio=0.15):
            n = len(x)
            train_end = int(n * train_ratio)
            val_end = int(n * (train_ratio + val_ratio))
            return (
                x[:train_end], y[:train_end],
                x[train_end:val_end], y[train_end:val_end],
                x[val_end:], y[val_end:]
            )


        x_train, y_train, x_val, y_val, x_test, y_test = train_val_test_split(x_data, y_data)

        print("Train:", x_train.shape, y_train.shape)
        print("Val  :", x_val.shape, y_val.shape)
        print("Test :", x_test.shape, y_test.shape)
        """
    ),
    code_cell(
        """
        fig, axes = plt.subplots(2, 5, figsize=(12, 5))
        for ax, idx in zip(axes.flat, range(10)):
            ax.imshow(x_train[idx])
            ax.set_title(WASTE_LABELS[int(y_train[idx])])
            ax.axis("off")

        plt.suptitle("Sample Images from the Proxy Waste Dataset")
        plt.tight_layout()
        sample_path = os.path.join(PLOTS_DIR, "task0_sample_images.png")
        plt.savefig(sample_path, dpi=130, bbox_inches="tight")
        plt.show()
        print("Saved:", sample_path)
        """
    ),
    md_cell(
        """
        ## Task 0.1: TensorFlow Input Pipelines
        """
    ),
    code_cell(
        """
        def preprocess_for_cnn(images, labels):
            images = tf.cast(images, tf.float32) / 255.0
            labels = tf.cast(labels, tf.int32)
            return images, labels


        def preprocess_for_transfer(images, labels):
            images = tf.cast(images, tf.float32)
            images = tf.image.resize(images, TRANSFER_IMG_SIZE)
            images = tf.keras.applications.mobilenet_v2.preprocess_input(images)
            labels = tf.cast(labels, tf.int32)
            return images, labels


        def make_dataset(x, y, preprocess_fn, training=False):
            ds = tf.data.Dataset.from_tensor_slices((x, y))
            if training:
                ds = ds.shuffle(len(x), seed=SEED)
            ds = ds.map(preprocess_fn, num_parallel_calls=AUTOTUNE)
            ds = ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)
            return ds


        train_ds = make_dataset(x_train, y_train, preprocess_for_cnn, training=True)
        val_ds = make_dataset(x_val, y_val, preprocess_for_cnn, training=False)
        test_ds = make_dataset(x_test, y_test, preprocess_for_cnn, training=False)

        train_transfer_ds = make_dataset(x_train, y_train, preprocess_for_transfer, training=True)
        val_transfer_ds = make_dataset(x_val, y_val, preprocess_for_transfer, training=False)
        test_transfer_ds = make_dataset(x_test, y_test, preprocess_for_transfer, training=False)
        """
    ),
    md_cell(
        """
        ## Task 1 (1 Mark): Overfitting Diagnosis

        Train the **given CNN architecture** for **10 epochs**, plot training vs validation accuracy, and decide whether the model is overfitting.
        """
    ),
    code_cell(
        """
        def build_baseline_model():
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


        baseline_model = build_baseline_model()
        baseline_model.summary()
        """
    ),
    code_cell(
        """
        history_baseline = baseline_model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=10,
            verbose=1,
        )

        baseline_train_acc = float(history_baseline.history["accuracy"][-1])
        baseline_val_acc = float(history_baseline.history["val_accuracy"][-1])
        baseline_gap = baseline_train_acc - baseline_val_acc

        print("Baseline train accuracy:", baseline_train_acc)
        print("Baseline val accuracy  :", baseline_val_acc)
        print("Generalization gap     :", baseline_gap)
        """
    ),
    code_cell(
        """
        plt.figure(figsize=(8, 5))
        plt.plot(history_baseline.history["accuracy"], marker="o", label="Training Accuracy")
        plt.plot(history_baseline.history["val_accuracy"], marker="s", label="Validation Accuracy")
        plt.title("Task 1: Training vs Validation Accuracy (Baseline CNN)")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        task1_curve_path = os.path.join(PLOTS_DIR, "task1_accuracy_curve.png")
        plt.savefig(task1_curve_path, dpi=130, bbox_inches="tight")
        plt.show()
        print("Saved:", task1_curve_path)
        """
    ),
    code_cell(
        """
        if baseline_gap > 0.05:
            overfitting_decision = "Yes, the model is overfitting."
            overfitting_reason = (
                f"Training accuracy ({baseline_train_acc:.4f}) is noticeably higher than validation accuracy "
                f"({baseline_val_acc:.4f}), giving a gap of {baseline_gap:.4f}."
            )
            overfitting_reason_2 = (
                "This means the model is learning the training data better than unseen validation images, "
                "so its generalization is weaker."
            )
        else:
            overfitting_decision = "No strong overfitting is visible after 10 epochs."
            overfitting_reason = (
                f"Training accuracy ({baseline_train_acc:.4f}) and validation accuracy ({baseline_val_acc:.4f}) "
                f"remain relatively close, with a gap of {baseline_gap:.4f}."
            )
            overfitting_reason_2 = (
                "The model may still be improving, but the current run does not show a severe generalization drop."
            )

        print(overfitting_decision)
        print(overfitting_reason)
        print(overfitting_reason_2)
        """
    ),
    md_cell(
        """
        ## Task 2 (2 Marks): Modify Network Parameters

        Improve validation performance by changing **at least two** parameters.  
        This notebook changes **four**:

        - Adds `BatchNormalization`
        - Adds `Dropout`
        - Reduces dense units from `128` to `64`
        - Reduces learning rate from Adam default to `3e-4`
        """
    ),
    code_cell(
        """
        def build_modified_model():
            model = tf.keras.Sequential([
                tf.keras.layers.Input(shape=(32, 32, 3)),
                tf.keras.layers.Conv2D(32, (3, 3), padding="same", activation=None),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Activation("relu"),
                tf.keras.layers.MaxPooling2D(2, 2),
                tf.keras.layers.Dropout(0.25),

                tf.keras.layers.Conv2D(64, (3, 3), padding="same", activation=None),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Activation("relu"),
                tf.keras.layers.MaxPooling2D(2, 2),
                tf.keras.layers.Dropout(0.30),

                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(64, activation="relu"),
                tf.keras.layers.Dropout(0.40),
                tf.keras.layers.Dense(3, activation="softmax"),
            ])

            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=3e-4),
                loss="sparse_categorical_crossentropy",
                metrics=["accuracy"],
            )
            return model


        modified_model = build_modified_model()
        modified_model.summary()
        """
    ),
    code_cell(
        """
        history_modified = modified_model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=10,
            verbose=1,
        )

        modified_train_acc = float(history_modified.history["accuracy"][-1])
        modified_val_acc = float(history_modified.history["val_accuracy"][-1])

        print("Modified train accuracy:", modified_train_acc)
        print("Modified val accuracy  :", modified_val_acc)
        print("Validation improvement :", modified_val_acc - baseline_val_acc)
        """
    ),
    code_cell(
        """
        plt.figure(figsize=(8, 5))
        plt.plot(history_baseline.history["val_accuracy"], marker="o", label="Baseline Val Accuracy")
        plt.plot(history_modified.history["val_accuracy"], marker="s", label="Modified Val Accuracy")
        plt.title("Task 2: Validation Accuracy Comparison")
        plt.xlabel("Epoch")
        plt.ylabel("Validation Accuracy")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        task2_curve_path = os.path.join(PLOTS_DIR, "task2_validation_comparison.png")
        plt.savefig(task2_curve_path, dpi=130, bbox_inches="tight")
        plt.show()
        print("Saved:", task2_curve_path)

        comparison_rows = [
            ["Original", baseline_train_acc, baseline_val_acc],
            ["Modified", modified_train_acc, modified_val_acc],
        ]
        for row in comparison_rows:
            print(row)
        """
    ),
    md_cell(
        """
        ## Task 3 (3 Marks): Transfer Learning Enhancement

        Replace the CNN with **MobileNetV2** using:
        - `include_top=False`
        - Frozen base layers
        - A custom classification head
        - Training for **5 epochs**
        """
    ),
    code_cell(
        """
        def build_transfer_model():
            data_augmentation = tf.keras.Sequential([
                tf.keras.layers.RandomFlip("horizontal"),
                tf.keras.layers.RandomRotation(0.08),
            ], name="augmentation")

            base_model = tf.keras.applications.MobileNetV2(
                input_shape=TRANSFER_IMG_SIZE + (3,),
                include_top=False,
                weights="imagenet",
            )
            base_model.trainable = False

            inputs = tf.keras.Input(shape=TRANSFER_IMG_SIZE + (3,))
            x = data_augmentation(inputs)
            x = base_model(x, training=False)
            x = tf.keras.layers.GlobalAveragePooling2D()(x)
            x = tf.keras.layers.Dropout(0.30)(x)
            outputs = tf.keras.layers.Dense(NUM_CLASSES, activation="softmax")(x)

            model = tf.keras.Model(inputs, outputs, name="mobilenetv2_transfer")
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                loss="sparse_categorical_crossentropy",
                metrics=["accuracy"],
            )
            return model, base_model


        transfer_model, mobilenet_base = build_transfer_model()
        transfer_model.summary()
        """
    ),
    code_cell(
        """
        history_transfer = transfer_model.fit(
            train_transfer_ds,
            validation_data=val_transfer_ds,
            epochs=5,
            verbose=1,
        )

        transfer_val_acc = float(history_transfer.history["val_accuracy"][-1])
        transfer_train_acc = float(history_transfer.history["accuracy"][-1])

        print("Transfer train accuracy:", transfer_train_acc)
        print("Transfer val accuracy  :", transfer_val_acc)
        """
    ),
    code_cell(
        """
        plt.figure(figsize=(8, 5))
        plt.plot(history_transfer.history["accuracy"], marker="o", label="Train Accuracy")
        plt.plot(history_transfer.history["val_accuracy"], marker="s", label="Val Accuracy")
        plt.title("Task 3: MobileNetV2 Transfer Learning Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        task3_curve_path = os.path.join(PLOTS_DIR, "task3_transfer_accuracy.png")
        plt.savefig(task3_curve_path, dpi=130, bbox_inches="tight")
        plt.show()
        print("Saved:", task3_curve_path)
        """
    ),
    md_cell(
        """
        ## Task 4 (4 Marks): Fine-Tuning Decision

        - Unfreeze the **last 10 layers**
        - Reduce the learning rate
        - Train **3 more epochs**
        - Answer:
          1. Did validation improve?
          2. Why reduce learning rate during fine-tuning?
        """
    ),
    code_cell(
        """
        mobilenet_base.trainable = True

        for layer in mobilenet_base.layers[:-10]:
            layer.trainable = False

        transfer_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        history_finetune = transfer_model.fit(
            train_transfer_ds,
            validation_data=val_transfer_ds,
            epochs=3,
            verbose=1,
        )

        finetune_train_acc = float(history_finetune.history["accuracy"][-1])
        finetune_val_acc = float(history_finetune.history["val_accuracy"][-1])

        print("Fine-tune train accuracy:", finetune_train_acc)
        print("Fine-tune val accuracy  :", finetune_val_acc)
        """
    ),
    code_cell(
        """
        plt.figure(figsize=(8, 5))
        plt.plot(history_finetune.history["accuracy"], marker="o", label="Train Accuracy")
        plt.plot(history_finetune.history["val_accuracy"], marker="s", label="Val Accuracy")
        plt.title("Task 4: Fine-Tuning Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        task4_curve_path = os.path.join(PLOTS_DIR, "task4_finetune_accuracy.png")
        plt.savefig(task4_curve_path, dpi=130, bbox_inches="tight")
        plt.show()
        print("Saved:", task4_curve_path)
        """
    ),
    code_cell(
        """
        final_test_loss, final_test_acc = transfer_model.evaluate(test_transfer_ds, verbose=0)

        y_test_pred_probs = transfer_model.predict(test_transfer_ds, verbose=0)
        y_test_pred = np.argmax(y_test_pred_probs, axis=1)
        cm = confusion_matrix(y_test, y_test_pred)
        report_text = classification_report(y_test, y_test_pred, target_names=WASTE_LABELS, digits=4)

        print("Final test accuracy:", final_test_acc)
        print("Final test loss    :", final_test_loss)
        print("\\nClassification report:\\n")
        print(report_text)

        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(NUM_CLASSES))
        ax.set_yticks(range(NUM_CLASSES))
        ax.set_xticklabels(WASTE_LABELS)
        ax.set_yticklabels(WASTE_LABELS)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title("Final Confusion Matrix")
        for i in range(NUM_CLASSES):
            for j in range(NUM_CLASSES):
                ax.text(j, i, int(cm[i, j]), ha="center", va="center", color="black")
        plt.colorbar(im)
        plt.tight_layout()
        cm_path = os.path.join(PLOTS_DIR, "task4_confusion_matrix.png")
        plt.savefig(cm_path, dpi=130, bbox_inches="tight")
        plt.show()
        print("Saved:", cm_path)
        """
    ),
    code_cell(
        """
        validation_improved = finetune_val_acc > transfer_val_acc
        improvement_text = (
            f"Yes. Validation accuracy increased from {transfer_val_acc:.4f} to {finetune_val_acc:.4f}."
            if validation_improved else
            f"No. Validation accuracy changed from {transfer_val_acc:.4f} to {finetune_val_acc:.4f}, so there was no improvement in this run."
        )

        fine_tuning_lr_reason = (
            "The learning rate is reduced during fine-tuning so pretrained ImageNet features are adjusted gently. "
            "A large learning rate could overwrite useful weights too quickly and hurt the knowledge already learned by MobileNetV2."
        )

        print("Answer 1:", improvement_text)
        print("Answer 2:", fine_tuning_lr_reason)
        """
    ),
    md_cell(
        """
        ## Generate the Final Report

        The next cell writes:
        - `Lab_Report_OpenEnded.md`
        - `Lab_Report_OpenEnded.html`
        - `metrics_summary.json`
        """
    ),
    code_cell(
        """
        metrics_summary = {
            "baseline_train_acc": baseline_train_acc,
            "baseline_val_acc": baseline_val_acc,
            "baseline_gap": baseline_gap,
            "modified_train_acc": modified_train_acc,
            "modified_val_acc": modified_val_acc,
            "transfer_train_acc": transfer_train_acc,
            "transfer_val_acc": transfer_val_acc,
            "finetune_train_acc": finetune_train_acc,
            "finetune_val_acc": finetune_val_acc,
            "final_test_acc": float(final_test_acc),
            "final_test_loss": float(final_test_loss),
        }

        metrics_path = os.path.join(BASE_DIR, "metrics_summary.json")
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics_summary, f, indent=2)

        report_md = f\"\"\"# Open Ended Lab Report: Real-Time Waste Classification

        ---

        **Course Code:** COMP-341L  
        **Course Name:** Artificial Neural Networks Lab  
        **Lab Title:** Open Ended Lab - Waste Classification  
        **Date:** {datetime.now().strftime('%B %d, %Y')}  
        **Name:** {STUDENT_NAME}  
        **Roll Number:** {STUDENT_ROLL}  
        **Section:** {STUDENT_SECTION}

        ---

        ## Objective

        Diagnose overfitting in a CNN, improve validation performance through parameter tuning, apply transfer learning with MobileNetV2, and evaluate the effect of fine-tuning with a lower learning rate.

        ## Dataset

        This notebook uses a **CIFAR-10 subset as a proxy dataset**, which is allowed by the lab manual.  
        The three selected classes are mapped to the required labels:

        - `airplane` -> `Plastic`
        - `automobile` -> `Paper`
        - `ship` -> `Metal`

        The dataset was kept intentionally small to make overfitting easier to observe in the baseline CNN.

        ![Task 0 Sample Images](plots/task0_sample_images.png)

        ## Task 1: Overfitting Diagnosis

        - Baseline train accuracy: **{baseline_train_acc:.4f}**
        - Baseline validation accuracy: **{baseline_val_acc:.4f}**
        - Generalization gap: **{baseline_gap:.4f}**

        **Decision:** {overfitting_decision}

        {overfitting_reason}

        {overfitting_reason_2}

        ![Task 1 Accuracy Curve](plots/task1_accuracy_curve.png)

        ## Task 2: Parameter Modification

        The following modifications were applied:

        - Added Batch Normalization
        - Added Dropout
        - Reduced Dense units from 128 to 64
        - Reduced learning rate to 0.0003

        | Model Version | Train Acc | Val Acc |
        |---|---:|---:|
        | Original | {baseline_train_acc:.4f} | {baseline_val_acc:.4f} |
        | Modified | {modified_train_acc:.4f} | {modified_val_acc:.4f} |

        The modified model changed validation accuracy by **{(modified_val_acc - baseline_val_acc):.4f}** compared with the original baseline.

        ![Task 2 Validation Comparison](plots/task2_validation_comparison.png)

        ## Task 3: Transfer Learning Enhancement

        MobileNetV2 was used with `include_top=False`, frozen base layers, and a new classification head.

        - Transfer-learning train accuracy after 5 epochs: **{transfer_train_acc:.4f}**
        - Transfer-learning validation accuracy after 5 epochs: **{transfer_val_acc:.4f}**

        ![Task 3 Transfer Accuracy](plots/task3_transfer_accuracy.png)

        ## Task 4: Fine-Tuning Decision

        The last 10 layers of MobileNetV2 were unfrozen and the learning rate was reduced to `1e-5` before training for 3 more epochs.

        - Fine-tuned train accuracy: **{finetune_train_acc:.4f}**
        - Fine-tuned validation accuracy: **{finetune_val_acc:.4f}**
        - Final test accuracy: **{final_test_acc:.4f}**
        - Final test loss: **{final_test_loss:.4f}**

        **Answer 1: Did validation improve?**  
        {improvement_text}

        **Answer 2: Why reduce learning rate during fine-tuning?**  
        {fine_tuning_lr_reason}

        ![Task 4 Fine-Tuning Accuracy](plots/task4_finetune_accuracy.png)

        ![Task 4 Confusion Matrix](plots/task4_confusion_matrix.png)

        ## Final Classification Report

        ```text
        {report_text.strip()}
        ```

        ## Conclusion

        The baseline CNN was first trained for 10 epochs to diagnose overfitting. After observing the train-validation gap, the network was improved with regularization and normalization changes. Transfer learning with MobileNetV2 provided a stronger starting point than training from scratch, and fine-tuning the last 10 layers with a smaller learning rate allowed the pretrained features to adapt more safely to the task. Overall, the open-ended lab demonstrates how model capacity, regularization, and transfer learning affect generalization performance.
        \"\"\"

        report_html = f\"\"\"<!DOCTYPE html>
        <html lang="en">
        <head>
          <meta charset="utf-8">
          <title>Open Ended Lab Report - Ali Hamza</title>
          <style>
            body {{
              font-family: Arial, sans-serif;
              max-width: 900px;
              margin: 40px auto;
              line-height: 1.6;
              color: #222;
              padding: 0 20px;
            }}
            h1, h2, h3 {{ color: #111; }}
            table {{
              border-collapse: collapse;
              width: 100%;
              margin: 16px 0;
            }}
            th, td {{
              border: 1px solid #ccc;
              padding: 8px 10px;
              text-align: left;
            }}
            th {{ background: #f3f3f3; }}
            code, pre {{
              background: #f7f7f7;
              border-radius: 6px;
            }}
            pre {{
              padding: 12px;
              overflow-x: auto;
            }}
            img {{
              max-width: 100%;
              margin: 10px 0 24px 0;
              border: 1px solid #ddd;
            }}
          </style>
        </head>
        <body>
          <h1>Open Ended Lab Report: Real-Time Waste Classification</h1>
          <p><strong>Course Code:</strong> COMP-341L<br>
          <strong>Course Name:</strong> Artificial Neural Networks Lab<br>
          <strong>Lab Title:</strong> Open Ended Lab - Waste Classification<br>
          <strong>Date:</strong> {datetime.now().strftime('%B %d, %Y')}<br>
          <strong>Name:</strong> {STUDENT_NAME}<br>
          <strong>Roll Number:</strong> {STUDENT_ROLL}<br>
          <strong>Section:</strong> {STUDENT_SECTION}</p>

          <h2>Objective</h2>
          <p>Diagnose overfitting in a CNN, improve validation performance through parameter tuning, apply transfer learning with MobileNetV2, and evaluate the effect of fine-tuning with a lower learning rate.</p>

          <h2>Dataset</h2>
          <p>This report uses a CIFAR-10 subset as a proxy dataset, which is allowed by the manual. The mapping is airplane -&gt; Plastic, automobile -&gt; Paper, and ship -&gt; Metal.</p>
          <img src="plots/task0_sample_images.png" alt="Task 0 Sample Images">

          <h2>Task 1: Overfitting Diagnosis</h2>
          <p><strong>Baseline train accuracy:</strong> {baseline_train_acc:.4f}<br>
          <strong>Baseline validation accuracy:</strong> {baseline_val_acc:.4f}<br>
          <strong>Generalization gap:</strong> {baseline_gap:.4f}</p>
          <p><strong>Decision:</strong> {overfitting_decision}</p>
          <p>{overfitting_reason}</p>
          <p>{overfitting_reason_2}</p>
          <img src="plots/task1_accuracy_curve.png" alt="Task 1 Accuracy Curve">

          <h2>Task 2: Parameter Modification</h2>
          <ul>
            <li>Added Batch Normalization</li>
            <li>Added Dropout</li>
            <li>Reduced Dense units from 128 to 64</li>
            <li>Reduced learning rate to 0.0003</li>
          </ul>
          <table>
            <tr><th>Model Version</th><th>Train Acc</th><th>Val Acc</th></tr>
            <tr><td>Original</td><td>{baseline_train_acc:.4f}</td><td>{baseline_val_acc:.4f}</td></tr>
            <tr><td>Modified</td><td>{modified_train_acc:.4f}</td><td>{modified_val_acc:.4f}</td></tr>
          </table>
          <p>The modified model changed validation accuracy by <strong>{(modified_val_acc - baseline_val_acc):.4f}</strong> compared with the baseline.</p>
          <img src="plots/task2_validation_comparison.png" alt="Task 2 Validation Comparison">

          <h2>Task 3: Transfer Learning Enhancement</h2>
          <p>MobileNetV2 was used with include_top=False, frozen base layers, and a custom classification head.</p>
          <p><strong>Transfer train accuracy:</strong> {transfer_train_acc:.4f}<br>
          <strong>Transfer validation accuracy:</strong> {transfer_val_acc:.4f}</p>
          <img src="plots/task3_transfer_accuracy.png" alt="Task 3 Transfer Accuracy">

          <h2>Task 4: Fine-Tuning Decision</h2>
          <p>The last 10 layers of MobileNetV2 were unfrozen and the learning rate was reduced to 1e-5.</p>
          <p><strong>Fine-tuned train accuracy:</strong> {finetune_train_acc:.4f}<br>
          <strong>Fine-tuned validation accuracy:</strong> {finetune_val_acc:.4f}<br>
          <strong>Final test accuracy:</strong> {final_test_acc:.4f}<br>
          <strong>Final test loss:</strong> {final_test_loss:.4f}</p>
          <p><strong>Did validation improve?</strong> {improvement_text}</p>
          <p><strong>Why reduce learning rate during fine-tuning?</strong> {fine_tuning_lr_reason}</p>
          <img src="plots/task4_finetune_accuracy.png" alt="Task 4 Fine-Tuning Accuracy">
          <img src="plots/task4_confusion_matrix.png" alt="Task 4 Confusion Matrix">

          <h2>Final Classification Report</h2>
          <pre>{report_text.strip()}</pre>

          <h2>Conclusion</h2>
          <p>The open-ended lab shows that a simple CNN can overfit on a small image dataset, but regularization and better hyperparameters improve generalization. Transfer learning gives a stronger starting point than training from scratch, and fine-tuning with a small learning rate helps adapt pretrained features without damaging them too aggressively.</p>
        </body>
        </html>
        \"\"\"

        md_path = os.path.join(BASE_DIR, "Lab_Report_OpenEnded.md")
        html_path = os.path.join(BASE_DIR, "Lab_Report_OpenEnded.html")

        with open(md_path, "w", encoding="utf-8") as f:
            f.write(report_md)
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(report_html)

        print("Saved:", metrics_path)
        print("Saved:", md_path)
        print("Saved:", html_path)
        """
    ),
    md_cell(
        """
        ## Submission Note

        After all cells finish:
        1. Open the generated `Lab_Report_OpenEnded.html`
        2. Use browser **Print -> Save as PDF**
        3. Rename the PDF according to your course naming format
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
        # Open Ended Lab Files

        This folder now contains a **Google Colab-ready notebook** for the open-ended lab:

        - `open_ended_lab_waste_classification_colab.ipynb`
        - `build_open_ended_lab_notebook.py`

        ## What the notebook does

        1. Loads a **CIFAR-10 subset** as allowed by the manual
        2. Trains the baseline CNN for 10 epochs
        3. Diagnoses overfitting using training vs validation accuracy
        4. Modifies at least two parameters and compares performance
        5. Replaces the CNN with **MobileNetV2**
        6. Fine-tunes the last 10 layers with a smaller learning rate
        7. Saves plots, metrics, Markdown report, and HTML report

        ## How to use in Google Colab

        1. Open Google Colab
        2. Upload `open_ended_lab_waste_classification_colab.ipynb`
        3. Run all cells in order
        4. The notebook will save outputs to:
           `/content/drive/MyDrive/COMP-341L/Open Ended Lab/Ali Hamza's Lab`
        5. Export `Lab_Report_OpenEnded.html` to PDF if needed
        """
    ).strip()
    + "\n",
    encoding="utf-8",
)

print(f"Created {NOTEBOOK_PATH}")
print(f"Created {README_PATH}")
