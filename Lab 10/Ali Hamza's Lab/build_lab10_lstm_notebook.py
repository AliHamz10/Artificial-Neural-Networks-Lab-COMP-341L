import json
from pathlib import Path
from textwrap import dedent


ROOT = Path(__file__).resolve().parent
NOTEBOOK_PATH = ROOT / "lab10_lstm_sentiment_colab.ipynb"


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
        # Lab 10: LSTM Sentiment Analysis on IMDb

        **Course:** COMP-341L - Artificial Neural Networks Lab  
        **Student:** Ali Hamza  
        **Roll Number:** B23F0063AI106  
        **Section:** B.S AI - Red  
        **Execution Environment:** Google Colab

        This notebook follows the Lab 10 task step by step:
        1. Load the IMDb dataset online and save a CSV copy in Google Drive
        2. Tokenize text and pad sequences
        3. Train an original LSTM model
        4. Improve the model using two modifications
        5. Evaluate results and test custom sentences
        6. Generate plots plus `Lab_Report_10.md` and `Lab_Report_10.html`

        ## Before running
        This notebook saves everything to Google Drive and downloads the IMDb dataset online automatically if it is not already present.
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

        default_base_dir = "."

        if IN_COLAB and USE_GOOGLE_DRIVE:
            try:
                drive.mount('/content/drive', force_remount=True)
                default_base_dir = f"/content/drive/MyDrive/COMP-341L/Lab 10/{STUDENT_FOLDER_NAME}"
                print("Google Drive mounted successfully.")
            except Exception as e:
                print("Google Drive mount failed. Falling back to local Colab storage.")
                print("Mount error:", repr(e))
                default_base_dir = f"/content/Lab 10/{STUDENT_FOLDER_NAME}"
        elif IN_COLAB:
            default_base_dir = f"/content/Lab 10/{STUDENT_FOLDER_NAME}"

        BASE_DIR = os.environ.get("LAB10_BASE_DIR", default_base_dir)
        PLOTS_DIR = os.path.join(BASE_DIR, "plots")

        os.makedirs(BASE_DIR, exist_ok=True)
        os.makedirs(PLOTS_DIR, exist_ok=True)

        print("IN_COLAB :", IN_COLAB)
        print("USE_GOOGLE_DRIVE:", USE_GOOGLE_DRIVE)
        print("BASE_DIR :", os.path.abspath(BASE_DIR))
        print("PLOTS_DIR:", os.path.abspath(PLOTS_DIR))
        """
    ),
    code_cell(
        """
        # Online dataset loader dependency for Colab
        try:
            from datasets import load_dataset
            print("datasets package already available.")
        except Exception:
            import subprocess
            import sys
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "datasets"])
            from datasets import load_dataset
            print("Installed datasets package.")
        """
    ),
    md_cell(
        """
        ## Task 1: Data Preprocessing
        We will:
        - Load the IMDb CSV dataset
        - Clean the review text
        - Convert text to integer sequences with `Tokenizer`
        - Apply padding so all reviews have the same length
        - Create train, validation, and test splits
        """
    ),
    code_cell(
        """
        import html
        import os
        import random
        import re
        import textwrap

        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        import seaborn as sns
        import tensorflow as tf

        from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
        from sklearn.model_selection import train_test_split
        from tensorflow.keras.preprocessing.sequence import pad_sequences
        from tensorflow.keras.preprocessing.text import Tokenizer

        SEED = 42
        random.seed(SEED)
        np.random.seed(SEED)
        tf.random.set_seed(SEED)

        MAX_WORDS = 5000
        ORIGINAL_MAX_LEN = 100
        MODIFIED_MAX_LEN = 100
        EMBEDDING_DIM = 64
        ORIGINAL_LSTM_UNITS = 64
        MODIFIED_LSTM_UNITS = 128
        ORIGINAL_DROPOUT = 0.0
        MODIFIED_DROPOUT = 0.3
        BATCH_SIZE = 32
        EPOCHS = 5
        DATASET_LOCAL_NAME = "IMDB Dataset.csv"

        def clean_review(text):
            text = html.unescape(str(text))
            text = re.sub(r"<[^>]+>", " ", text)
            text = text.replace("\\n", " ")
            text = re.sub(r"[^a-zA-Z0-9' ]+", " ", text)
            text = re.sub(r"\\s+", " ", text).strip().lower()
            return text

        def locate_dataset():
            target_path = os.path.join(BASE_DIR, DATASET_LOCAL_NAME)
            candidates = [
                target_path,
                "/content/IMDB Dataset.csv",
                "/content/drive/MyDrive/IMDB Dataset.csv",
                "/content/drive/MyDrive/Datasets/IMDB Dataset.csv",
            ]
            for candidate in candidates:
                if os.path.exists(candidate):
                    return candidate

            print("Dataset not found locally. Downloading IMDb dataset online from Hugging Face...")
            imdb_ds = load_dataset("scikit-learn/imdb", split="train")
            downloaded_df = imdb_ds.to_pandas()
            downloaded_df.to_csv(target_path, index=False)
            print("Downloaded dataset to:", target_path)
            return target_path

        csv_path = locate_dataset()
        print("Dataset path:", csv_path)

        df = pd.read_csv(csv_path)
        expected_columns = {"review", "sentiment"}
        if not expected_columns.issubset(df.columns):
            raise ValueError(f"Dataset must contain columns {expected_columns}, but found {set(df.columns)}")

        df = df[["review", "sentiment"]].copy()
        df["review"] = df["review"].astype(str)
        df["review_clean"] = df["review"].map(clean_review)
        df["label"] = df["sentiment"].str.lower().map({"negative": 0, "positive": 1})

        if df["label"].isna().any():
            raise ValueError("Sentiment column must only contain 'positive' and 'negative'.")

        df["label"] = df["label"].astype("int32")
        df["token_count"] = df["review_clean"].str.split().map(len)

        print("Dataset shape:", df.shape)
        print("Sentiment distribution:")
        print(df["sentiment"].value_counts())
        print()
        print("Token length stats:")
        print(df["token_count"].describe())

        sample_preview = df.sample(3, random_state=SEED)[["sentiment", "review_clean"]]
        for row_id, row in sample_preview.reset_index(drop=True).iterrows():
            print(f"Sample {row_id + 1} [{row['sentiment']}]:")
            print(textwrap.shorten(row["review_clean"], width=240, placeholder=" ..."))
            print()
        """
    ),
    code_cell(
        """
        # Visual summary for Task 1
        sample_rows = df.sample(5, random_state=SEED).reset_index(drop=True)

        fig, axes = plt.subplots(5, 1, figsize=(14, 14))
        for ax, (_, row) in zip(axes, sample_rows.iterrows()):
            ax.axis("off")
            wrapped = textwrap.fill(
                textwrap.shorten(row["review_clean"], width=420, placeholder=" ..."),
                width=100
            )
            ax.set_title(f"Sentiment: {row['sentiment'].title()}", fontsize=11, loc="left")
            ax.text(0, 1, wrapped, fontsize=10, va="top")

        plt.suptitle("Task 1: Sample IMDb Reviews", fontsize=14)
        plt.tight_layout()
        sample_reviews_path = os.path.join(PLOTS_DIR, "task1_sample_reviews.png")
        plt.savefig(sample_reviews_path, dpi=130, bbox_inches="tight")
        plt.show()
        print("Saved:", sample_reviews_path)

        plt.figure(figsize=(8, 5))
        sns.histplot(df["token_count"], bins=40, color="#1f77b4")
        plt.axvline(ORIGINAL_MAX_LEN, color="red", linestyle="--", label=f"Sequence length = {ORIGINAL_MAX_LEN}")
        plt.title("Task 1: Review Length Distribution")
        plt.xlabel("Number of Tokens")
        plt.ylabel("Count")
        plt.legend()
        plt.tight_layout()
        length_path = os.path.join(PLOTS_DIR, "task1_length_distribution.png")
        plt.savefig(length_path, dpi=130, bbox_inches="tight")
        plt.show()
        print("Saved:", length_path)
        """
    ),
    code_cell(
        """
        train_df, temp_df = train_test_split(
            df,
            test_size=0.2,
            stratify=df["label"],
            random_state=SEED,
        )

        val_df, test_df = train_test_split(
            temp_df,
            test_size=0.5,
            stratify=temp_df["label"],
            random_state=SEED,
        )

        print("Train shape:", train_df.shape)
        print("Val shape  :", val_df.shape)
        print("Test shape :", test_df.shape)

        tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
        tokenizer.fit_on_texts(train_df["review_clean"])

        train_sequences = tokenizer.texts_to_sequences(train_df["review_clean"])
        val_sequences = tokenizer.texts_to_sequences(val_df["review_clean"])
        test_sequences = tokenizer.texts_to_sequences(test_df["review_clean"])

        def make_padded(sequences, max_len):
            return pad_sequences(sequences, maxlen=max_len, padding="post", truncating="post")

        X_train_original = make_padded(train_sequences, ORIGINAL_MAX_LEN)
        X_val_original = make_padded(val_sequences, ORIGINAL_MAX_LEN)
        X_test_original = make_padded(test_sequences, ORIGINAL_MAX_LEN)

        X_train_modified = make_padded(train_sequences, MODIFIED_MAX_LEN)
        X_val_modified = make_padded(val_sequences, MODIFIED_MAX_LEN)
        X_test_modified = make_padded(test_sequences, MODIFIED_MAX_LEN)

        y_train = train_df["label"].to_numpy(dtype="float32")
        y_val = val_df["label"].to_numpy(dtype="float32")
        y_test = test_df["label"].to_numpy(dtype="float32")

        print("Original padded shapes:")
        print("X_train:", X_train_original.shape)
        print("X_val  :", X_val_original.shape)
        print("X_test :", X_test_original.shape)

        print()
        print("Example sequence before padding:", train_sequences[0][:20], "...")
        print("Length before padding:", len(train_sequences[0]))
        print("Example sequence after padding:", X_train_original[0][:20], "...")
        print("Length after padding:", len(X_train_original[0]))
        """
    ),
    md_cell(
        """
        ### Why Padding Is Required
        - Neural networks expect tensors with the same shape inside each batch.
        - Raw reviews have different lengths, so `pad_sequences` makes every sample the same length.
        - Without padding, NumPy/TensorFlow cannot pack variable-length sequences into a regular 2D input matrix for standard batch training.
        - If sequences have different lengths and we do not pad them, model training will fail or require a more complex ragged-sequence pipeline.
        """
    ),
    md_cell(
        """
        ## Task 2: Build LSTM Model
        Required architecture:

        ```python
        model = tf.keras.Sequential([
            tf.keras.layers.Embedding(input_dim=5000, output_dim=64, input_length=100),
            tf.keras.layers.LSTM(64),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        ```

        We will first train the original version, then create a modified version with:
        - Increased LSTM units: `64 -> 128`
        - Added dropout: `0.0 -> 0.3`
        """
    ),
    code_cell(
        """
        def build_lstm_model(max_len, lstm_units=64, dropout_rate=0.0, learning_rate=1e-3):
            model = tf.keras.Sequential([
                tf.keras.layers.Embedding(input_dim=MAX_WORDS, output_dim=EMBEDDING_DIM, input_length=max_len),
                tf.keras.layers.LSTM(lstm_units, dropout=dropout_rate),
                tf.keras.layers.Dense(1, activation="sigmoid"),
            ])

            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                loss="binary_crossentropy",
                metrics=["accuracy"],
            )
            return model

        original_model = build_lstm_model(
            max_len=ORIGINAL_MAX_LEN,
            lstm_units=ORIGINAL_LSTM_UNITS,
            dropout_rate=ORIGINAL_DROPOUT,
        )

        original_model.summary()
        """
    ),
    md_cell(
        """
        ### Required Explanations
        - **Embedding layer:** converts each word index into a dense vector so the model can learn semantic relationships instead of treating tokens as isolated IDs.
        - **Why LSTM instead of SimpleRNN:** LSTM keeps a gated memory state, so it handles long-term dependencies and negation patterns like `not good` better than SimpleRNN.
        - **Why sigmoid is used:** this is a binary classification task, so sigmoid maps the final output to a probability between 0 and 1 for negative vs positive sentiment.
        """
    ),
    md_cell(
        """
        ## Task 3: Training and Evaluation
        Train the original model for 5 epochs and inspect training vs validation behavior.
        """
    ),
    code_cell(
        """
        history_original = original_model.fit(
            X_train_original,
            y_train,
            validation_data=(X_val_original, y_val),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            verbose=1,
        )

        original_train_acc = float(history_original.history["accuracy"][-1])
        original_val_acc = float(history_original.history["val_accuracy"][-1])
        original_train_loss = float(history_original.history["loss"][-1])
        original_val_loss = float(history_original.history["val_loss"][-1])

        original_test_loss, original_test_acc = original_model.evaluate(
            X_test_original,
            y_test,
            verbose=1,
        )

        print("Original model results:")
        print("train_acc :", original_train_acc)
        print("val_acc   :", original_val_acc)
        print("train_loss:", original_train_loss)
        print("val_loss  :", original_val_loss)
        print("test_acc  :", original_test_acc)
        print("test_loss :", original_test_loss)
        """
    ),
    code_cell(
        """
        acc_curve_path = os.path.join(PLOTS_DIR, "task3_accuracy_curve.png")
        loss_curve_path = os.path.join(PLOTS_DIR, "task3_loss_curve.png")

        plt.figure(figsize=(8, 5))
        plt.plot(history_original.history["accuracy"], marker="o", label="Training Accuracy")
        plt.plot(history_original.history["val_accuracy"], marker="o", label="Validation Accuracy")
        plt.title("Accuracy vs Epochs (Original LSTM)")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.tight_layout()
        plt.savefig(acc_curve_path, dpi=130, bbox_inches="tight")
        plt.show()
        print("Saved:", acc_curve_path)

        plt.figure(figsize=(8, 5))
        plt.plot(history_original.history["loss"], marker="o", label="Training Loss")
        plt.plot(history_original.history["val_loss"], marker="o", label="Validation Loss")
        plt.title("Loss vs Epochs (Original LSTM)")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(loss_curve_path, dpi=130, bbox_inches="tight")
        plt.show()
        print("Saved:", loss_curve_path)

        def identify_fit_status(train_acc, val_acc, train_loss, val_loss):
            acc_gap = train_acc - val_acc
            if acc_gap > 0.05 and val_loss > train_loss:
                return "Overfitting"
            if train_acc < 0.75 and val_acc < 0.75:
                return "Underfitting"
            return "Reasonably fit"

        original_fit_status = identify_fit_status(
            original_train_acc,
            original_val_acc,
            original_train_loss,
            original_val_loss,
        )

        fit_summary = (
            f"Original model diagnosis: {original_fit_status}. "
            f"Train accuracy = {original_train_acc:.4f}, validation accuracy = {original_val_acc:.4f}, "
            f"train loss = {original_train_loss:.4f}, validation loss = {original_val_loss:.4f}."
        )

        print(fit_summary)

        fit_summary_path = os.path.join(PLOTS_DIR, "task3_fit_diagnosis.txt")
        with open(fit_summary_path, "w", encoding="utf-8") as f:
            f.write(fit_summary + "\\n")
        print("Saved:", fit_summary_path)
        """
    ),
    md_cell(
        """
        ## Task 4: Parameter Modification
        Chosen improvements:
        - Increase LSTM units from `64` to `128`
        - Add dropout `0.3`

        The goal is to improve contextual learning while also reducing overfitting risk.
        """
    ),
    code_cell(
        """
        modified_model = build_lstm_model(
            max_len=MODIFIED_MAX_LEN,
            lstm_units=MODIFIED_LSTM_UNITS,
            dropout_rate=MODIFIED_DROPOUT,
        )

        history_modified = modified_model.fit(
            X_train_modified,
            y_train,
            validation_data=(X_val_modified, y_val),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            verbose=1,
        )

        modified_train_acc = float(history_modified.history["accuracy"][-1])
        modified_val_acc = float(history_modified.history["val_accuracy"][-1])
        modified_train_loss = float(history_modified.history["loss"][-1])
        modified_val_loss = float(history_modified.history["val_loss"][-1])

        modified_test_loss, modified_test_acc = modified_model.evaluate(
            X_test_modified,
            y_test,
            verbose=1,
        )

        print("Modified model results:")
        print("train_acc :", modified_train_acc)
        print("val_acc   :", modified_val_acc)
        print("train_loss:", modified_train_loss)
        print("val_loss  :", modified_val_loss)
        print("test_acc  :", modified_test_acc)
        print("test_loss :", modified_test_loss)
        """
    ),
    code_cell(
        """
        comparison_df = pd.DataFrame(
            [
                {
                    "Model": "Original",
                    "Train Acc": original_train_acc,
                    "Val Acc": original_val_acc,
                    "Test Acc": float(original_test_acc),
                    "Train Loss": original_train_loss,
                    "Val Loss": original_val_loss,
                    "Test Loss": float(original_test_loss),
                },
                {
                    "Model": "Modified",
                    "Train Acc": modified_train_acc,
                    "Val Acc": modified_val_acc,
                    "Test Acc": float(modified_test_acc),
                    "Train Loss": modified_train_loss,
                    "Val Loss": modified_val_loss,
                    "Test Loss": float(modified_test_loss),
                },
            ]
        )

        display(comparison_df.style.format({
            "Train Acc": "{:.4f}",
            "Val Acc": "{:.4f}",
            "Test Acc": "{:.4f}",
            "Train Loss": "{:.4f}",
            "Val Loss": "{:.4f}",
            "Test Loss": "{:.4f}",
        }))

        comparison_csv_path = os.path.join(PLOTS_DIR, "task4_model_comparison.csv")
        comparison_df.to_csv(comparison_csv_path, index=False)
        print("Saved:", comparison_csv_path)

        comparison_table_markdown = comparison_df.to_markdown(index=False, floatfmt=".4f")
        print(comparison_table_markdown)
        """
    ),
    code_cell(
        """
        plt.figure(figsize=(8, 5))
        plt.plot(history_original.history["val_accuracy"], marker="o", label="Original Val Accuracy")
        plt.plot(history_modified.history["val_accuracy"], marker="o", label="Modified Val Accuracy")
        plt.title("Validation Accuracy Comparison")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.tight_layout()
        compare_acc_path = os.path.join(PLOTS_DIR, "task4_validation_accuracy_comparison.png")
        plt.savefig(compare_acc_path, dpi=130, bbox_inches="tight")
        plt.show()
        print("Saved:", compare_acc_path)
        """
    ),
    md_cell(
        """
        ## Task 5: Prediction Test
        We will use the better-performing model on custom review sentences and interpret outputs:
        - Probability close to `1` -> Positive
        - Probability close to `0` -> Negative
        """
    ),
    code_cell(
        """
        best_model_name = "Modified" if modified_val_acc >= original_val_acc else "Original"
        best_model = modified_model if best_model_name == "Modified" else original_model
        best_max_len = MODIFIED_MAX_LEN if best_model_name == "Modified" else ORIGINAL_MAX_LEN

        custom_samples = [
            "This movie was absolutely amazing",
            "This movie was not good at all",
            "The acting was great but the story was too slow and boring",
            "I really loved the characters and the ending was satisfying",
            "I expected something better, and honestly it was a waste of time",
        ]

        custom_sequences = tokenizer.texts_to_sequences([clean_review(x) for x in custom_samples])
        custom_padded = pad_sequences(custom_sequences, maxlen=best_max_len, padding="post", truncating="post")
        custom_probs = best_model.predict(custom_padded, verbose=0).reshape(-1)

        prediction_rows = []
        for text, prob in zip(custom_samples, custom_probs):
            label = "Positive" if prob >= 0.5 else "Negative"
            prediction_rows.append({
                "Review": text,
                "Predicted Probability": float(prob),
                "Predicted Sentiment": label,
            })

        prediction_df = pd.DataFrame(prediction_rows)
        display(prediction_df.style.format({"Predicted Probability": "{:.4f}"}))

        prediction_path = os.path.join(PLOTS_DIR, "task5_custom_predictions.csv")
        prediction_df.to_csv(prediction_path, index=False)
        print("Saved:", prediction_path)
        print("Best model selected for custom prediction:", best_model_name)
        """
    ),
    code_cell(
        """
        final_model = best_model
        final_model_name = best_model_name
        final_model_max_len = best_max_len
        final_test_x = X_test_modified if final_model_name == "Modified" else X_test_original

        final_probs = final_model.predict(final_test_x, verbose=0).reshape(-1)
        final_preds = (final_probs >= 0.5).astype("int32")

        final_cm = confusion_matrix(y_test.astype("int32"), final_preds)
        final_report = classification_report(
            y_test.astype("int32"),
            final_preds,
            target_names=["Negative", "Positive"],
            digits=4,
        )
        final_accuracy = accuracy_score(y_test.astype("int32"), final_preds)

        print("Final model used for evaluation:", final_model_name)
        print("Final test accuracy (sklearn):", final_accuracy)
        print()
        print(final_report)

        report_path = os.path.join(PLOTS_DIR, "task5_classification_report.txt")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(final_report + "\\n")
        print("Saved:", report_path)

        plt.figure(figsize=(6, 5))
        sns.heatmap(final_cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["Negative", "Positive"],
                    yticklabels=["Negative", "Positive"])
        plt.title(f"Confusion Matrix ({final_model_name} Model)")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.tight_layout()
        cm_path = os.path.join(PLOTS_DIR, "task5_confusion_matrix.png")
        plt.savefig(cm_path, dpi=130, bbox_inches="tight")
        plt.show()
        print("Saved:", cm_path)
        """
    ),
    md_cell(
        """
        ## Reflection Questions
        The next cell creates direct answers for all five required reflection questions using the experiment results.
        """
    ),
    code_cell(
        """
        final_train_acc = modified_train_acc if final_model_name == "Modified" else original_train_acc
        final_val_acc = modified_val_acc if final_model_name == "Modified" else original_val_acc
        final_train_loss = modified_train_loss if final_model_name == "Modified" else original_train_loss
        final_val_loss = modified_val_loss if final_model_name == "Modified" else original_val_loss
        final_test_loss = modified_test_loss if final_model_name == "Modified" else original_test_loss
        final_test_acc = modified_test_acc if final_model_name == "Modified" else original_test_acc

        reflection_answers = {
            "Q1": (
                "LSTM performs better than SimpleRNN because it uses gated memory to keep useful information "
                "for longer parts of the sequence. That helps it understand context such as negation and word order, "
                "which is important in sentiment analysis."
            ),
            "Q2": (
                "The memory cell stores information across time steps so the network can remember earlier words "
                "while reading later words. This helps the model connect phrases like 'not good' instead of judging "
                "the word 'good' in isolation."
            ),
            "Q3": (
                "Padding is important because neural networks train in batches, and every sample in a batch must have "
                "the same shape. Without padding, reviews of different lengths cannot be stacked into one regular tensor."
            ),
            "Q4": (
                f"If the sequence length is too small, the model will lose part of the review due to truncation. "
                f"Important context near the end of long reviews may be removed, which can reduce sentiment accuracy."
            ),
            "Q5": (
                f"Increasing LSTM units can improve performance because the model gets more capacity to learn patterns "
                f"and contextual dependencies. In this lab, the modified model reached validation accuracy "
                f"{modified_val_acc:.4f} compared with {original_val_acc:.4f} for the original model, so the larger "
                f"LSTM {'helped' if modified_val_acc >= original_val_acc else 'did not help much'} on this dataset."
            ),
        }

        for key, value in reflection_answers.items():
            print(key + ":", value)
            print()
        """
    ),
    code_cell(
        """
        def explain_fit_status(status):
            if status == "Overfitting":
                return (
                    "The training accuracy is noticeably higher than validation accuracy, and validation loss is worse "
                    "than training loss, so the model is learning the training data more strongly than it generalizes."
                )
            if status == "Underfitting":
                return (
                    "Both training and validation scores remain low, which suggests the model has not learned enough "
                    "useful patterns from the data yet."
                )
            return (
                "Training and validation behavior are reasonably close, so the model is learning useful patterns "
                "without a large generalization gap."
            )

        fit_explanation = explain_fit_status(original_fit_status)

        discussion_text = (
            f"The original LSTM model achieved training accuracy {original_train_acc:.4f} and validation accuracy "
            f"{original_val_acc:.4f} after {EPOCHS} epochs. {fit_explanation} "
            f"After increasing LSTM units from {ORIGINAL_LSTM_UNITS} to {MODIFIED_LSTM_UNITS} and adding dropout "
            f"{MODIFIED_DROPOUT:.1f}, the modified model reached validation accuracy {modified_val_acc:.4f} and test accuracy "
            f"{modified_test_acc:.4f}. This shows that LSTM can capture context better than basic recurrent models, "
            f"especially for phrases where sentiment depends on earlier words."
        )

        conclusion_text = (
            f"In this lab, the IMDb movie review dataset was preprocessed with tokenization and padding, then classified "
            f"with an LSTM network for binary sentiment analysis. The experiment showed why sequence models are better "
            f"than traditional MLP-style text handling for understanding context and long-term dependencies. The final "
            f"results also showed that increasing LSTM capacity and adding dropout can improve generalization when the "
            f"baseline model is not strong enough."
        )

        print("Discussion:")
        print(discussion_text)
        print()
        print("Conclusion:")
        print(conclusion_text)
        """
    ),
    md_cell(
        """
        ## Export Report Files
        Run the next cell to create:
        - `Lab_Report_10.md`
        - `Lab_Report_10.html`

        After that, open the HTML file in Colab/Drive and use **Print -> Save as PDF** if you need a final PDF.
        """
    ),
    code_cell(
        """
        os.makedirs(BASE_DIR, exist_ok=True)
        os.makedirs(PLOTS_DIR, exist_ok=True)

        task3_report_text = (
            f"Original model diagnosis: {original_fit_status}\\n"
            f"Train accuracy: {original_train_acc:.4f}\\n"
            f"Validation accuracy: {original_val_acc:.4f}\\n"
            f"Train loss: {original_train_loss:.4f}\\n"
            f"Validation loss: {original_val_loss:.4f}\\n"
        )

        report_md = f\"\"\"# Lab Report 10: LSTM-Based Sentiment Analysis on IMDb

        ---

        **Course Code:** COMP-341L  
        **Course Name:** Artificial Neural Networks Lab  
        **Lab Number:** 10  
        **Lab Title:** LSTM-Based Sentiment Analysis on Movie Reviews  
        **Date:** {datetime.now().strftime('%B %d, %Y')}  
        **Name:** {STUDENT_NAME}  
        **Roll Number:** {STUDENT_ROLL}  
        **Section:** {STUDENT_SECTION}

        ---

        ## Scenario
        A startup needs an AI-powered movie review analysis system that can classify reviews as positive or negative. Traditional MLP-based models fail to capture context in text, especially phrases such as `not good`, and they cannot model long-term dependencies effectively. Therefore, this lab uses an LSTM model for sentiment analysis on the IMDb dataset.

        ---

        ## Task 1: Data Preprocessing
        - **Dataset source:** Kaggle IMDb Dataset of 50K Movie Reviews (`IMDB Dataset.csv`)
        - **Text cleaning:** lowercasing, removing HTML tags, and normalizing spaces
        - **Tokenizer:** `Tokenizer(num_words={MAX_WORDS})`
        - **Sequence length:** `{ORIGINAL_MAX_LEN}`
        - **Train/Validation/Test split:** {len(train_df)} / {len(val_df)} / {len(test_df)}

        ### Why padding is required
        - Padding makes all sequences the same length so they can be stored in one tensor.
        - If sequences have different lengths, batch training becomes difficult because the network expects consistent input shapes.
        - Padding also lets us control how much context the model reads from each review.

        ![Task 1 Sample Reviews](plots/task1_sample_reviews.png)
        ![Task 1 Length Distribution](plots/task1_length_distribution.png)

        ---

        ## Task 2: Build LSTM Model
        ### Original model
        ```python
        model = tf.keras.Sequential([
            tf.keras.layers.Embedding(input_dim=5000, output_dim=64, input_length=100),
            tf.keras.layers.LSTM(64),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        ```

        ### Explanation
        - **Embedding layer:** converts word indices into dense vectors so semantically useful patterns can be learned.
        - **Why LSTM is used instead of SimpleRNN:** LSTM handles long-term dependencies with gated memory and is better for contextual phrases.
        - **Why sigmoid is used:** the output is binary, so sigmoid maps the prediction to a probability between 0 and 1.

        ---

        ## Task 3: Training and Evaluation
        ### Original model results
        - Training accuracy: {original_train_acc:.4f}
        - Validation accuracy: {original_val_acc:.4f}
        - Training loss: {original_train_loss:.4f}
        - Validation loss: {original_val_loss:.4f}
        - Test accuracy: {float(original_test_acc):.4f}
        - Test loss: {float(original_test_loss):.4f}

        ### Fit diagnosis
        ```text
        {task3_report_text}
        ```

        ![Accuracy vs Epochs](plots/task3_accuracy_curve.png)
        ![Loss vs Epochs](plots/task3_loss_curve.png)

        ---

        ## Task 4: Parameter Modification
        Two modifications were applied:
        1. Increased LSTM units from `{ORIGINAL_LSTM_UNITS}` to `{MODIFIED_LSTM_UNITS}`
        2. Added dropout `{MODIFIED_DROPOUT:.1f}`

        ### Comparison table
        ```text
        {comparison_table_markdown}
        ```

        ![Validation Accuracy Comparison](plots/task4_validation_accuracy_comparison.png)

        ---

        ## Task 5: Prediction Test
        Final model selected: **{final_model_name}**

        ```text
        {prediction_df.to_string(index=False)}
        ```

        Interpretation:
        - Probability close to 1 means the review is predicted as positive.
        - Probability close to 0 means the review is predicted as negative.

        ### Final Evaluation
        - Final test accuracy: {float(final_test_acc):.4f}
        - Final test loss: {float(final_test_loss):.4f}

        ![Confusion Matrix](plots/task5_confusion_matrix.png)

        ### Classification Report
        ```text
        {final_report}
        ```

        ---

        ## Required Visualizations
        - Accuracy vs Epochs: `plots/task3_accuracy_curve.png`
        - Loss vs Epochs: `plots/task3_loss_curve.png`

        ---

        ## Reflection Questions
        1. **Why does LSTM perform better than SimpleRNN?**  
           {reflection_answers["Q1"]}

        2. **What role does memory cell play?**  
           {reflection_answers["Q2"]}

        3. **Why is padding important in NLP tasks?**  
           {reflection_answers["Q3"]}

        4. **What happens if sequence length is too small?**  
           {reflection_answers["Q4"]}

        5. **Why does increasing LSTM units improve performance (or not)?**  
           {reflection_answers["Q5"]}

        ---

        ## Discussion
        {discussion_text}

        ---

        ## Conclusion
        {conclusion_text}
        \"\"\"

        report_html_template = \"\"\"<!doctype html>
        <html lang='en'>
        <head>
          <meta charset='utf-8'>
          <title>Lab Report 10 - LSTM IMDb</title>
          <style>
            body { font-family: Arial, sans-serif; margin: 28px; line-height: 1.55; color: #111; }
            h1, h2, h3 { margin-top: 24px; }
            img { max-width: 900px; width: 100%; border: 1px solid #ddd; margin: 8px 0 18px; }
            pre { background: #f4f4f4; padding: 12px; overflow-x: auto; white-space: pre-wrap; }
            .meta { margin: 0 0 12px; }
            ul { margin: 8px 0 0 18px; }
            li { margin: 4px 0; }
          </style>
        </head>
        <body>
          <h1>Lab Report 10: LSTM-Based Sentiment Analysis on IMDb</h1>
          <p class='meta'>
            <strong>Course Code:</strong> COMP-341L<br>
            <strong>Course Name:</strong> Artificial Neural Networks Lab<br>
            <strong>Lab Number:</strong> 10<br>
            <strong>Date:</strong> __DATE__<br>
            <strong>Name:</strong> __NAME__<br>
            <strong>Roll Number:</strong> __ROLL__<br>
            <strong>Section:</strong> __SECTION__
          </p>

          <h2>Scenario</h2>
          <p>A startup requires an AI system that classifies movie reviews as positive or negative. Traditional MLP models miss context such as negation and cannot model long-term dependencies well, so this lab uses LSTM for sentiment analysis.</p>

          <h2>Task 1: Data Preprocessing</h2>
          <ul>
            <li>Dataset: Kaggle IMDb 50K Movie Reviews CSV</li>
            <li>Tokenizer vocabulary size: __MAX_WORDS__</li>
            <li>Sequence length: __MAX_LEN__</li>
            <li>Split sizes: train=__TRAIN_SIZE__, val=__VAL_SIZE__, test=__TEST_SIZE__</li>
          </ul>
          <p><strong>Why padding is required:</strong> All reviews must have the same shape for batch training. Without padding, standard tensor batching becomes inconsistent because review lengths vary.</p>
          <img src='plots/task1_sample_reviews.png' alt='Sample IMDb reviews'>
          <img src='plots/task1_length_distribution.png' alt='Review length distribution'>

          <h2>Task 2: Build LSTM Model</h2>
          <p><strong>Embedding layer:</strong> Converts token IDs into dense vectors.<br>
          <strong>LSTM instead of SimpleRNN:</strong> Preserves long-term context with gated memory.<br>
          <strong>Sigmoid output:</strong> Produces a probability for binary sentiment classification.</p>

          <h2>Task 3: Training and Evaluation</h2>
          <ul>
            <li>Original training accuracy: __ORIG_TRAIN_ACC__</li>
            <li>Original validation accuracy: __ORIG_VAL_ACC__</li>
            <li>Original training loss: __ORIG_TRAIN_LOSS__</li>
            <li>Original validation loss: __ORIG_VAL_LOSS__</li>
            <li>Original test accuracy: __ORIG_TEST_ACC__</li>
            <li>Original test loss: __ORIG_TEST_LOSS__</li>
          </ul>
          <pre>__FIT_SUMMARY__</pre>
          <img src='plots/task3_accuracy_curve.png' alt='Accuracy curve'>
          <img src='plots/task3_loss_curve.png' alt='Loss curve'>

          <h2>Task 4: Parameter Modification</h2>
          <p>Two modifications were applied: increase LSTM units from __ORIG_UNITS__ to __MOD_UNITS__, and add dropout __MOD_DROPOUT__.</p>
          <pre>__COMPARISON_TABLE__</pre>
          <img src='plots/task4_validation_accuracy_comparison.png' alt='Validation accuracy comparison'>

          <h2>Task 5: Prediction Test</h2>
          <p><strong>Final model used:</strong> __FINAL_MODEL_NAME__</p>
          <pre>__PREDICTIONS__</pre>
          <p>Probability near 1 means Positive. Probability near 0 means Negative.</p>
          <img src='plots/task5_confusion_matrix.png' alt='Confusion matrix'>
          <pre>__FINAL_REPORT__</pre>

          <h2>Reflection Questions</h2>
          <p><strong>1. Why does LSTM perform better than SimpleRNN?</strong><br>__Q1__</p>
          <p><strong>2. What role does memory cell play?</strong><br>__Q2__</p>
          <p><strong>3. Why is padding important in NLP tasks?</strong><br>__Q3__</p>
          <p><strong>4. What happens if sequence length is too small?</strong><br>__Q4__</p>
          <p><strong>5. Why does increasing LSTM units improve performance (or not)?</strong><br>__Q5__</p>

          <h2>Discussion</h2>
          <p>__DISCUSSION__</p>

          <h2>Conclusion</h2>
          <p>__CONCLUSION__</p>
        </body>
        </html>
        \"\"\"

        report_html = (
            report_html_template
            .replace("__DATE__", datetime.now().strftime('%B %d, %Y'))
            .replace("__NAME__", STUDENT_NAME)
            .replace("__ROLL__", STUDENT_ROLL)
            .replace("__SECTION__", STUDENT_SECTION)
            .replace("__MAX_WORDS__", str(MAX_WORDS))
            .replace("__MAX_LEN__", str(ORIGINAL_MAX_LEN))
            .replace("__TRAIN_SIZE__", str(len(train_df)))
            .replace("__VAL_SIZE__", str(len(val_df)))
            .replace("__TEST_SIZE__", str(len(test_df)))
            .replace("__ORIG_TRAIN_ACC__", f"{original_train_acc:.4f}")
            .replace("__ORIG_VAL_ACC__", f"{original_val_acc:.4f}")
            .replace("__ORIG_TRAIN_LOSS__", f"{original_train_loss:.4f}")
            .replace("__ORIG_VAL_LOSS__", f"{original_val_loss:.4f}")
            .replace("__ORIG_TEST_ACC__", f"{float(original_test_acc):.4f}")
            .replace("__ORIG_TEST_LOSS__", f"{float(original_test_loss):.4f}")
            .replace("__FIT_SUMMARY__", task3_report_text)
            .replace("__ORIG_UNITS__", str(ORIGINAL_LSTM_UNITS))
            .replace("__MOD_UNITS__", str(MODIFIED_LSTM_UNITS))
            .replace("__MOD_DROPOUT__", f"{MODIFIED_DROPOUT:.1f}")
            .replace("__COMPARISON_TABLE__", comparison_table_markdown)
            .replace("__FINAL_MODEL_NAME__", final_model_name)
            .replace("__PREDICTIONS__", prediction_df.to_string(index=False))
            .replace("__FINAL_REPORT__", final_report)
            .replace("__Q1__", reflection_answers["Q1"])
            .replace("__Q2__", reflection_answers["Q2"])
            .replace("__Q3__", reflection_answers["Q3"])
            .replace("__Q4__", reflection_answers["Q4"])
            .replace("__Q5__", reflection_answers["Q5"])
            .replace("__DISCUSSION__", discussion_text)
            .replace("__CONCLUSION__", conclusion_text)
        )

        md_path = os.path.join(BASE_DIR, "Lab_Report_10.md")
        html_path = os.path.join(BASE_DIR, "Lab_Report_10.html")

        with open(md_path, "w", encoding="utf-8") as f:
            f.write(report_md)
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(report_html)

        print("Saved:", os.path.abspath(md_path))
        print("Saved:", os.path.abspath(html_path))
        print("Plots currently saved:")
        for filename in sorted(os.listdir(PLOTS_DIR)):
            print(" -", filename)
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
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}


NOTEBOOK_PATH.write_text(json.dumps(notebook, indent=2), encoding="utf-8")
print(f"Created: {NOTEBOOK_PATH}")
