from transformers import pipeline
import torch
from nltk.tokenize import sent_tokenize
import nltk
import pandas as pd
import numpy as np
import os
import sys
import pathlib
import traceback

# Resolve paths
folder_path = pathlib.Path(__file__).parent.resolve()
sys.path.append(os.path.join(folder_path, '../'))

from utils import load_subtitles_dataset

# Download required NLTK data
nltk.download('punkt')

class ThemeClassifier():
    def __init__(self, theme_list):
        self.model_name = "facebook/bart-large-mnli"
        self.device = 0 if torch.cuda.is_available() else -1  # Use GPU if available
        self.torch_dtype = torch.float16 if self.device == 0 else torch.float32
        self.theme_list = theme_list
        self.theme_classifier = self.load_model()

    def load_model(self):
        theme_classifier = pipeline(
            "zero-shot-classification",
            model=self.model_name,
            device=self.device,
            torch_dtype=self.torch_dtype
        )
        return theme_classifier

    def get_themes_inference(self, script):
        # Optional truncation for very long inputs
        if len(script) > 5000:
            script = script[:5000]

        script_sentences = sent_tokenize(script)

        # Batch sentences
        sentence_batch_size = 50
        script_batches = [
            " ".join(script_sentences[i:i + sentence_batch_size])
            for i in range(0, len(script_sentences), sentence_batch_size)
        ]

        # Run model
        theme_output = self.theme_classifier(
            script_batches,
            self.theme_list,
            multi_label=True
        )

        # Process output
        themes = {}
        for output in theme_output:
            for label, score in zip(output['labels'], output['scores']):
                themes.setdefault(label, []).append(score)

        themes = {key: np.mean(value) for key, value in themes.items()}
        return themes

    def get_themes(self, dataset_path, save_path=None):
        try:
            print(f"▶ get_themes started with: {dataset_path}")
            if save_path:
                print(f"▶ Will save results to: {save_path}")

            # Return cached results if available
            if save_path is not None and os.path.exists(save_path):
                print("ℹ️ Using cached CSV result.")
                return pd.read_csv(save_path)

            # Load dataset
            df = load_subtitles_dataset(dataset_path)
            print(f"✅ Loaded {len(df)} subtitle rows")

            # Run inference
            output_themes = df['script'].apply(self.get_themes_inference)
            themes_df = pd.DataFrame(output_themes.tolist())
            df = pd.concat([df, themes_df], axis=1)

            # Save results
            if save_path is not None:
                df.to_csv(save_path, index=False)
                print(f"✅ Saved themes to: {save_path}")

            return df

        except Exception as e:
            print("❌ Exception in get_themes():")
            traceback.print_exc()
            raise
