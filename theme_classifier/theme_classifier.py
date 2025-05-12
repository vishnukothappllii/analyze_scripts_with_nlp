from transformers import pipeline
import torch
from nltk.tokenize import sent_tokenize
import nltk
import pandas as pd
import numpy as np
import os
import sys
import pathlib

folder_path = pathlib.Path(__file__).parent.resolve()
sys.path.append(os.path.join(folder_path, '../'))
from utils import load_subtitles_dataset

nltk.download('punkt')

class ThemeClassifier():
    def __init__(self, theme_list):
        self.model_name = "facebook/bart-large-mnli"
        self.device = 0 if torch.cuda.is_available() else -1  # -1 = CPU for pipeline
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

        # Batch Sentence
        sentence_batch_size = 50  # Increase batch size for speed
        script_batches = [
            " ".join(script_sentences[i:i + sentence_batch_size])
            for i in range(0, len(script_sentences), sentence_batch_size)
        ]

        # TEMPORARY: Limit batches for testing
        script_batches = script_batches[:2]

        # Run Model
        theme_output = self.theme_classifier(
            script_batches,
            self.theme_list,
            multi_label=True
        )

        # Wrangle Output
        themes = {}
        for output in theme_output:
            for label, score in zip(output['labels'], output['scores']):
                themes.setdefault(label, []).append(score)

        themes = {key: np.mean(value) for key, value in themes.items()}
        return themes

    def get_themes(self, dtaset_path, save_path=None):
        if save_path is not None and os.path.exists(save_path):
            return pd.read_csv(save_path)

        df = load_subtitles_dataset(dtaset_path)

        # TEMPORARY: For testing, only a few rows
        df = df.head(2)

        output_themes = df['script'].apply(self.get_themes_inference)
        themes_df = pd.DataFrame(output_themes.tolist())
        df = pd.concat([df, themes_df], axis=1)

        if save_path is not None:
            df.to_csv(save_path, index=False)

        return df
