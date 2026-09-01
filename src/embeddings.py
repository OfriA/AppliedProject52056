"""Text embedding utilities for the CBCL prediction project.

This module generates multilingual E5 embeddings for the project's
free-text fields and optionally caches the resulting feature matrices.

Each text field is embedded separately and the resulting vectors are
concatenated to create the final representation used by the baseline
classifier.
"""

import os
import numpy as np
from sentence_transformers import SentenceTransformer
from config import (
    EMBEDDING_MODEL,
    TEXT_COLUMNS
)


def load_embedding_model():

    return SentenceTransformer(
        EMBEDDING_MODEL
    )


def embed_column(model, texts):

    texts = (
        texts
        .fillna("")
        .astype(str)
        .tolist()
    )

    texts = [
        "passage: " + t
        for t in texts
    ]

    return model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=True
    )


def build_embeddings(
    train_df,
    test_df,
    text_columns=TEXT_COLUMNS
):

    model = load_embedding_model()

    train_parts = []
    test_parts = []

    for col in text_columns:

        train_emb = embed_column(
            model,
            train_df[col]
        )

        test_emb = embed_column(
            model,
            test_df[col]
        )

        train_parts.append(train_emb)
        test_parts.append(test_emb)

    X_train = np.hstack(train_parts)
    X_test = np.hstack(test_parts)

    return X_train, X_test

def load_or_create_embeddings(
    train_df,
    test_df,
    cache_dir="cache"
):

    os.makedirs(cache_dir, exist_ok=True)

    train_path = os.path.join(
        cache_dir,
        "X_train_embeddings.npy"
    )

    test_path = os.path.join(
        cache_dir,
        "X_test_embeddings.npy"
    )

    if (
        os.path.exists(train_path)
        and
        os.path.exists(test_path)
    ):

        print("Loading cached embeddings...")

        X_train = np.load(train_path)
        X_test = np.load(test_path)

        return X_train, X_test

    print("Creating embeddings...")

    X_train, X_test = build_embeddings(
        train_df,
        test_df
    )

    np.save(train_path, X_train)
    np.save(test_path, X_test)

    return X_train, X_test
