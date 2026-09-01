"""Data loading and preprocessing utilities for the CBCL prediction project.

This module handles:
- loading the raw dataset
- constructing war-exposure scores
- constructing CBCL scores
- computing basic text-length features
- filtering incomplete observations
- train/test splitting
- creation of Low / Medium / High CBCL labels

Important:
CBCL severity thresholds are estimated from the training set only in order
to avoid information leakage from the test set.
"""

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

from config import (
    RANDOM_STATE,
    TEST_SIZE,
    TARGET_COLUMN,
    LOW_Q,
    HIGH_Q
)


DATA_URL = "https://github.com/OfriA/AppliedProject52056/raw/refs/heads/main/data/ER_data.xlsx"


SELF_EXPOSURE_FEATURES = [
    'SelfExposure_1',
    'SelfExposure_2',
    'SelfExposure_3',
    'SelfExposure_4',
    'SelfExposure_5',
    'SelfExposure_6'
]

OTHER_EXPOSURE_FEATURES = [
    'OtherExposure_6',
    'OtherExposure_7',
    'OtherExposure_8',
    'OtherExposure_9',
    'OtherExposure_10',
    'OtherExposure_11',
    'OtherExposure_12'
]


def load_data(url=DATA_URL):
    return pd.read_excel(url)


def add_war_exposure_scores(df):

    df = df.copy()

    df["self_exposure_score"] = df[SELF_EXPOSURE_FEATURES].sum(axis=1)

    df["other_exposure_score"] = df[OTHER_EXPOSURE_FEATURES].sum(axis=1)

    df["war_exposure_score"] = (
        df["self_exposure_score"]
        + df["other_exposure_score"]
    )

    return df


def add_cbcl_scores(df):

    df = df.copy()

    cbcl_features = df.columns[240:283].tolist()

    cbcl_d = cbcl_features[:13]
    cbcl_a = cbcl_features[13:31]
    cbcl_s = cbcl_features[31:]

    df["CBCL_D_score"] = df[cbcl_d].sum(axis=1)
    df["CBCL_A_score"] = df[cbcl_a].sum(axis=1)
    df["CBCL_S_score"] = df[cbcl_s].sum(axis=1)

    df["CBCL_score"] = (
        df["CBCL_D_score"]
        + df["CBCL_A_score"]
        + df["CBCL_S_score"]
    )

    return df, cbcl_features


def add_text_lengths(df):

    df = df.copy()

    df["Event_length"] = (
        df["Event"]
        .fillna("")
        .astype(str)
        .str.len()
    )

    df["EER_text_length"] = (
        df["EER_text"]
        .fillna("")
        .astype(str)
        .str.len()
    )

    return df


def remove_missing_rows(df, cbcl_features):

    exposure_features = (
        SELF_EXPOSURE_FEATURES
        + OTHER_EXPOSURE_FEATURES
    )

    keep_mask = (
        df[
            exposure_features
            + cbcl_features
        ]
        .isna()
        .sum(axis=1)
        == 0
    )

    df = df.loc[keep_mask].copy()

    df = df[df["EER_text"].notna()]

    return df


def prepare_dataset():

    df = load_data()

    df = add_war_exposure_scores(df)

    df, cbcl_features = add_cbcl_scores(df)

    df = add_text_lengths(df)

    df = remove_missing_rows(df, cbcl_features)

    df["child_age"] = df["child_age"].astype("int16")

    return df


def split_data(df):

    train_df, test_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )

    return train_df.copy(), test_df.copy()


def create_cbcl_labels(train_df, test_df):

    train_df = train_df.copy()
    test_df = test_df.copy()

    t1 = train_df[TARGET_COLUMN].quantile(LOW_Q)
    t2 = train_df[TARGET_COLUMN].quantile(HIGH_Q)

    def assign_group(x):

        if x <= t1:
            return "Low"

        if x <= t2:
            return "Medium"

        return "High"

    train_df["CBCL_label"] = train_df[TARGET_COLUMN].apply(assign_group)

    test_df["CBCL_label"] = test_df[TARGET_COLUMN].apply(assign_group)

    return train_df, test_df, t1, t2
