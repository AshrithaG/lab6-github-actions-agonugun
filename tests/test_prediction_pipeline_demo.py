import pytest
import pandas as pd
import numpy as np
import math
import os
import runpy
import pathlib

from prediction_pipeline_demo import (
    data_preparation,
    data_split,
    train_model,
    eval_model,
)

@pytest.fixture
def housing_data_sample():
    rows = []
    base = {
        "price": 13_300_000,
        "area": 7420,
        "bedrooms": 4,
        "bathrooms": 2,
        "stories": 3,
        "mainroad": "yes",
        "guestroom": "no",
        "basement": "no",
        "hotwaterheating": "no",
        "airconditioning": "no",
        "parking": 2,
        "prefarea": "no",
        "furnishingstatus": "semi-furnished",
    }
    for i in range(30):
        r = base.copy()
        r["price"] += i * 1000
        r["area"] += i * 10
        r["furnishingstatus"] = ["unfurnished", "semi-furnished", "furnished"][i % 3]
        rows.append(r)
    return pd.DataFrame(rows)

@pytest.fixture
def feature_target_sample(housing_data_sample):
    X, y = data_preparation(housing_data_sample)
    return X, y

def test_feature_columns_and_types(feature_target_sample):
    X, y = feature_target_sample
    expected = {
        "furnishingstatus_furnished",
        "furnishingstatus_semi-furnished",
        "furnishingstatus_unfurnished",
    }
    assert expected.issubset(set(X.columns))
    assert len(X) == len(y)
    assert all(np.issubdtype(dt, np.number) for dt in X.dtypes)

def test_data_split_shapes(feature_target_sample):
    X_train, X_test, y_train, y_test = data_split(*feature_target_sample)
    assert isinstance((X_train, X_test, y_train, y_test), tuple)
    assert len((X_train, X_test, y_train, y_test)) == 4
    assert X_train.shape[0] + X_test.shape[0] == feature_target_sample[0].shape[0]
    assert y_train.shape[0] + y_test.shape[0] == feature_target_sample[1].shape[0]

def test_end_to_end_train_and_eval(feature_target_sample):
    X_train, X_test, y_train, y_test = data_split(*feature_target_sample)
    model = train_model(X_train, y_train)
    score = eval_model(X_test, y_test, model)
    assert isinstance(score, float)
    assert math.isfinite(score)

def test_main_block_runs_and_prints_score(capsys):
    """Execute the script so coverage captures the __main__ block."""
    project_root = pathlib.Path(__file__).resolve().parents[1]
    cwd = os.getcwd()
    try:
        os.chdir(project_root)
        runpy.run_path(str(project_root / "prediction_pipeline_demo.py"), run_name="__main__")
    finally:
        os.chdir(cwd)
    out = capsys.readouterr().out
    assert "Trained model score is:" in out
