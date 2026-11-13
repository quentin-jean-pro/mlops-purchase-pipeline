import pandas as pd
import numpy as np
import pytest
from purchase_predict.pipelines.processing.nodes import encode_features, split_dataset

@pytest.fixture
def sample_dataset():
    data = {
        "user_id": [1, 2, 3, 4, 5, 6],
        "user_session": ["s1", "s2", "s3", "s4", "s5", "s6"],
        "category": ["electronics", "apparel", None, "electronics", "apparel", "home"],
        "sub_category": ["phone", "shirt", "lamp", "tv", None, "desk"],
        "brand": ["samsung", "nike", "philips", None, "adidas", "ikea"],
        "weekday": [1, 2, 3, 4, 5, 6],
        "purchased": [1, 0, 1, 0, 1, 0]
    }
    return pd.DataFrame(data)

def test_encode_features(sample_dataset):
    result = encode_features(sample_dataset)
    features = result["features"]
    pipeline = result["transform_pipeline"]

    assert "user_id" not in features.columns
    assert "user_session" not in features.columns
    assert "category" in pipeline
    assert "brand" in pipeline
    assert not features["category"].isnull().any()
    assert np.issubdtype(features["category"].dtype, np.integer)

def test_split_dataset(sample_dataset):
    result = split_dataset(sample_dataset, test_ratio=0.5)
    X_train = result["X_train"]
    X_test = result["X_test"]
    y_train = result["y_train"]
    y_test = result["y_test"]

    assert len(X_train) == 3
    assert len(X_test) == 3
    assert len(y_train) == 3
    assert len(y_test) == 3
    assert set(X_train.index).isdisjoint(set(X_test.index))

