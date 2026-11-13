from unittest.mock import MagicMock, patch
import pandas as pd
from purchase_predict.pipelines.loading.nodes import load_csv_from_bucket


def test_load_csv_from_bucket():
    mock_df = pd.DataFrame({f"col_{i}": [1, 2] for i in range(15)})
    mock_df["purchased"] = [0, 1]

    with patch("purchase_predict.pipelines.loading.nodes.storage.Client") as mock_client:
        mock_instance = MagicMock()
        mock_client.return_value = mock_instance
        mock_instance.list_blobs.return_value = []

        with patch("purchase_predict.pipelines.loading.nodes.glob.glob", return_value=["/tmp/fake.csv"]):
            with patch("pandas.read_csv", return_value=mock_df):
                df = load_csv_from_bucket("dummy-project", "dummy-bucket/data.csv")
                assert isinstance(df, pd.DataFrame)
                assert df.shape[1] == 16
                assert "purchased" in df
