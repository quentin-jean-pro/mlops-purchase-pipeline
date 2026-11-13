from unittest.mock import MagicMock, patch
import pandas as pd
from kedro.runner import SequentialRunner
from purchase_predict.pipelines.loading.pipeline import create_pipeline


def test_pipeline(catalog_test):
    mock_df = pd.DataFrame({f"col_{i}": [1, 2] for i in range(15)})
    mock_df["purchased"] = [0, 1]

    with patch("purchase_predict.pipelines.loading.nodes.storage.Client") as mock_client:
        mock_instance = MagicMock()
        mock_client.return_value = mock_instance
        mock_instance.list_blobs.return_value = []

        with patch("purchase_predict.pipelines.loading.nodes.glob.glob", return_value=["/tmp/fake.csv"]):
            with patch("pandas.read_csv", return_value=mock_df):
                runner = SequentialRunner()
                pipeline = create_pipeline()
                pipeline_output = runner.run(pipeline, catalog_test)
                df = pipeline_output["primary"].load()
                assert isinstance(df, pd.DataFrame)
                assert df.shape[1] == 16
                assert "purchased" in df
