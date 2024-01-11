from tests import _PATH_DATA
import torch
import pandas as pd
from mlops_enzyme_stability.data.make_dataset import load_data, save_tensor, preprocessing

def test_load_data():
    train_df, test_df, test_labels = load_data()
    for df in [train_df, test_df, test_labels]:
        assert isinstance(df, pd.DataFrame)
        assert not df.empty

def test_preprocessing():
    # Mock data simulating the scenarios in your preprocessing function
    mock_train_data = {
        'seq_id': [1, 2, 3, 4],
        'pH': [7.0, 8.0, 7.5, 6.5],
        'tm': [50, 55, 60, 45],
    }
    mock_train_updates_data = {
        'seq_id': [1, 2, 3],
        'pH': [None, 7.8, None],
        'tm': [None, 62, None],
    }
    mock_solution = {
        'seq_id': [2, 4],
        'pH': [7.8, 6.5],
        'tm': [62, 45],
    }

    df_train = pd.DataFrame(mock_train_data).set_index('seq_id')
    df_train_updates = pd.DataFrame(mock_train_updates_data).set_index('seq_id')
    df_solution = pd.DataFrame(mock_solution).set_index('seq_id')

    # Call preprocessing
    processed_df = preprocessing(df_train, df_train_updates)

    # Check that rows with all features missing are removed
    assert processed_df.equals(df_solution)


def test_save_tensor(tmp_path):
    tensor_dummy = [torch.tensor([1, 2, 3]), torch.tensor([4, 5, 6])]
    file_path = tmp_path / "test_tensor.pt"
    save_tensor(tensor_dummy, file_path)
    assert file_path.exists()