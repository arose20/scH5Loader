import pytest
import sys
from typing import Dict, Any

sys.path.append('../')

@pytest.fixture
def data_dir():
    # Setup code for data_dir
    data_dir = "../dummy_data/test_adata.h5ad"
    return data_dir

@pytest.fixture
def test_input_settings(data_dir: str) -> Dict[str, Any]:
    #Create a dictionary of input settings for the test using the data directory

    test_input_settings = {
        'data_dir': data_dir,
        'obs_filter_dict': {
            'anno_LVL1': ['haematopoetic'],
            'anno_LVL2': ['hepatocyte']
        },
        'obs_additional_cols_keep': [],
        'obs_filter_method': 'union',
        'var_filter_dict': {
            'high_var': ['yes'],
        },
        'var_additional_cols_keep': [],
        'var_filter_method': 'intersection',
        'filter_layers': [],
        'filter_obsm': [],
        'filter_obsp': [],
        'filter_varm': [],
        'filter_varp': [],
        'filter_uns': [],
        'keep_layers': False,
        'keep_obsm': False,
        'keep_obsp': False,
        'keep_varm': False,
        'keep_varp': False,
        'keep_uns': False,
    }

    return test_input_settings


def pytest_collection_modifyitems(items):
    # Define the order you want to enforce
    # For example, prioritize tests from specific files

    # Separate items into different categories based on file names
    first_file_tests = [item for item in items if "test_dataset_initialisation" in item.fspath.basename]
    second_file_tests = [item for item in items if "test_exploration_functions" in item.fspath.basename]
    third_file_tests = [item for item in items if "test_loading_functions" in item.fspath.basename]
    
    # Concatenate the lists to enforce the order
    items[:] = first_file_tests + second_file_tests + third_file_tests