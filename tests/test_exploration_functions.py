import pytest

import os, sys

from typing import Dict, Any, List, Optional

import logging
import anndata
import pandas as pd

# better way to perform this?
sys.path.append('../')
#import src as h5py_custom_utils


from src.exploration import * #generate_summary_report, generate_anndata_report
from src.loading import * #create_obs_subset, check_values_for_key, check_input_dictionary, grab_row_values, create_anndata_subset

from dataset_initialisation import data_dir


# Set Logging
logging.basicConfig(level=logging.BASIC_FORMAT)

# Test functions in exploration.py

@pytest.fixture
def test_generate_summary_report_xpass(
    data_dir : str = data_dir
    ) -> Dict[str, Any]:
    
    summary_report = generate_anndata_report(data_dir)
    
    assert isinstance(summary_report, dict)
    
    assert 'Partition: X' in summary_report
    assert 'Partition: layers' in summary_report
    assert 'Partition: obs' in summary_report
    assert 'Partition: obsm' in summary_report
    assert 'Partition: obsp' in summary_report
    assert 'Partition: var' in summary_report
    assert 'Partition: varm' in summary_report
    assert 'Partition: varp' in summary_report
    assert 'Partition: uns' in summary_report
    
    return summary_report

    
def test_generate_anndata_report_xpass(
    summary_report: Dict[str, Any],
    capsys: pytest.CaptureFixture
    ):
    
    # Capture printed output
    captured = capsys.readouterr()
    
    assert 'Anndata object checking' in captured.out
    assert 'Partition: X' in captured.out
    assert 'Partition: layers' in captured.out
    assert 'Partition: obs' in captured.out
    assert 'Partition: obsm' in captured.out
    assert 'Partition: obsp' in captured.out
    assert 'Partition: var' in captured.out
    assert 'Partition: varm' in captured.out
    assert 'Partition: varp' in captured.out
    assert 'Partition: uns' in captured.out

def test_investigate_obs_columns_unique_contents_xpass(
    capsys: pytest.CaptureFixture
    
    data_dir : str = data_dir,
    output_dataframe_name : str = 'df',
    dataframe : str = 'obs',
    columns : list = [
        'anno_LVL1',
        'anno_LVL2',
        'donor'
    ],
    ):

    df : pd.DataFrame = investigate_obs_columns_unique_contents(data_dir, output_dataframe_name, dataframe, columns)
    
    assert isinstance(df, pd.DataFrame)
    for col in columns:
        assert f'{col} unique values' in df.columns
    
    # Capture printed output
    captured = capsys.readouterr()
    
    assert 'DataFrame output to see all unique values for each column of interest:' in captured.out

    
    
