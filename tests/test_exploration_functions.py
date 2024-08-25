import pytest
import cProfile
import pstats
import io

import os, sys

from typing import Dict, Any, List, Optional

import logging
import anndata
import pandas as pd

# better way to perform this?
sys.path.append('../')

from scH5Loader import *

# Set Logging
logging.basicConfig(level=logging.BASIC_FORMAT)


@pytest.fixture
def test_generate_summary_report_xpass(
    data_dir : str = data_dir
    ) -> Dict[str, Any]:
    
    summary_report = generate_summary_report(data_dir)
    
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
    data_dir,
    capsys: pytest.CaptureFixture
    ):
    
    generate_anndata_report(data_dir)
    
    #Capture printed output
    captured = capsys.readouterr()
    
    # Assert statements to check output
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


def test_inspect_column_categories_xpass(
    data_dir,
    capsys: pytest.CaptureFixture,
    
    dataframe : Literal["obs", "var"] = 'obs',
    columns : List[str] = [
        'anno_LVL1', 
        'anno_LVL2',
        'biological_unit'
    ]
    )-> pd.DataFrame:

    df : pd.DataFrame = inspect_column_categories(data_dir, dataframe, columns)
    
    assert isinstance(df, pd.DataFrame)
    for col in columns:
        assert f'{col} unique values' in df.columns
