import pytest

import os, sys

from typing import Dict, Any, List, Optional

import logging
import anndata
import pandas as pd

# better way to perform this?
sys.path.append('../')
#import src as h5py_custom_utils


from src.exploration import generate_summary_report, generate_anndata_report
from src.loading import create_obs_subset, check_values_for_key, check_input_dictionary, grab_row_values, create_anndata_subset

from dataset_initialisation import data_dir, test_input_settings


# Set Logging
logging.basicConfig(level=logging.BASIC_FORMAT)


def test_create_obs_subset_xpass(
    data_dir : str = data_dir,
    filter_column : str = 'anno_LVL2',
    filter_values : list = ['macrophage', 'erythroid'],
    additional_cols_keep : list = ['anno_LVL1', 'donor', 'biological_unit'],
    ):

    obs_subset = create_obs_subset(data_dir, filter_column, filter_values, additional_cols_keep)

    assert isinstance(obs_subset, pd.DataFrame)
    assert 'anno_LVL2' in obs_subset.columns
    for col in additional_cols_keep:
        assert col in obs_subset.columns
        
    for value in filter_values:
        assert value in list(obs_subset[filter_column].unique())




def test_create_anndata_subset_check(
    
    filter_column_var : list,
    filter_layers : list,
    filter_obsm : list,
    filter_obsp : list,
    filter_varm : list,
    filter_varp : list,
    filter_uns : list,
    
    
    
    
    data_dir : str = data_dir,
    filter_column_obs : str = 'anno_LVL2',
    filter_values_obs : list = ['macrophage', 'erythroid'],
    additional_cols_keep_obs : list = ['anno_LVL1', 'donor', 'biological_unit'],
    

    keep_layers : bool = False,
    keep_obsm : bool = False,
    keep_obsp : bool = False,
    keep_varm : bool = False,
    keep_varp : bool = False,
    keep_uns : bool = False,
    
    test_input_settings : dict = test_input_settings,
    
    ):
        
    adata_slice = create_anndata_subset_check(**test_input_settings)   
        
    assert isinstance(adata_slice, anndata.AnnData)
    assert adata_result.shape == (2051, 36000)
    
    adata_check = anndata.read_h5ad(data_dir)
    adata_check = adata_check[adata_check.obs[filter_column_obs].isin(filter_values_obs)]
    assert np.array_equal(adata_check.X.toarray(), adata_slice.X.toarray())
    