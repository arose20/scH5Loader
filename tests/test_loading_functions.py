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
#import src as h5py_custom_utils

from scH5Loader import *

# Set Logging
logging.basicConfig(level=logging.BASIC_FORMAT)



def test_create_dataframe_subset_obs_intersection_xpass(
    data_dir,
    dataframe = 'obs',
    filter_dict = {

        'anno_LVL1' : ['haematopoetic'],
        'anno_LVL2' : ['macrophage']
    },
    additional_cols_keep = [
        'biological_unit',
        ],
    filter_method = 'intersection'
):
    
    obs_subset = create_dataframe_subset(data_dir, dataframe, filter_dict, additional_cols_keep, filter_method)

    assert isinstance(obs_subset, pd.DataFrame)
    
    assert 'Original_index_value' in obs_subset.columns
    assert 'Original_index_position' in obs_subset.columns
    
    for key, values in filter_dict.items():
        assert key in obs_subset.columns
        for value in values:
            assert value in list(obs_subset[key].unique())
    
    for col in additional_cols_keep:
        assert col in obs_subset.columns
        
    assert len(obs_subset) == 1531




def test_create_dataframe_subset_obs_union_xpass(
    data_dir,
    dataframe = 'obs',
    filter_dict = {

        'anno_LVL1' : ['haematopoetic'],
        'anno_LVL2' : ['hepatocyte']
    },
    additional_cols_keep = [
        'biological_unit',
        ],
    filter_method = 'union'
):
    
    obs_subset = create_dataframe_subset(data_dir, dataframe, filter_dict, additional_cols_keep, filter_method)

    assert isinstance(obs_subset, pd.DataFrame)
    
    assert 'Original_index_value' in obs_subset.columns
    assert 'Original_index_position' in obs_subset.columns
    
    for key, values in filter_dict.items():
        assert key in obs_subset.columns
        for value in values:
            assert value in list(obs_subset[key].unique())
    
    for col in additional_cols_keep:
        assert col in obs_subset.columns
        
    assert len(obs_subset) == 3053



def test_create_dataframe_subset_var_intersection_xpass(
    data_dir,
    dataframe = 'var',
    filter_dict = {

        'high_var' : ['yes'],
    },
    additional_cols_keep = [],
    filter_method = 'intersection'
):
    
    var_subset = create_dataframe_subset(data_dir, dataframe, filter_dict, additional_cols_keep, filter_method)

    assert isinstance(var_subset, pd.DataFrame)
    
    assert 'Original_index_value' in var_subset.columns
    assert 'Original_index_position' in var_subset.columns
    
    for key, values in filter_dict.items():
        assert key in var_subset.columns
        for value in values:
            assert value in list(var_subset[key].unique())
        
    assert len(var_subset) == 21605  
    

    
def test_create_anndata_subset_check_xpass(
    data_dir,
    test_input_settings,
    ):
    
    adata_out = create_anndata_subset(**test_input_settings)
    
    assert isinstance(adata_out, anndata.AnnData)
    assert adata_out.shape == (3053, 21605)

    for key, values in test_input_settings['obs_filter_dict'].items():
        assert key in adata_out.obs.columns
        for value in values:
            assert value in list(adata_out.obs[key].unique())
    
    for key, values in test_input_settings['var_filter_dict'].items():
        assert key in adata_out.var.columns
        for value in values:
            assert value in list(adata_out.var[key].unique())

    
    adata_check = anndata.read_h5ad(data_dir)
    adata_check = adata_check[adata_check.obs.index.isin(list(adata_out.obs.index))]
    adata_check = adata_check[:,adata_check.var.index.isin(list(adata_out.var.index))]
    assert np.array_equal(adata_check.X.toarray(), adata_out.X.toarray())
    