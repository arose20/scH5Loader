import pytest

import os, sys

from typing import Dict, Any, List, Optional

import logging
import anndata
import pandas as pd


#sys.path.insert(0, "/home.jovyan/mount_farm/repos/H5py_anndata_checker/src")
# better way to perform this?
sys.path.append('../')
#import src as h5py_custom_utils

import src




#from src.exploration import generate_summary_report, generate_anndata_report
#from src.loading import create_obs_subset, check_values_for_key, check_input_dictionary, grab_row_values, create_anndata_subset


# Test and create setup for testing

@pytest.fixture
def establish_test_data_dir_xpass():
    
    data_dir = "./dummy_data/data1.h5ad"
    
    if os.path.exists(data_dir):
        # If the file exists, delete the file
        os.remove(data_dir)
        logging.info(f"The file {data_dir} already exists, removing for testing")
    else:
        logging.info(f"The file {data_dir} does not exist. Proceeding with test")
    
    return str(data_dir)
    

def create_anndata_h5ad_test_file_xpass(self):

    adata : anndata.AnnData = generate_anndata_object_1()
    
    assert isinstance(adata, anndata.AnnData)
    assert adata_result.shape == (5000, 36000)
    
    adata.write(data_dir)
    logging.info(f"Dummy data generated successfully")

        
    
def create_input_dictionary_1(
    data_dir : str = data_dir,
    ):
    
    test_input_settings = {
    'data_dir' : data_dir,
    'filter_column_obs' : 'anno_LVL2',
    'filter_values_obs' : ['macrophage', 'erythroid'],
    'additional_cols_keep_obs' : ['anno_LVL1', 'donor', 'biological_unit'],
    'filter_column_var' : [],
    
    'filter_layers' : [],
    'filter_obsm' : [],
    'filter_obsp' : [],
    'filter_varm' : [],
    'filter_varp' : [],
    'filter_uns' : [],
    
    'keep_layers' : False,
    'keep_obsm' : False,
    'keep_obsp' : False,
    'keep_varm' : False,
    'keep_varp' : False,
    'keep_uns' : False,
    }

    return test_input_settings





