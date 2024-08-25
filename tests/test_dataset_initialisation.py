import pytest
import cProfile
import pstats
import io
import os
import sys
import logging
import anndata
import pandas as pd
from typing import Dict, Any, List, Optional
from scH5Loader import generate_anndata_object_1

sys.path.append('../')

# Test and create setup for testing

@pytest.fixture
def establish_test_data_dir_xpass(data_dir: str):
    """Set up the testing environment by ensuring the test data directory is ready."""

    if os.path.exists(data_dir):
        # If the file exists, delete the file
        os.remove(data_dir)
        logging.info(f"The file {data_dir} already exists, removing for testing")
    else:
        logging.info(f"The file {data_dir} does not exist. Proceeding with test")
    
    return str(data_dir)
    

def create_anndata_h5ad_test_file_xpass(self, data_dir: str):
    """Create a test AnnData object and save it to an H5AD file."""
    
    adata: anndata.AnnData = generate_anndata_object_1()
    
    assert isinstance(adata, anndata.AnnData)
    assert adata.shape == (5000, 36000)
    
    adata.write(data_dir)
    logging.info(f"Dummy data generated successfully at {data_dir}")
