import numpy as np
import pandas as pd
import scanpy as sc
import h5py
import anndata
from anndata._io.h5ad import read_elem
from scipy.sparse import csr_matrix
from pprint import pprint
import warnings
from tqdm.notebook import tqdm

from IPython.display import display

from typing import List

def create_obs_subset(
    
    data_dir : str,
    filter_column : str,
    filter_values_obs : List[str],
    additional_cols_keep : List[str],
    ) -> pd.DataFrame:
    
    with h5py.File(data_dir, 'r') as file:
        # Get the index from the 'obs' dataset attributes
        original_index = pd.DataFrame(index=np.vectorize(lambda x: x.decode('utf-8'))(np.array(file["obs"]["_index"], dtype=object)))
        original_index['Original_index_position'] = np.arange(len(original_index))
        
        # Establish the rows to keep
        cols = {}
        col_data = pd.DataFrame(read_elem(file['obs'][filter_column])).astype(str)
        
        # Filter the DataFrame and check for missing values in one line
        col_data = col_data[col_data.iloc[:, 0].isin(filter_values_obs)]
        
        # Check if all values in filter_values_obs are present
        missing_values = set(filter_values_obs) - set(col_data.iloc[:, 0])
        
        if missing_values:
            raise ValueError(f"The following values in filter_values_obs are not present in filter_column: {missing_values}")
        
        # Filter the DataFrame and check for missing values in one line
        #col_data = col_data[col_data.iloc[:, 0].isin(filter_values_obs)]
        
        # Check if all values in filter_values_obs are present
        #if not set(filter_values_obs).issubset(set(col_data.iloc[:, 0])):
        #    raise ValueError("Not all values in filter_values_obs are present in filter_column.")
            
        col_data.rename(columns={0:filter_column}, inplace=True)
        col_data.index = original_index.iloc[col_data.index].index
        col_data.insert(0, 'Original_index_position', original_index.loc[original_index.index.isin(col_data.index), 'Original_index_position'])
        
        #col_data['orig_index_position'] = original_index.loc[original_index.index.isin(col_data.index), 'orig_index_position']
        cols[filter_column] = col_data
        
        # Get the rest of the columns of interest
        for col in additional_cols_keep:
            col_data = pd.DataFrame(read_elem(file['obs'][col]))
            col_data.rename(columns={0:col}, inplace=True)
            col_data.index = original_index.index
            col_data = col_data[col_data.index.isin(cols[filter_column].index)]
            cols[col] = col_data
            
        out_df = pd.concat(cols.values(), keys=None, axis=1)
        original_col_order = ['Original_index_position'] + list(file['obs'].attrs['column-order']) 
        out_df = out_df.filter(items=original_col_order)
        out_df = out_df.loc[:, ~out_df.columns.duplicated()]
    
    print(f"\033[1mDataFrame output for only columns of interest for cells selected by filtering {filter_column}:\033[0m\n")
    print("Values selected to keep:")
    pprint(filter_values_obs)
    print("")
    display(out_df)
    
    return out_df
    

    
    
    
#def check_values_for_key(
#    input_settings, 
#    file, 
#    filter_key, 
#    filter_values):
#    
#    if input_settings[filter_key]:

#        # Check for each value
#        not_present_values = [value for value in input_settings[filter_values] if value not in file[filter_key.split('_')[1]].keys()]

#        if not_present_values:
#            raise ValueError(f"Values not present in {filter_key.split('_')[1]}:", not_present_values)


#def check_input_dictionary(
#    input_settings, 
#    file):
#    
#    # Check all values have been passed in correctly
#    required_keys = [
#        'data_dir',
#        'filter_column_obs',
#        'filter_values_obs',
#        'additional_cols_keep_obs',
#        'filter_column_var',
#        'keep_layers',
#        'filter_layers',
#        'keep_obsm',
#        'filter_obsm',
#        'keep_obsp',
#        'filter_obsp',
#        'keep_varm',
#        'filter_varm',
#        'keep_varp',
#        'filter_varp',
#        'keep_uns',
#        'filter_uns',
#    ]
#    
#    missing_keys = [key for key in required_keys if key not in input_settings]
#
#    if missing_keys:
#        raise ValueError(f"Missing required keys: {', '.join(missing_keys)}")
#
#    # Check for values not present in file['obs'].keys()
#    obs_cols = [input_settings['filter_column_obs']] + input_settings['additional_cols_keep_obs']
#    not_present_values = [value for value in obs_cols if value not in file['obs'].keys()]
#
#    if not_present_values:
#        raise ValueError("Values not present in obs columns:", not_present_values)
#        
#    # Check for values not present in file['var'].keys()   
#    not_present_values = [value for value in input_settings['filter_column_var'] if value not in file['var'].keys()]
#
#    if not_present_values:
#        raise ValueError("Values not present in var columns:", not_present_values)
#           
#        
#    # Check layers
#    check_values_for_key(input_settings, file, 'keep_layers', 'filter_layers')
#       
#    # Check obsm
#    check_values_for_key(input_settings, file, 'keep_obsm', 'filter_obsm')
#    
#    # Check obsp
#    check_values_for_key(input_settings, file, 'keep_obsp', 'filter_obsp')
#    
#    # Check varm
#    check_values_for_key(input_settings, file, 'keep_varm', 'filter_varm')
#    
#    # Check varp
#    check_values_for_key(input_settings, file, 'keep_varp', 'filter_varp')
#    
#    # Check uns
#    check_values_for_key(input_settings, file, 'keep_uns', 'filter_uns')
    
    
    
def grab_row_values(
    
    rows_to_load: List[str],
    data_dset,
    indices_dset,
    indptr_dset,
    description: str,
    ):
    
    # Initalise empty lists - lists are dynamic in size so adding to a list then converting to a numpy array can be faster then initialising the entire size of the required numpy array
    selected_rows_data = []
    selected_rows_indices = []
    selected_rows_indptr = [0]
    
    # Use tqdm for progress bar to show progression of assigning variables
    for row_idx in tqdm(rows_to_load, desc=f"Processing Rows for {description}", unit="row", position=0, leave=True):
        start_idx = indptr_dset[row_idx]
        end_idx = indptr_dset[row_idx + 1]
        selected_rows_data.extend(data_dset[start_idx:end_idx])
        selected_rows_indices.extend(indices_dset[start_idx:end_idx])
        selected_rows_indptr.append(selected_rows_indptr[-1] + (end_idx - start_idx))
        
    # Convert lists to NumPy arrays
    selected_rows_data = np.array(selected_rows_data) # type: ignore
    selected_rows_indices = np.array(selected_rows_indices) # type: ignore
    selected_rows_indptr = np.array(selected_rows_indptr) # type: ignore
    
    return selected_rows_data, selected_rows_indices, selected_rows_indptr


def create_anndata_subset_check(
    data_dir : str,
    filter_column_obs : str,
    filter_values_obs : List[str],#, 'Haematopoeitic_lineage', 'Endothelial'],
    additional_cols_keep_obs : List[str], # leave empty will install all
    filter_column_var : List[str],
    filter_layers : List[str],
    filter_obsm : List[str],
    filter_obsp : List[str],
    filter_varm : List[str],
    filter_varp : List[str],
    filter_uns : List[str],


    keep_layers : bool = False,
    keep_obsm : bool = False,
    keep_obsp : bool = False,
    keep_varm : bool = False,
    keep_varp : bool = False,
    keep_uns : bool = False,



    ):#, **kwargs): 
    
    # unpack all values
    #for key, value in kwargs.items():
    #    globals()[key] = value
    #kwargs.update(locals())
    
    with h5py.File(data_dir, 'r') as file:
        
        # Check input dictionary
        #check_input_dictionary(input_settings, file)
        
        # Get the index from the 'obs' dataset attributes
        original_index = pd.DataFrame(index=np.vectorize(lambda x: x.decode('utf-8'))(np.array(file["obs"]["_index"], dtype=object)))
        original_index['Original_index_position'] = np.arange(len(original_index))
        
        # Establish the rows to keep
        cols = {}
        col_data = pd.DataFrame(read_elem(file['obs'][filter_column_obs]))
        
        # Filter the DataFrame and check for missing values in one line
        col_data = col_data[col_data.iloc[:, 0].isin(filter_values_obs)]
        
        # Check if all values in filter_values_obs are present
        missing_values = set(filter_values_obs) - set(col_data.iloc[:, 0])
        
        if missing_values:
            raise ValueError(f"The following values in filter_values_obs are not present in filter_column: {missing_values}")
        
        # Filter the DataFrame and check for missing values in one line
        #col_data = col_data[col_data.iloc[:, 0].isin(filter_values_obs)]
        
        # Check if all values in filter_values_obs are present
        #if not set(filter_values_obs).issubset(set(col_data.iloc[:, 0])):
        #    raise ValueError("Not all values in filter_values_obs are present in filter_column.")

        
        
        col_data.rename(columns={0:filter_column_obs}, inplace=True)
        col_data.index = original_index.iloc[col_data.index].index
        col_data.insert(0, 'Original_index_position', original_index.loc[original_index.index.isin(col_data.index), 'Original_index_position'])
        
        #col_data['orig_index_position'] = original_index.loc[original_index.index.isin(col_data.index), 'orig_index_position']
        cols[filter_column_obs] = col_data
        
        # Get the rest of the columns of interest
        for col in additional_cols_keep_obs:
            col_data = pd.DataFrame(read_elem(file['obs'][col]))
            col_data.rename(columns={0:col}, inplace=True)
            col_data.index = original_index.index
            col_data = col_data[col_data.index.isin(cols[filter_column_obs].index)]
            cols[col] = col_data
            
        out_df = pd.concat(cols.values(), keys=None, axis=1)
        original_col_order = ['Original_index_position'] + list(file['obs'].attrs['column-order']) 
        out_df = out_df.filter(items=original_col_order)
        out_df = out_df.loc[:, ~out_df.columns.duplicated()]
    
        # List of row positions to load
        rows_to_load = out_df['Original_index_position'].values
        
        
        # Set warning parameters
        warnings.filterwarnings("ignore", category=FutureWarning)
        
        # Assign variables to query
        data_dset = file['X']['data']
        indices_dset = file['X']['indices']
        indptr_dset = file['X']['indptr']
        
        # Determine the number of columns from the maximum value in the indices array
        num_columns = file["var"][file["var"].attrs["_index"]].shape[0]
        
        selected_rows_data, selected_rows_indices, selected_rows_indptr = grab_row_values(rows_to_load,data_dset,indices_dset,indptr_dset, 'main counts data')
        
        # Create csr_matrix directly from NumPy arrays
        print('Constructing data into csr_matrix format:  \U0001F527', flush=True)
        subset_matrix = csr_matrix(
            (selected_rows_data, selected_rows_indices, selected_rows_indptr),
            shape=(len(rows_to_load), num_columns),
            dtype=file["X"]["data"].dtype
        )
        print('Construction complete \u2705')
        
        if filter_column_var:
                var = pd.DataFrame(read_elem(file['var'][filter_column_var]))
        else:
            var = pd.DataFrame(read_elem(file['var']))

        if keep_layers:
            if not filter_layers:
                layers = {}
                for x in file["layers"].keys():
                    
                    # Assign variables to query
                    data_dset = file['layers'][x]['data']
                    indices_dset = file['layers'][x]['indices']
                    indptr_dset = file['layers'][x]['indptr']
                   
                    name = f'layer {x} data'
                
                    selected_rows_data, selected_rows_indices, selected_rows_indptr = grab_row_values(rows_to_load,data_dset,indices_dset,indptr_dset, name)
                
                    # Create csr_matrix directly from NumPy arrays
                    print('Constructing data into csr_matrix format:  \U0001F527', flush=True)
                    layers[x] = csr_matrix(
                        (selected_rows_data, selected_rows_indices, selected_rows_indptr),
                        shape=(len(rows_to_load), num_columns),
                        dtype=file["layers"][x]["data"].dtype
                    )
                    print('Construction complete \u2705')
                    
            else:
                layers = {}
                for x in (value for value in filter_layers if value in file["layers"].keys()):
                    
                    # Assign variables to query
                    data_dset = file['layers'][x]['data']
                    indices_dset = file['layers'][x]['indices']
                    indptr_dset = file['layers'][x]['indptr']
                    
                    name = f'layer {x} data'
                   
                    selected_rows_data, selected_rows_indices, selected_rows_indptr = grab_row_values(rows_to_load,data_dset,indices_dset,indptr_dset, name)
                
                
                    # Create csr_matrix directly from NumPy arrays
                    print('Constructing data into csr_matrix format:  \U0001F527', flush=True)
                    layers[x] = csr_matrix(
                        (selected_rows_data, selected_rows_indices, selected_rows_indptr),
                        shape=(len(rows_to_load), num_columns),
                        dtype=file["layers"][x]["data"].dtype
                    )
                    print('Construction complete \u2705')
                    
        else:
            layers = None
            
        if keep_obsm:
            if not filter_obsm:
                obsm = {x: anndata._io.h5ad.read_elem(file["obsm"][x])[rows_to_load] for x in file["obsm"].keys()}
            else:
                obsm = {x: anndata._io.h5ad.read_elem(file["obsm"][x])[rows_to_load] for x in filter_obsm if x in file["obsm"].keys()}
        else:
            obsm = None

        if keep_obsp:
            if not filter_obsp:
                obsp = {x: anndata._io.h5ad.read_elem(file["obsp"][x])[rows_to_load][:, rows_to_load]  for x in file["obsp"].keys()}
            else:
                obsp = {x: anndata._io.h5ad.read_elem(file["obsp"][x])[rows_to_load][:, rows_to_load] for x in filter_obsp if x in file["obsp"].keys()}
        else:
            obsp = None

        if keep_varm:
            if not filter_varm:
                varm = anndata._io.h5ad.read_elem(file["varm"])
            else:
                varm = {x: anndata._io.h5ad.read_elem(file["varm"][x]) for x in filter_varm if x in file["varm"].keys()}
        else:
            varm = None


        if keep_varp:
            if not filter_varp:
                varp = anndata._io.h5ad.read_elem(file["varp"])
            else:
                varp = {x: anndata._io.h5ad.read_elem(file["varp"][x]) for x in filter_varp if x in file["varp"].keys()}
        else:
            varp = None


        if keep_uns:
            if not filter_uns:
                uns = anndata._io.h5ad.read_elem(file["uns"])
            else:
                uns = {x: anndata._io.h5ad.read_elem(file["uns"][x]) for x in filter_uns if x in file["uns"].keys()}
        else:
            uns = None
            
        adata = anndata.AnnData(
            X = subset_matrix,
            obs=out_df,
            var=var,
            layers=layers,
            obsm=obsm,
            obsp=obsp,
            varm=varm,
            varp=varp,
            uns=uns,
        )
        warnings.filterwarnings("default")
    
    print('')
    print('')
    print(f"\033[1mSubset anndata object generated successfully\033[0m\n")
    print('\033[1m' + 'Anndata whole preview:' + '\033[0m')
    display(adata)
    print('')
    print('')
    print(f"\033[1mQuick view of the anndata object generated\033[0m\n")
    print(f'Overall shape: {adata.shape}')
    print(f'Min count: {adata.X.min()}')
    print(f'Max count: {adata.X.max()}')
    print('')
    print('\033[1m' + 'obs preview:' + '\033[0m')
    display(adata.obs)
    print('')
    print('\033[1m' + 'var preview:' + '\033[0m')
    display(adata.var)
    print('')
    
    return adata

