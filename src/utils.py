import numpy as np
import pandas as pd
import scanpy as sc
import h5py
import anndata
from anndata._io.h5ad import read_elem
from scipy.sparse import csr_matrix
from pprint import pprint
import warnings
import time
from tqdm import tqdm



def generate_summary_report(data_dir):
    with h5py.File(data_dir, 'r') as file:
        keys = file.keys()

        # Exclude 'X' partition
        #keys_without_x = [key for key in keys if key != 'X']

        # Generate summary report
        summary_report = {}
        for key in keys:
            group = file[key]
            if isinstance(group, h5py.Group):
                # For groups, get keys and types inside the group
                group_keys = list(group.keys())
                group_types = [type(group[key]) for key in group_keys]
                summary_report[key] = {'group_keys': group_keys, 'group_types': group_types}
                
                if key == "X":
                    summary_report[key]['Cell_num:'] = file[key]['indptr'].shape[0] - 1
                    
                elif key == 'layers':
                    layer_dict = {}
                    
                    for k in file['layers'].keys():
                        sub_group = file[key][k]
                        sub_group_keys = list(sub_group.keys())
                        sub_group_types = [type(sub_group[key]) for key in sub_group]
                        layer_dict[k] = {'group_keys': sub_group_keys, 'group_types': sub_group_types}
                        layer_dict[k]['Cell_num:'] = file[key][k]['indptr'].shape[0] - 1
                        
                    summary_report[key]['sublayer_info'] = layer_dict
            
            elif isinstance(group, h5py.Dataset):
                # For datasets, get shape and dtype
                dtype = group.dtype if hasattr(group, 'dtype') else None
                summary_report[key] = {'shape': group.shape, 'dtype': dtype}

    return summary_report



def generate_anndata_report(
    
    data_dir : str,
    
    ):
    
    report = generate_summary_report(data_dir)

    print(f"\033[1mAnndata object checking: {data_dir}\033[0m\n")
    
    # Print the summary report
    for key, value in report.items():
        #print("")
        print(f"Partition: {key}")
        if 'group_keys' in value:
            pprint(f"  Group Keys: {value['group_keys']}")
            if key == 'X':
                print(f"   Number of cells: {report['X']['Cell_num:']}")
            elif key == 'layers':
                print("")
                print("Sub layer information:")
                for k, v in report['layers']['sublayer_info'].items():
                    print(f"{k}:")
                    pprint(f"  Group Keys: {v['group_keys']}")
                    #pprint(f"  Group Types: {v['group_types']}")
                    print(f"   Number of cells: {report['layers']['sublayer_info'][k]['Cell_num:']}")
                    print('')
            #pprint(f"  Group Types: {value['group_types']}")
        else:
            pprint(f"  Shape: {value['shape']}")
            pprint(f"  Dtype: {value['dtype']}")
        if report[key]['group_keys']:
            print("")
        print("---------------------------")

        
def investigate_obs_columns_unique_contents(
    
    data_dir : str,
    output_dataframe_name : str,
    dataframe : str,
    columns : list,
    ):
    
    # Initialize an empty dictionary to store decoded categories
    decoded_data = {col: [] for col in columns}
    
    with h5py.File(data_dir, 'r') as file:
        # Iterate over columns and decode categories
        for col in columns:
            decoded_data[col] = [value.decode() for value in file[dataframe][col]['categories']]
    
    # Find the maximum length among all categories
    max_length = max(len(decoded_data[col]) for col in columns)
    
    # Pad the lists in the dictionary with None if needed
    for col in columns:
        decoded_data[col] += [''] * (max_length - len(decoded_data[col]))
    
    # Create a DataFrame from the decoded data
    df = pd.DataFrame(decoded_data)
    
    # Update column names with ' unique values'
    df.columns = [f"{col} unique values" for col in columns]
    
    print("\033[1mDataFrame output to see all unique values for each column of interest:\033[0m\n")
    display(df)
    
    if output_dataframe_name in locals():
        print(f"Variable '{output_dataframe_name}' already exists. Overwriting.")
        
    locals()[output_dataframe_name] = df
    
    
def create_obs_subset(
    
    data_dir : str,
    filter_column : str,
    filter_values_obs : list,
    additional_cols_keep : list,
    ):
    
    with h5py.File(data_dir, 'r') as file:
        # Get the index from the 'obs' dataset attributes
        original_index = pd.DataFrame(index=np.vectorize(lambda x: x.decode('utf-8'))(np.array(file["obs"]["_index"], dtype=object)))
        original_index['Original_index_position'] = np.arange(len(original_index))
        
        # Establish the rows to keep
        cols = {}
        col_data = pd.DataFrame(read_elem(file['obs'][filter_column]))
        
        # Filter the DataFrame and check for missing values in one line
        col_data = col_data[col_data.iloc[:, 0].isin(filter_values_obs)]
        
        # Check if all values in filter_values_obs are present
        if not set(filter_values_obs).issubset(set(col_data.iloc[:, 0])):
            raise ValueError("Not all values in filter_values_obs are present in filter_column.")
            
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
    
        
def grab_row_values(
    
    rows_to_load: list,
    data_dset,
    indices_dset,
    indptr_dset,
    ):
    
    # Initalise empty lists - lists are dynamic in size so adding to a list then converting to a numpy array can be faster then initialising the entire size of the required numpy array
    selected_rows_data = []
    selected_rows_indices = []
    selected_rows_indptr = [0]
    
    # Cycle through the rows of interest
    for row_idx in rows_to_load:
        start_idx = indptr_dset[row_idx]
        end_idx = indptr_dset[row_idx + 1]
        selected_rows_data.extend(data_dset[start_idx:end_idx])
        selected_rows_indices.extend(indices_dset[start_idx:end_idx])
        selected_rows_indptr.append(selected_rows_indptr[-1] + (end_idx - start_idx))
        
    # Convert lists to NumPy arrays
    selected_rows_data = np.array(selected_rows_data)
    selected_rows_indices = np.array(selected_rows_indices)
    selected_rows_indptr = np.array(selected_rows_indptr)
    
    return selected_rows_data, selected_rows_indices, selected_rows_indptr
        
    
def create_anndata_subset(**kwargs): 
    
    for key, value in kwargs.items():
        globals()[key] = value
    kwargs.update(locals())
    
    with h5py.File(data_dir, 'r') as file:
        # Get the index from the 'obs' dataset attributes
        original_index = pd.DataFrame(index=np.vectorize(lambda x: x.decode('utf-8'))(np.array(file["obs"]["_index"], dtype=object)))
        original_index['Original_index_position'] = np.arange(len(original_index))
        
        # Establish the rows to keep
        cols = {}
        col_data = pd.DataFrame(read_elem(file['obs'][filter_column_obs]))
        
        # Check that all values in filter_values are in filter_column
        #if not set(filter_values_obs).issubset(set(col_data.iloc[:, 0])):
        
        # Filter the DataFrame and check for missing values in one line
        col_data = col_data[col_data.iloc[:, 0].isin(filter_values_obs)]
        
        # Check if all values in filter_values_obs are present
        if not set(filter_values_obs).issubset(set(col_data.iloc[:, 0])):
            raise ValueError("Not all values in filter_values_obs are present in filter_column_obs.")

        
        
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
        
        
        ## Initalise empty lists - lists are dynamic in size so adding to a list then converting to a numpy array can be faster then initialising the entire size of the required numpy array
        #selected_rows_data = []
        #selected_rows_indices = []
        #selected_rows_indptr = [0]
        #
        ## Cycle through the rows of interest
        #for row_idx in rows_to_load:
        #    start_idx = indptr_dset[row_idx]
        #    end_idx = indptr_dset[row_idx + 1]
        #    selected_rows_data.extend(data_dset[start_idx:end_idx])
        #    selected_rows_indices.extend(indices_dset[start_idx:end_idx])
        #    selected_rows_indptr.append(selected_rows_indptr[-1] + (end_idx - start_idx))
        #
        ## Convert lists to NumPy arrays
        #selected_rows_data = np.array(selected_rows_data)
        #selected_rows_indices = np.array(selected_rows_indices)
        #selected_rows_indptr = np.array(selected_rows_indptr)
        
        
        selected_rows_data, selected_rows_indices, selected_rows_indptr = grab_row_values(rows_to_load,data_dset,indices_dset,indptr_dset)
        
        
        # Create csr_matrix directly from NumPy arrays
        subset_matrix = csr_matrix(
            (selected_rows_data, selected_rows_indices, selected_rows_indptr),
            shape=(len(rows_to_load), num_columns),
            dtype=file["X"]["data"].dtype
        )
        
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
                    
                    # Initalise empty lists - lists are dynamic in size so adding to a list then converting to a numpy array can be faster then initialising the entire size of the required numpy array
                    #selected_rows_data = []
                    #selected_rows_indices = []
                    #selected_rows_indptr = [0]
                    
                    # Cycle through the rows of interest
                    #for row_idx in rows_to_load:
                    #    start_idx = indptr_dset[row_idx]
                    #    end_idx = indptr_dset[row_idx + 1]
                    #    selected_rows_data.extend(data_dset[start_idx:end_idx])
                    #    selected_rows_indices.extend(indices_dset[start_idx:end_idx])
                    #    selected_rows_indptr.append(selected_rows_indptr[-1] + (end_idx - start_idx))
                    
                    # Convert lists to NumPy arrays
                    #selected_rows_data = np.array(selected_rows_data)
                    #selected_rows_indices = np.array(selected_rows_indices)
                    #selected_rows_indptr = np.array(selected_rows_indptr)
                   
                
                
                    selected_rows_data, selected_rows_indices, selected_rows_indptr = grab_row_values(rows_to_load,data_dset,indices_dset,indptr_dset)
                
                    # Create csr_matrix directly from NumPy arrays
                    layers[x] = csr_matrix(
                        (selected_rows_data, selected_rows_indices, selected_rows_indptr),
                        shape=(len(rows_to_load), num_columns),
                        dtype=file["layers"][x]["data"].dtype
                    )
                    
            else:
                layers = {}
                for x in (value for value in filter_layers if value in file["layers"].keys()):
                    
                    # Assign variables to query
                    data_dset = file['layers'][x]['data']
                    indices_dset = file['layers'][x]['indices']
                    indptr_dset = file['layers'][x]['indptr']
                    
                    # Initalise empty lists - lists are dynamic in size so adding to a list then converting to a numpy array can be faster then initialising the entire size of the required numpy array
                    #selected_rows_data = []
                    #selected_rows_indices = []
                    #selected_rows_indptr = [0]
                    
                    # Cycle through the rows of interest
                    #for row_idx in rows_to_load:
                    #    start_idx = indptr_dset[row_idx]
                    #    end_idx = indptr_dset[row_idx + 1]
                    #    selected_rows_data.extend(data_dset[start_idx:end_idx])
                    #    selected_rows_indices.extend(indices_dset[start_idx:end_idx])
                    #    selected_rows_indptr.append(selected_rows_indptr[-1] + (end_idx - start_idx))
                    
                    # Convert lists to NumPy arrays
                    #selected_rows_data = np.array(selected_rows_data)
                    #selected_rows_indices = np.array(selected_rows_indices)
                    #selected_rows_indptr = np.array(selected_rows_indptr)
                   
                    selected_rows_data, selected_rows_indices, selected_rows_indptr = grab_row_values(rows_to_load,data_dset,indices_dset,indptr_dset)
                
                
                    # Create csr_matrix directly from NumPy arrays
                    layers[x] = csr_matrix(
                        (selected_rows_data, selected_rows_indices, selected_rows_indptr),
                        shape=(len(rows_to_load), num_columns),
                        dtype=file["layers"][x]["data"].dtype
                    )
                    
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
    