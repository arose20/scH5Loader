import pandas as pd
import h5py
from pprint import pprint

from IPython.display import display

from typing import List, Dict, Any


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
                        
                    summary_report[key]['sublayer_info'] = layer_dict # type: ignore
            
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
    columns : List[str],
    ):
    
    # Initialize an empty dictionary to store decoded categories
    decoded_data: Dict[str, List[Any]] = {col: [] for col in columns}
    
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
        
    globals()[output_dataframe_name] = df