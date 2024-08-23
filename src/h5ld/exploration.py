import pandas as pd
import h5py
from pprint import pprint
from typing import List, Dict, Any, Literal, Union
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)


def detect_matrix_format(group) -> str:
    """
    Determine the matrix format based on the keys and structure of the given HDF5 group or dataset.

    This function analyzes the structure of an HDF5 group or dataset to identify the format
    of the matrix it represents. The function checks for common sparse matrix formats like
    CSR, CSC, and COO, as well as dense matrix formats like NumPy arrays.

    Parameters:
    ----------
    group : h5py.Group or h5py.Dataset
        The HDF5 group or dataset to be analyzed. This could represent a sparse matrix
        format (e.g., CSR, CSC, COO) or a dense matrix/array.

    Returns:
    -------
    str
        A string indicating the matrix format. Possible return values are:
        - "NumPy Array or Dense Matrix": Indicates that the group is a multi-dimensional
          dataset, likely a dense matrix or NumPy array.
        - "1D Dataset": Indicates that the dataset is one-dimensional.
        - "CSR/CSC Matrix": Indicates that the group contains the necessary keys and
          attributes to represent a CSR (Compressed Sparse Row) or CSC (Compressed Sparse Column) matrix.
        - "COO Matrix": Indicates that the group contains the necessary keys to represent
          a COO (Coordinate) sparse matrix.
        - "Unknown Format": Indicates that the matrix format could not be determined based
          on the group's structure.

    Notes:
    -----
    This function assumes that the input `group` is either a valid HDF5 group or dataset.
    It does not perform extensive validation beyond checking the presence of specific keys
    and attributes.
    """

    # If the group is an HDF5 dataset, check its shape and attributes.
    if isinstance(group, h5py.Dataset):
        # Check if it represents a multi-dimensional array (e.g., a dense matrix or NumPy array)
        if len(group.shape) >= 2:
            return "NumPy Array or Dense Matrix"
        else:
            return "1D Dataset"

    # Keys in the group
    keys = set(group.keys())

    # CSR or CSC Matrix Check
    if {"data", "indices", "indptr"}.issubset(keys):
        if "shape" in group.attrs:
            return "CSR/CSC Matrix"

    # COO Matrix Check
    elif {"data", "row", "col"}.issubset(keys):
        return "COO Matrix"

    # Unknown format
    return "Unknown Format"


def generate_summary_report(data_dir: str) -> Dict[str, Any]:
    """
    Generate a summary report of the structure and content of an HDF5 file.

    This function analyzes the contents of an HDF5 file specified by `data_dir` and
    generates a summary report. The report includes information about the groups and
    datasets in the file, such as their keys, types, shapes, and formats. It specifically
    handles the 'X' and 'layers' groups in a special manner to provide detailed information
    about their structure, including the matrix format and shape.

    Parameters:
    ----------
    data_dir : str
        The path to the HDF5 file to be analyzed.

    Returns:
    -------
    Dict[str, Any]
        A dictionary containing the summary report of the HDF5 file. The structure of
        the dictionary is as follows:
        - Keys are the names of the top-level groups or datasets.
        - Values are dictionaries that contain information about the keys and types of
          groups, or the shape and dtype of datasets. For example:
            - For groups:
                {
                    "group_keys": List[str],       # List of keys within the group.
                    "group_types": List[type],     # List of types of the corresponding keys.
                    "format": str,                 # (For 'X' or 'layers' group) Matrix format.
                    "shape": tuple,                # (For 'X' or 'layers' group) Shape of the matrix.
                    "sublayer_info": dict          # (For 'layers' group) Information about sub-groups.
                }
            - For datasets:
                {
                    "shape": tuple,                # Shape of the dataset.
                    "dtype": numpy.dtype           # Data type of the dataset (if available).
                }

    Exceptions:
    ----------
    If an error occurs while processing the file, an error message is printed, and
    an empty dictionary is returned.

    Notes:
    -----
    - The function is designed to be flexible and handle both dense (NumPy array) and
      sparse (CSR, CSC, COO) matrix formats.
    - The shape of CSR/CSC matrices is derived from the 'indptr' and 'indices' datasets
      if the 'shape' attribute is not available.
    - This function assumes that the 'X' group represents a matrix and processes it
      accordingly. Similarly, the 'layers' group is expected to contain sub-groups
      that represent different layers or matrices.
    """

    try:
        with h5py.File(data_dir, "r") as file:
            keys = file.keys()
            summary_report: Dict[str, Dict[str, Union[str, Any]]] = {}

            for key in keys:
                group = file[key]

                if isinstance(group, h5py.Group):
                    group_keys = list(group.keys())
                    group_types = [type(group[k]) for k in group_keys]
                    summary_report[key] = {
                        "group_keys": group_keys,
                        "group_types": group_types,
                    }

                    # Process the 'X' group
                    if key == "X":
                        matrix_format = detect_matrix_format(group)
                        summary_report[key]["format"] = matrix_format

                        # Report the shape directly for 'X'
                        if matrix_format in ["CSR/CSC Matrix", "COO Matrix"]:
                            if "shape" in group.attrs:
                                summary_report[key]["shape"] = group.attrs["shape"]
                            else:
                                # Fallback shape calculation (try to avoid if possible)
                                summary_report[key]["shape"] = (
                                    group["indptr"].shape[0] - 1,
                                    max(group["indices"]) + 1,
                                )
                        elif matrix_format == "NumPy Array or Dense Matrix":
                            summary_report[key][
                                "shape"
                            ] = group.shape  # Directly use group.shape

                    # Process the 'layers' group
                    if key == "layers":
                        layer_dict: Dict[str, Dict[str, Any]] = {}
                        for k in file["layers"].keys():
                            sub_group = file[key][k]

                            matrix_format = detect_matrix_format(sub_group)
                            layer_dict[k] = {
                                "format": matrix_format,
                            }

                            # Directly use the shape if available
                            if "shape" in sub_group.attrs:
                                layer_dict[k]["shape"] = sub_group.attrs["shape"]
                            else:
                                if matrix_format == "NumPy Array or Dense Matrix":
                                    layer_dict[k]["shape"] = sub_group.shape
                                elif matrix_format == "CSR/CSC Matrix":
                                    # Only calculate shape manually if absolutely necessary
                                    layer_dict[k]["shape"] = (
                                        sub_group["indptr"].shape[0] - 1,
                                        max(sub_group["indices"]) + 1,
                                    )

                        summary_report[key]["sublayer_info"] = layer_dict

                elif isinstance(group, h5py.Dataset):
                    dtype = group.dtype if hasattr(group, "dtype") else None
                    summary_report[key] = {"shape": group.shape, "dtype": dtype}

        return summary_report

    except Exception as e:
        print(f"Error processing file: {e}")
        return {}


def generate_anndata_report(
    data_dir: str,
):
    """
    Generate and print a detailed report of the contents of an HDF5 file assumed to represent an AnnData object.

    This function analyzes the structure of an HDF5 file at the specified `data_dir` by invoking the
    `generate_summary_report` function. It prints out a formatted report that includes information about
    the file's groups, datasets, and specific matrix formats, such as the main data matrix ('X') and any
    additional layers. The report highlights the structure, data types, shapes, and number of cells and features
    in these matrices, which are common attributes in AnnData objects used in single-cell RNA sequencing (scRNA-seq) analysis.

    Parameters:
    ----------
    data_dir : str
        The path to the HDF5 file to be analyzed.

    Returns:
    -------
    None
        This function does not return a value. Instead, it prints the report directly to the console.

    Notes:
    -----
    - The function is designed to handle HDF5 files structured similarly to AnnData objects, which often contain
      groups like 'X' (the main data matrix) and 'layers' (additional matrices). The 'X' group is typically
      a matrix where rows correspond to cells and columns correspond to features (e.g., genes).
    - The function prints detailed information including:
        - The keys and types of each top-level group.
        - For the 'X' group, it prints the matrix format, shape, and the number of cells and features.
        - For the 'layers' group, it prints information for each sub-layer, including matrix format, shape,
          and the number of cells and features.
    - The output is formatted with bold section titles and is intended for easy readability in a console environment.

    Example:
    -------
    To generate a report for an AnnData HDF5 file:

    >>> generate_anndata_report("path/to/anndata_file.h5")

    This would print a detailed report about the contents and structure of the HDF5 file.
    """

    report = generate_summary_report(data_dir)
    print(f"\033[1mAnndata object checking: {data_dir}\033[0m\n")

    # Print the summary report
    for key, value in report.items():
        print(f"\033[1mPartition: {key}\033[0m\n")

        if "group_keys" in value:
            group_info = {
                "Group Keys": value["group_keys"],
                "Group Types": value["group_types"],
            }
            pprint(group_info, indent=4, width=80, compact=True)

            if key == "X":
                matrix_info = {
                    "Matrix Format": value["format"],
                    "Shape": value.get("shape"),
                }

                # Extract and add number of cells and features from shape
                if "shape" in value:
                    num_cells = value["shape"][0]
                    num_features = value["shape"][1]
                    matrix_info["Number of cells"] = num_cells
                    matrix_info["Number of features"] = num_features

                pprint(matrix_info, indent=4, width=80, compact=True)

            elif key == "layers":
                print("\nSub layer information:\n")
                for k, v in value["sublayer_info"].items():
                    print(f"{k}:")
                    layer_info = {
                        "Matrix Format": v["format"],
                    }

                    # Handle shape and extract number of cells and features if shape is present
                    if "shape" in v:
                        layer_info["Shape"] = v["shape"]
                        num_cells = v["shape"][0]
                        num_features = v["shape"][1]
                        layer_info["Number of cells"] = num_cells
                        layer_info["Number of features"] = num_features

                    if "Cell_num" in v:
                        layer_info["Number of cells"] = v["Cell_num"]
                    if "Feat_num" in v:
                        layer_info["Number of features"] = v["Feat_num"]

                    pprint(layer_info, indent=6, width=80, compact=True)
                    print("")  # Empty line for readability

        else:
            dataset_info = {"Shape": value["shape"], "Dtype": value["dtype"]}
            pprint(dataset_info, indent=4, width=80, compact=True)

        print("---------------------------\n")


def inspect_column_categories(
    data_dir: str,
    dataframe: Literal["obs", "var"],
    columns: List[str],
) -> pd.DataFrame:
    """
    Inspect and retrieve categories or unique values from specified columns within an AnnData-like HDF5 file.

    This function examines specific columns in a given dataframe group (either 'obs' or 'var')
    within an HDF5 file. It decodes the categories for categorical columns or gathers unique values
    for non-categorical columns, then compiles the results into a pandas DataFrame for easy inspection.

    Parameters:
    ----------
    data_dir : str
        The path to the HDF5 file containing the AnnData-like structure.

    dataframe : Literal['obs', 'var']
        The name of the dataframe group within the HDF5 file to inspect. Typically, 'obs' refers to
        observations (cells), and 'var' refers to variables (features/genes).

    columns : List[str]
        A list of column names to inspect within the specified dataframe. These columns may contain
        categorical data or other types of data for which unique values will be retrieved.

    Returns:
    -------
    pd.DataFrame
        A DataFrame where each column corresponds to one of the inspected columns from the HDF5 file.
        The DataFrame contains either decoded category values (for categorical columns) or unique values
        (for non-categorical columns). The DataFrame columns are renamed to indicate that they contain
        unique values.

    Raises:
    ------
    ValueError:
        If the specified dataframe ('obs' or 'var') is not found in the HDF5 file, a ValueError is raised.

    Notes:
    -----
    - If a column is not found in the specified dataframe group, the corresponding DataFrame column
      will contain "Column not found" as its value.
    - The resulting DataFrame is padded with empty strings to ensure all columns have the same length,
      which is equal to the maximum number of unique values found across all specified columns.
    - This function assumes that categorical data in the HDF5 file is stored with a 'categories' attribute,
      and non-categorical data is stored as arrays from which unique values can be derived.

    Example:
    -------
    To inspect the unique values or categories of certain columns in the 'obs' group of an HDF5 file:

    >>> df = inspect_column_categories("path/to/anndata_file.h5", "obs", ["cell_type", "condition"])
    >>> print(df)

    This would output a DataFrame showing the unique values or decoded categories for the specified columns.
    """

    # Initialize an empty dictionary to store decoded categories or unique values
    decoded_data: Dict[str, List[Any]] = {col: [] for col in columns}

    with h5py.File(data_dir, "r") as file:
        # Check if the specified dataframe exists in the file
        if dataframe not in file:
            raise ValueError(f"DataFrame type '{dataframe}' not found in file.")

        df_group = file[dataframe]

        # Iterate over columns and decode categories or gather unique values
        for col in columns:
            if col in df_group:
                data = df_group[col]

                if "categories" in data:
                    # Decode category values if the column has categories
                    decoded_data[col] = [value.decode() for value in data["categories"]]
                else:
                    # For non-categorical columns, gather unique values
                    # Note: Adjust depending on actual data storage
                    decoded_data[col] = list(set(data))
            else:
                # Handle the case where the column is not found
                decoded_data[col] = ["Column not found"]

    # Find the maximum length among all categories or unique values
    max_length = max(len(decoded_data[col]) for col in columns)

    # Pad the lists in the dictionary with empty strings if needed
    for col in columns:
        if len(decoded_data[col]) < max_length:
            decoded_data[col] += [""] * (max_length - len(decoded_data[col]))

    # Create a DataFrame from the decoded data
    df = pd.DataFrame(decoded_data)

    # Update column names with ' unique values'
    df.columns = [f"{col} unique values" for col in columns]

    return df
