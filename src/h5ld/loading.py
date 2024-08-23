import numpy as np
from numpy.typing import NDArray
import pandas as pd
import h5py
import anndata
from anndata._io.h5ad import read_elem
from scipy.sparse import csr_matrix
from tqdm import tqdm

from typing import List, Dict, Any, Literal, Tuple
import warnings

from .exploration import detect_matrix_format

# from .exploration import *  # noqa

warnings.filterwarnings("ignore", category=FutureWarning)


def extract_dataframe(
    file,
    dataframe: Literal["obs", "var"],
    filter_dict: Dict[str, List[str]],
    additional_cols: List[str],
    filter_method: Literal["intersection", "union"] = "intersection",
):
    """
    Extract and filter a dataframe from an HDF5 file based on specified criteria.

    This function extracts a dataframe ('obs' or 'var') from an HDF5 file, filters it based
    on criteria specified in `filter_dict`, and optionally adds additional columns. The filtering
    can be performed using either an intersection or union method, depending on the `filter_method` parameter.

    Parameters:
    ----------
    file : h5py.File
        An open HDF5 file handle from which to extract the data.

    dataframe : Literal['obs', 'var']
        The name of the dataframe to extract from the HDF5 file. Typically, 'obs' refers to
        observations (cells) and 'var' refers to variables (features/genes).

    filter_dict : Dict[str, List[str]]
        A dictionary where keys are column names in the dataframe and values are lists of
        filter values. Only rows where the column values match the filter values will be retained.

    additional_cols : List[str]
        A list of additional column names to extract from the dataframe and include in the
        output DataFrame, after filtering.

    filter_method : Literal['intersection', 'union'], default='intersection'
        The method to use when combining filters:
        - 'intersection': Only include rows that match all filter criteria (AND logic).
        - 'union': Include rows that match any filter criteria (OR logic).

    Returns:
    -------
    pd.DataFrame
        A pandas DataFrame containing the filtered and optionally extended data. The DataFrame
        includes the filtered columns specified in `filter_dict` and any additional columns
        specified in `additional_cols`. The original index values and positions are also included.

    Raises:
    ------
    ValueError:
        - If the specified `dataframe` is not found in the HDF5 file.
        - If any values in `filter_dict` are not found in the corresponding columns.
        - If an invalid `filter_method` is provided.

    Notes:
    -----
    - The original index values and positions are included in the output DataFrame to maintain
      traceability back to the original data.
    - The function assumes that the `_index` column is encoded as a byte string, which is decoded
      to UTF-8 for processing.
    - The output DataFrame columns are reordered according to the `column-order` attribute in
      the HDF5 file, if available.

    Example:
    -------
    To extract and filter data from the 'obs' dataframe based on certain criteria and include
    additional columns:

    >>> df = extract_dataframe(
            file=h5py.File("path/to/anndata_file.h5", "r"),
            dataframe="obs",
            filter_dict={"cell_type": ["B cell", "T cell"]},
            additional_cols=["age", "condition"],
            filter_method="intersection"
        )
    >>> print(df)

    This would output a DataFrame with only the cells of type "B cell" or "T cell", and includes
    the 'age' and 'condition' columns.
    """

    # Retrieve and decode index
    original_index_values = np.vectorize(lambda x: x.decode("utf-8"))(
        np.array(file[dataframe]["_index"], dtype=object)
    )

    # Create a DataFrame with both original index values and their positions
    original_index_df = pd.DataFrame(
        {
            "Original_index_value": original_index_values,
            "Original_index_position": np.arange(len(original_index_values)),
        },
        index=original_index_values,
    )

    indexes = []

    # Apply filtering for each column specified in filter_dict
    for filter_column, filter_values in filter_dict.items():
        col_data = pd.DataFrame(read_elem(file[dataframe][filter_column])).astype(str)
        col_data = col_data.loc[col_data.iloc[:, 0].isin(filter_values)]
        missing_values = set(filter_values) - set(col_data.iloc[:, 0])

        if missing_values:
            raise ValueError(
                f"Missing values in filter_column '{filter_column}': {missing_values}"
            )

        col_data.rename(columns={0: filter_column}, inplace=True)
        col_data.index = original_index_values[col_data.index]

        indexes.append(col_data.index)

    # Combine filtered DataFrames based on filter_method
    if filter_method == "intersection":
        # Intersect
        intersection = set(indexes[0])
        for lst in indexes[1:]:
            intersection &= set(lst)

        index = list(intersection)

        if len(index) < 1:
            raise ValueError(
                "\u26A0\uFE0F There are no cells which match all input parameters. \u274C\nPlease review columns and column values selected for intersection."
            )

    elif filter_method == "union":
        # Union
        union = set()
        for lst in indexes:
            union |= set(lst)

        index = list(union)

    else:
        raise ValueError(
            "Invalid filter_method. Choose either 'intersection' or 'union'."
        )

    data_frame = original_index_df.loc[original_index_df.index.isin(index)].copy()

    for col in list(filter_dict.keys()) + additional_cols:
        col_data = pd.DataFrame(read_elem(file[dataframe][col]))
        col_data.rename(columns={0: col}, inplace=True)
        col_data.index = original_index_values
        col_data = col_data[col_data.index.isin(index)]
        data_frame.loc[:, col] = col_data[col]

    return data_frame


def create_dataframe_subset(
    data_dir: str,
    dataframe: Literal["obs", "var"],
    filter_dict: Dict[
        str, List[str]
    ],  # Dictionary to specify filter columns and their values
    additional_cols: List[str],
    filter_method: Literal[
        "intersection", "union"
    ] = "intersection",  # Method to apply filtering
) -> pd.DataFrame:
    """
    Create a subset of a dataframe from an HDF5 file based on specified filtering criteria.

    This function loads an HDF5 file, extracts a specific dataframe ('obs' or 'var'), applies filtering
    based on the criteria provided in `filter_dict`, and optionally includes additional columns.
    The filtering can be performed using either an intersection or union method.

    Parameters:
    ----------
    data_dir : str
        The path to the HDF5 file containing the AnnData-like structure.

    dataframe : Literal['obs', 'var']
        The name of the dataframe to subset from the HDF5 file. Typically, 'obs' refers to
        observations (cells) and 'var' refers to variables (features/genes).

    filter_dict : Dict[str, List[str]]
        A dictionary where keys are column names in the dataframe and values are lists of
        filter values. Only rows where the column values match the filter values will be retained.

    additional_cols : List[str]
        A list of additional column names to extract from the dataframe and include in the
        subset after filtering.

    filter_method : Literal['intersection', 'union'], default='intersection'
        The method to use when combining the filter criteria:
        - 'intersection': Only include rows that match all filter criteria (AND logic).
        - 'union': Include rows that match any filter criteria (OR logic).

    Returns:
    -------
    pd.DataFrame
        A pandas DataFrame containing the filtered subset of data. The DataFrame includes
        the filtered columns specified in `filter_dict` and any additional columns specified
        in `additional_cols`.

    Example:
    -------
    To create a subset of the 'obs' dataframe with specific filter criteria and include additional columns:

    >>> subset_df = create_dataframe_subset(
            data_dir="path/to/anndata_file.h5",
            dataframe="obs",
            filter_dict={"cell_type": ["B cell", "T cell"], "condition": ["treated"]},
            additional_cols=["age", "gender"],
            filter_method="intersection"
        )
    >>> print(subset_df)

    This would extract a subset of the 'obs' dataframe, filtering for rows where 'cell_type'
    is either "B cell" or "T cell", and 'condition' is "treated", while also including the
    'age' and 'gender' columns.
    """

    with h5py.File(data_dir, "r") as file:
        subset_dataframe = extract_dataframe(
            file, dataframe, filter_dict, additional_cols, filter_method
        )

    return subset_dataframe


def sparse_grab_filtered_values(
    rows_to_load: List[int],
    cols_to_load: List[int],
    data_dset: NDArray[Any],
    indices_dset: NDArray[Any],
    indptr_dset: NDArray[Any],
    description: str,
) -> Tuple[NDArray[Any], NDArray[Any], NDArray[Any]]:
    """
    Extract data for specific rows and columns from a sparse matrix in Compressed Sparse Row (CSR) or
    Compressed Sparse Column (CSC) format.

    This function filters and retrieves the data from a sparse matrix stored in CSR or CSC format,
    based on the specified rows and columns. The data, indices, and indptr arrays corresponding to
    the filtered subset are returned, allowing for efficient subsetting of the matrix.

    Parameters:
    ----------
    rows_to_load : List[int]
        A list of row indices to extract from the sparse matrix. The indices should correspond
        to the rows of interest in the original matrix.

    cols_to_load : List[int]
        A list of column indices to extract from the sparse matrix. The indices should correspond
        to the columns of interest in the original matrix.

    data_dset : NDArray[Any]
        The 'data' array from the sparse matrix dataset, which contains the non-zero values of
        the matrix.

    indices_dset : NDArray[Any]
        The 'indices' array from the sparse matrix dataset, which contains the column (for CSR)
        or row (for CSC) indices corresponding to each non-zero value in the 'data' array.

    indptr_dset : NDArray[Any]
        The 'indptr' array from the sparse matrix dataset, which defines the boundaries of the
        rows (for CSR) or columns (for CSC) in the 'data' and 'indices' arrays.

    description : str
        A descriptive string for the operation, which is used for display purposes in progress
        indicators.

    Returns:
    -------
    Tuple[NDArray[Any], NDArray[Any], NDArray[Any]]
        - selected_data: A numpy array containing the filtered non-zero values from the 'data' array.
        - selected_indices: A numpy array containing the filtered column (for CSR) or row (for CSC)
          indices corresponding to the 'selected_data'.
        - selected_indptr: A numpy array defining the boundaries of the filtered rows (for CSR)
          or columns (for CSC) in the 'selected_data' and 'selected_indices' arrays.
    """
    # Initialize lists to store intermediate results
    selected_data: List[Any] = []
    selected_indices: List[Any] = []
    selected_indptr: List[int] = [0]

    # Create mappings for row and column indices - row_map = {orig_idx: new_idx for new_idx, orig_idx in enumerate(rows_to_load)}
    col_map = {orig_idx: new_idx for new_idx, orig_idx in enumerate(cols_to_load)}

    # Process each row
    for row_idx in tqdm(
        rows_to_load,
        desc=f"Processing Rows and Columns for {description}",
        unit="row",
        position=0,
        leave=True,
    ):
        start_idx = indptr_dset[row_idx]
        end_idx = indptr_dset[row_idx + 1]

        # Extract and filter the indices and data for the columns of interest
        row_indices = indices_dset[start_idx:end_idx]
        row_data = data_dset[start_idx:end_idx]

        filtered_indices = [
            col_map.get(idx, -1) for idx in row_indices if idx in col_map
        ]
        filtered_data = [
            row_data[i] for i in range(len(row_indices)) if row_indices[i] in col_map
        ]

        # Append filtered data to lists
        selected_data.extend(filtered_data)
        selected_indices.extend(filtered_indices)
        selected_indptr.append(selected_indptr[-1] + len(filtered_data))

    # Convert lists to numpy arrays
    selected_data_out = np.array(selected_data, dtype=data_dset.dtype)
    selected_indices_out = np.array(selected_indices, dtype=indices_dset.dtype)
    selected_indptr_out = np.array(selected_indptr, dtype=indptr_dset.dtype)

    return selected_data_out, selected_indices_out, selected_indptr_out


def create_anndata_subset(
    data_dir: str,
    obs_filter_dict: Dict[str, List[str]],
    obs_additional_cols_keep: List[str],
    obs_filter_method: Literal["intersection", "union"],
    var_filter_dict: Dict[str, List[str]],
    var_additional_cols_keep: List[str],
    var_filter_method: Literal["intersection", "union"],
    filter_layers: List[str],
    filter_obsm: List[str],
    filter_obsp: List[str],
    filter_varm: List[str],
    filter_varp: List[str],
    filter_uns: List[str],
    keep_layers: bool = False,
    keep_obsm: bool = False,
    keep_obsp: bool = False,
    keep_varm: bool = False,
    keep_varp: bool = False,
    keep_uns: bool = False,
):
    with h5py.File(data_dir, "r") as file:
        if obs_filter_dict:
            obs_dataframe = extract_dataframe(
                file,
                "obs",
                obs_filter_dict,
                obs_additional_cols_keep,
                obs_filter_method,
            )

            # List of row positions to load
            obs_rows_to_load = obs_dataframe["Original_index_position"].values
            # display(obs_rows_to_load)

        if var_filter_dict:
            var_dataframe = extract_dataframe(
                file,
                "var",
                var_filter_dict,
                var_additional_cols_keep,
                var_filter_method,
            )

            # List of row positions to load
            var_cols_to_load = var_dataframe["Original_index_position"].values
            # display(var_rows_to_load)

        matrix_format = detect_matrix_format(file["X"])

        if matrix_format in [
            "CSR/CSC Matrix",
            "COO Matrix",
        ]:  # first test with csr, check later for csc and coo
            # Assign variables to query
            data_dset = file["X"]["data"]
            indices_dset = file["X"]["indices"]
            indptr_dset = file["X"]["indptr"]

            # Create the subset matrix
            print(
                "Look up cells of interest:  \U0001F50D",
                flush=True,
            )

            # Extract filtered data, indices, and indptr
            (
                filtered_data,
                filtered_indices,
                filtered_indptr,
            ) = sparse_grab_filtered_values(
                obs_rows_to_load,
                var_cols_to_load,
                data_dset,
                indices_dset,
                indptr_dset,
                "filtered data",
            )

            num_filtered_rows = len(obs_rows_to_load)
            num_filtered_cols = len(var_cols_to_load)

            # Create the subset matrix
            print(
                "Constructing data into csr_matrix format:  \U0001F527",
                flush=True,
            )
            subset_matrix = csr_matrix(
                (filtered_data, filtered_indices, filtered_indptr),
                shape=(num_filtered_rows, num_filtered_cols),
                dtype=data_dset.dtype,
            )
            print("Construction complete \u2705")

        # requires testing
        elif matrix_format == "NumPy Array or Dense Matrix":
            subset_matrix = file["X"][obs_rows_to_load, :][:, var_cols_to_load]

        obs_dataframe.index = obs_dataframe["Original_index_value"].copy()
        obs_dataframe.index.name = None

        var_dataframe.index = var_dataframe["Original_index_value"].copy()
        var_dataframe.index.name = None

        if keep_layers:
            if not filter_layers:
                layers = {}
                for x in file["layers"].keys():
                    # Assign variables to query
                    data_dset = file["layers"][x]["data"]
                    indices_dset = file["layers"][x]["indices"]
                    indptr_dset = file["layers"][x]["indptr"]

                    name = f"layer {x} data"

                    (
                        selected_rows_data,
                        selected_rows_indices,
                        selected_rows_indptr,
                    ) = sparse_grab_filtered_values(
                        obs_rows_to_load,
                        var_cols_to_load,
                        data_dset,
                        indices_dset,
                        indptr_dset,
                        name,
                    )

                    # Create csr_matrix directly from NumPy arrays
                    print(
                        "Constructing data into csr_matrix format:  \U0001F527",
                        flush=True,
                    )
                    layers[x] = csr_matrix(
                        (
                            selected_rows_data,
                            selected_rows_indices,
                            selected_rows_indptr,
                        ),
                        shape=(len(obs_rows_to_load), len(var_cols_to_load)),
                        dtype=file["layers"][x]["data"].dtype,
                    )
                    print("Construction complete \u2705")

            else:
                layers = {}
                for x in (
                    value for value in filter_layers if value in file["layers"].keys()
                ):
                    # Assign variables to query
                    data_dset = file["layers"][x]["data"]
                    indices_dset = file["layers"][x]["indices"]
                    indptr_dset = file["layers"][x]["indptr"]

                    name = f"layer {x} data"

                    (
                        selected_rows_data,
                        selected_rows_indices,
                        selected_rows_indptr,
                    ) = sparse_grab_filtered_values(
                        obs_rows_to_load,
                        var_cols_to_load,
                        data_dset,
                        indices_dset,
                        indptr_dset,
                        name,
                    )

                    # Create csr_matrix directly from NumPy arrays
                    print(
                        "Constructing data into csr_matrix format:  \U0001F527",
                        flush=True,
                    )
                    layers[x] = csr_matrix(
                        (
                            selected_rows_data,
                            selected_rows_indices,
                            selected_rows_indptr,
                        ),
                        shape=(len(obs_rows_to_load), len(var_cols_to_load)),
                        dtype=file["layers"][x]["data"].dtype,
                    )
                    print("Construction complete \u2705")

        else:
            layers = None

        if keep_obsm:
            if not filter_obsm:
                obsm = {
                    x: anndata._io.h5ad.read_elem(file["obsm"][x])[obs_rows_to_load]
                    for x in file["obsm"].keys()
                }
            else:
                obsm = {
                    x: anndata._io.h5ad.read_elem(file["obsm"][x])[obs_rows_to_load]
                    for x in filter_obsm
                    if x in file["obsm"].keys()
                }
        else:
            obsm = None

        if keep_obsp:
            if not filter_obsp:
                obsp = {
                    x: anndata._io.h5ad.read_elem(file["obsp"][x])[obs_rows_to_load][
                        :, obs_rows_to_load
                    ]
                    for x in file["obsp"].keys()
                }
            else:
                obsp = {
                    x: anndata._io.h5ad.read_elem(file["obsp"][x])[obs_rows_to_load][
                        :, obs_rows_to_load
                    ]
                    for x in filter_obsp
                    if x in file["obsp"].keys()
                }
        else:
            obsp = None

        if keep_varm:
            if not filter_varm:
                varm = {
                    x: anndata._io.h5ad.read_elem(file["varm"])[var_cols_to_load]
                    for x in file["varm"].keys()
                }
            else:
                varm = {
                    x: anndata._io.h5ad.read_elem(file["varm"][x])[var_cols_to_load]
                    for x in filter_varm
                    if x in file["varm"].keys()
                }
        else:
            varm = None

        if keep_varp:
            if not filter_varp:
                varp = {
                    x: anndata._io.h5ad.read_elem(file["varp"][x])[var_cols_to_load][
                        :, var_cols_to_load
                    ]
                    for x in file["varp"].keys()
                }
            else:
                varp = {
                    x: anndata._io.h5ad.read_elem(file["varp"][x])[var_cols_to_load][
                        :, var_cols_to_load
                    ]
                    for x in filter_varp
                    if x in file["varp"].keys()
                }
        else:
            varp = None

        if keep_uns:
            if not filter_uns:
                uns = anndata._io.h5ad.read_elem(file["uns"])
            else:
                uns = {
                    x: anndata._io.h5ad.read_elem(file["uns"][x])
                    for x in filter_uns
                    if x in file["uns"].keys()
                }
        else:
            uns = None

        adata = anndata.AnnData(
            X=subset_matrix,
            obs=obs_dataframe,
            var=var_dataframe,
            layers=layers,
            obsm=obsm,
            obsp=obsp,
            varm=varm,
            varp=varp,
            uns=uns,
        )

        return adata
