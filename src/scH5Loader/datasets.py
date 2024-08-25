import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse import lil_matrix
import anndata
from concurrent.futures import ThreadPoolExecutor
from numba import njit
import warnings

warnings.filterwarnings("ignore")
# Set random seed for reproducibility
np.random.seed(42)


def generate_anndata_object_1(
    data_size: int = 5000, num_features: int = 36000, batch_size: int = 1000
) -> anndata.AnnData:
    # Create column values using vectorized operations
    anno_LVL2_values = np.random.choice(
        [
            "macrophage",
            "fibroblast",
            "monocyte",
            "hepatocyte",
            "erythroid",
            "cardiomyocyte",
        ],
        size=data_size,
        p=[0.3, 0.3, 0.1, 0.1, 0.1, 0.1],
    )
    anno_LVL1_mapping = {
        "haematopoetic": ["macrophage", "monocyte", "erythroid"],
        "stromal": ["cardiomyocyte", "fibroblast"],
        "epithelial": ["hepatocyte"],
    }
    anno_LVL1_values = np.array(
        [
            next(key for key, value in anno_LVL1_mapping.items() if cell_type in value)
            for cell_type in anno_LVL2_values
        ]
    )

    # New layers with preallocated memory
    layer1_values = np.random.randn(data_size, num_features).astype(np.float32)
    layer2_values = np.random.randn(data_size, num_features).astype(np.float32)

    study_values = np.full(data_size, "Test_study")

    biological_unit_values = np.random.choice(
        ["cell", "nuclei"], size=data_size, p=[0.7, 0.3]
    )

    # Use np.random.choice with replace=True for 'donor' column
    donor_values = np.random.choice(
        ["Donor1", "Donor2", "Donor3", "Donor4", "Donor5"],
        size=data_size,
        replace=True,
        p=[0.2, 0.2, 0.2, 0.2, 0.2],
    )

    # Create the AnnData object without preallocating memory
    adata = anndata.AnnData(
        X=lil_matrix((data_size, num_features), dtype=np.float32),
        layers={"layer1": layer1_values, "layer2": layer2_values},
        obs=pd.DataFrame(
            {
                "anno_LVL1": anno_LVL1_values,
                "anno_LVL2": anno_LVL2_values,
                "study": study_values,
                "donor": donor_values,
                "biological_unit": biological_unit_values,
            }
        ),
    )

    # Set index explicitly to avoid ImplicitModificationWarning
    adata.obs.index = adata.obs.index.astype(str)

    # Function to fill a batch of the sparse matrix using numba
    @njit(parallel=True)
    def fill_batch_numba(indices, col_indices, row_indices, values, X):
        for i in range(indices.shape[0]):
            for j in range(col_indices.shape[1]):
                X[indices[i], col_indices[i, j]] += values[i]

    # Function to fill a batch of the sparse matrix
    def fill_batch(start, end, X):
        indices = np.random.choice(data_size, size=batch_size, replace=True)
        col_indices = np.tile(np.arange(start, end), (indices.shape[0], 1))
        row_indices = np.repeat(indices, end - start)
        values = np.random.uniform(1, 100, size=row_indices.shape[0]).astype(np.float32)

        fill_batch_numba(indices, col_indices, row_indices, values, X)

    # Use ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor() as executor:
        for i in range(0, num_features, batch_size):
            executor.submit(fill_batch, i, min(i + batch_size, num_features), adata.X)

    # Convert the lil_matrix to CSR format
    adata.X = adata.X.tocsr()

    # Randomly set 40% of values to 0 using vectorized operation
    mask = np.random.choice([1, 0], size=adata.X.nnz, p=[0.6, 0.4])

    # Apply the mask to the sparse matrix data
    adata.X.data = adata.X.data * mask

    # Make layer2 a csr matrix
    adata.layers["layer2"] = sparse.csr_matrix(adata.layers["layer2"])

    # Make var columns for testing
    adata.var["high_var"] = np.random.choice(
        ["yes", "no"], size=len(adata.var), p=[0.6, 0.4]
    )
    adata.var["hgnc"] = "template"

    # update index in obs for unique string
    adata.obs.index = "check_index_" + adata.obs.index

    # update index in var for unique string
    adata.var.index = "check_var_" + adata.var.index

    return adata
