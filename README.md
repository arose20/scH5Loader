# H5py_anndata_checker
Use h5py to explore without loading in memory and then load in a slice to save on total memory consumption

This can be useful for files which are typically loaded into anndata formats such as .h5ad files for single cell work in python using scanpy 

When loadig in a slice of the data, we are assuming you are working with numerical data not string so numpy arrays are passed into the csr_matrix for speed. 