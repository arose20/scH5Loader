# H5py_anndata_checker
***
Use h5py to explore without loading in memory and then load in a slice to save on total memory consumption

This can be useful for files which are typically loaded into anndata formats such as .h5ad files for single cell work in python using scanpy 

This work goal is to help the user only load the data of interest instead of all the data.

***
## Use case example:

If you have a dataset of single cell data you can:

- See what is in the data
- See the unique values of desired columns stored in the obs compartment
- Create just a pandas dataframe of the desired cells of interest to explore the metadata without the counts matrix
- Load in a subset of the original data which only contains the cells of interest to save on memory 

***
#### Note:
When loadig in a slice of the data, we are assuming you are working with numerical data not string so numpy arrays are passed into the csr_matrix for speed. 