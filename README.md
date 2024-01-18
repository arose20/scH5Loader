# scarf_tools

single cell anndata reading file tools :scarf:
***
Use h5py to explore siingle cell data stored in an anndata format without loading in memory and then load in only the data you are interested in to save on total memory consumption.

This can be useful for large single cell anndata files where you only want a subsec tion of the data and don't have enough memory to load all the data in and slice.

This goal is to help the user only load the data of interest instead of all the data.

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
