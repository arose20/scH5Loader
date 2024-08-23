# h5ld
[![Github_tests](https://img.shields.io/github/actions/workflow/status/arose20/H5py_anndata_checker/Github_tests.yml?branch=main&label=build-test&logo=github&style=plastic)](https://github.com/arose20/H5py_anndata_checker/actions/workflows/Github_tests.yml)

[![python](https://img.shields.io/badge/python-3.9-3776AB?style=plastic&logo=python&logoColor=white)](https://python.org)
[![jupyter](https://img.shields.io/badge/Works%20with-Jupyter-orange?style=plastic&logo=Jupyter)](https://jupyter.org/)



Python tool to investigate H5DF files of single cell data and load in only the subset of interest into an anndata object
***

![https://github.com/ar32/H5py_anndata_checker/resources/Workflow.gif](https://github.com/arose20/H5py_anndata_checker/blob/main/resources/Workflow.gif)


***
## About

- ⚙️: Functions to utalise the h5py package in order to explore and load in single cell data which are stored in .h5ad format.

- 🔍: Exploration can be done without loading the entire data into memory saving on memory overhead and time to load in the entire file just to see what is inside.

- 🔄: Once identifying the data of interest you can then either just load the metadata of interest associated with the cells of interest into a pandas dataframe format or load in only the cells of interest into memory itself in an anndata format.

- 🗄️: This can be useful for large single cell anndata files where you want to know what is inside the file; you only want a subsection of the total data and don't have enough memory to load in all the data and then subsequently slice to desired cells of interest.

- 💿: The goal of these functions is to help the user explore single cell file contents and only load the data of interest instead of all the data to save on memory consumption.

***
## Use case example

If you have a dataset of single cell data you can:

1. **See what is in the data:**

    - what columns are in observations (obs or var compartments)
    - are there any layers
    - is there anything stored in other compartments such as .uns, .obsp, .obsm, .varp, .varm


2. **See the unique values of desired columns stored in the obs compartment:**

    - for the column broad_anno what are all the unique cell types present
    - for the column donor what are all the unique donors in this data


3. **Create just a pandas dataframe of the desired cells of interest to explore the metadata without the counts matrix:**

    - For all cells in haematopoetic lineage only, return columns related to refined annotation, donor, chemistry, age, spatial but don't return any other information
    - Filter multiple columns at once and choose to return the intersection or union
    - Filter for cells of interest and return all columns of information


4. **Load in a subset of the original data which only contains the cells of interest to save on memory:**

    - load in a subset of the original anndata object but for only cells and metadata of interest with their associated counts matrix and any other metadata of interest

#### <ins>Note</ins>
When loading in a slice of the data, since this is designed for single cell data we are assuming you are working with numerical data not string. Therefore numpy arrays are passed into the csr_matrix for construction speed not lists which would favour string values.

This is not designed for .h5ad files that don't have counts information stored in the .X compartment (e.g. latent spaces from PCA or vae methodologies - could be stored in other compartments though)

***
## Installation

To install directly from github run below in the command line - recommend in a virtual environment such as venv or conda:

```bash
pip install git+https://github.com/arose20/H5py_anndata_checker.git
```

To clone and install:

```bash
git clone git+https://github.com/arose20/H5py_anndata_checker.git

cd ./H5py_anndata_checker

pip install -e .
```

To install through requirements.txt:

```bash
pip install -r requirements.txt
```

To further install developmental packages if desired:

```bash
pip install -r requirements_dev.txt
```

***

### Testing

For testing and cleaning code for this repo, the following packages  are used:

- mypy 

- flake8

- pytest

For formatting the ```black``` formatter is used