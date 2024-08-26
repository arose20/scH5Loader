scH5Loader
==========

.. image:: https://github.com/arose20/scH5Loader/actions/workflows/Github_tests.yml/badge.svg
   :alt: Github_tests
   :target: https://github.com/arose20/scH5Loader/actions/workflows/Github_tests.yml
.. image:: https://img.shields.io/badge/version-0.1.0-blue?logo=githubactions&logoColor=white
   :alt: release version
   :target: https://github.com/arose20/scH5Loader
.. image:: https://img.shields.io/badge/python-3.9-3776AB?style=plastic&logo=python&logoColor=white
   :alt: python
   :target: https://python.org
.. image:: https://img.shields.io/badge/PyPI-0.1.0-3775A9?logo=pypi&logoColor=white
   :alt: PyPI version
   :target: https://pypi.org/project/scH5Loader/
.. image:: https://img.shields.io/badge/Works%20with-Jupyter-orange?style=plastic&logo=Jupyter
   :alt: jupyter
   :target: https://jupyter.org/






Python tool to investigate H5DF files of single cell data and load in  
only the subset of interest into an anndata object  


.. image:: https://github.com/arose20/scH5Loader/blob/main/resources/Workflow.gif
   :alt: Workflow

About
=====

- ⚙️: Functions to utilize the h5py package in order to explore and load in single cell data which are stored in .h5ad format.

- 🔍: Exploration can be done without loading the entire data into memory, saving on memory overhead and time.

- 🔄: Once identifying the data of interest, you can either load just the metadata associated with the cells of interest into a pandas dataframe or load only the cells of interest into memory in an anndata format.

- 🗄️: This can be useful for large single cell anndata files where you want to know what is inside the file; you only want a subsection of the total data and don't have enough memory to load all the data and then subsequently slice to desired cells of interest.

- 💿: The goal of these functions is to help the user explore single cell file contents and only load the data of interest, saving on memory consumption.

Use case example
================

If you have a dataset of single cell data, you can:

1. **See what is in the data:**

    - What columns are in observations (obs or var compartments)?
    - Are there any layers?
    - Is there anything stored in other compartments such as .uns, .obsp, .obsm, .varp, .varm?

2. **See the unique values of desired columns stored in the obs compartment:**

    - For the column broad_anno, what are all the unique cell types present?
    - For the column donor, what are all the unique donors in this data?

3. **Create just a pandas dataframe of the desired cells of interest to explore the metadata without the counts matrix:**

    - For all cells in haematopoetic lineage only, return columns related to refined annotation, donor, chemistry, age, spatial but don't return any other information.
    - Filter multiple columns at once and choose to return the intersection or union.
    - Filter for cells of interest and return all columns of information.

4. **Load in a subset of the original data which only contains the cells of interest to save on memory:**

    - Load in a subset of the original anndata object but for only cells and metadata of interest with their associated counts matrix and any other metadata of interest.

.. note::

    When loading in a slice of the data, since this is designed for single cell data, we are assuming you are working with numerical data not string. Therefore numpy arrays are passed into the csr_matrix for construction speed, not lists which would favor string values.

    This is not designed for .h5ad files that don't have counts information stored in the .X compartment (e.g., latent spaces from PCA or VAE methodologies - could be stored in other compartments though).

Installation
============
To install through pypi, run the command below (recommended in a virtual environment such as venv or conda):

.. code-block:: bash

   pip install scH5Loader


To install directly from GitHub, run the command below (recommended in a virtual environment such as venv or conda):

.. code-block:: bash

   pip install git+https://github.com/arose20/scH5Loader.git

To clone and install:

.. code-block:: bash

   git clone git+https://github.com/arose20/scH5Loader.git
   cd ./scH5Loader
   pip install -e .

To install through `requirements.txt`:

.. code-block:: bash

   pip install -r requirements.txt

To further install developmental packages if desired:

.. code-block:: bash

   pip install -r requirements_dev.txt

Testing
=======

For testing and cleaning code for this repo, the following packages are used:

- mypy
- flake8
- pytest

For formatting, the ``black`` formatter is used.
