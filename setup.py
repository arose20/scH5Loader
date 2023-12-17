from setuptools import setup, find_packages

setup(name='to_name',
      version='0.1.0',
      description='exploration of h5ad single-cell objects and partial data loader',
      url='https://github.com/arose20/H5py_anndata_checker',
      author='Antony Rose',
      author_email='ar32@sanger.ac.uk',
      packages=find_packages(),
      install_requires=[
          "numpy",
          "pandas",
          "scanpy",
          "anndata",
          "h5py",
          "scipy",
          "tqdm"
      ])
