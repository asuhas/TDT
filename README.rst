====================
Treasury DV01 Trader
====================


.. image:: https://img.shields.io/pypi/v/tdt.svg
        :target: https://pypi.python.org/pypi/tdt

.. image:: https://img.shields.io/travis/asuhas/tdt.svg
        :target: https://travis-ci.com/asuhas/tdt

.. image:: https://readthedocs.org/projects/tdt/badge/?version=latest
        :target: https://tdt.readthedocs.io/en/latest/?version=latest
        :alt: Documentation Status




Treasury DV01 Trader - A Python package for on the run bond calculations and analysis



* Free software: MIT license



Features
--------

* Bond pricing and analytics calculations
* DV01 and risk metrics computation
* Principal Component Analysis (PCA) for yield curve analysis
* Market calendar integration
* Plotting utilities for bond analysis

Requirements
-----------

* Python >= 3.11
* numpy >= 1.24.0
* pandas >= 2.0.0
* polars >= 0.19.0
* plotly >= 5.13.0
* statsmodels >= 0.14.0
* scikit-learn >= 1.2.0
* pandas-market-calendars >= 4.1.4
* openbb >= 3.2.0
* jupyter >= 1.0.0
* notebook >= 7.0.0

Development Requirements
----------------------

Development dependencies can be installed using the [dev] extra:

* pytest - Testing
* coverage - Code coverage
* mypy - Type checking
* ruff - Linting
* black - Code formatting
* isort - Import sorting

Interactive Dashboard
-------------------

The package includes interactive Jupyter notebooks with comprehensive dashboards that demonstrate all features. To use these dashboards:

1. Enable Jupyter nbextensions (required for interactive widgets):

.. code-block:: bash

    pip install jupyter_contrib_nbextensions
    jupyter contrib nbextension install --user
    jupyter nbextension enable --py widgetsnbextension

2. Additional dashboard dependencies:

.. code-block:: bash

    pip install ipywidgets>=8.0.0
    pip install jupyterlab_widgets>=3.0.0

3. Verify installation:

.. code-block:: bash

    jupyter nbextension list

Dashboard Features
~~~~~~~~~~~~~~~~

The notebooks in the ``notebooks/`` directory contain interactive dashboards that showcase:

* PCA analysis of rate movements
* Risk metrics monitoring
* DV01 calculations and in sample PnL visualization

To use the dashboards:

1. Navigate to the ``notebooks/`` directory
2. Start Jupyter Notebook:

.. code-block:: bash

    jupyter notebook

3. Open any of the dashboard notebooks
4. Run all cells to initialize the interactive widgets

Note: If widgets are not displaying properly, ensure all nbextensions are properly enabled and try refreshing your browser.


Contributing
-----------

Contributions are welcome! Please read our Contributing Guidelines for details.

License
-------

This project is licensed under the MIT License - see the LICENSE file for details.

Authors
-------

* Suhas Anjaria - *Initial work* - anjaria.suhas@gmail.com

Links
-----

* Homepage: https://github.com/asuhas/tdt
* Bug Reports: https://github.com/asuhas/tdt/issues
* Change Log: https://github.com/asuhas/tdt/blob/master/changelog.md


