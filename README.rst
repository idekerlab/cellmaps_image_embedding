=============================================
Cell Maps ImmunoFluorescent Image Embedder
=============================================
The Cell Maps Image Embedder is part of the Cell Mapping Toolkit

|a| |b| |c| |d|

.. |a| image:: https://img.shields.io/pypi/v/cellmaps_image_embedding.svg
        :target: https://pypi.python.org/pypi/cellmaps_image_embedding


.. |b| image:: https://app.travis-ci.com/idekerlab/cellmaps_image_embedding.svg?branch=main
        :target: https://app.travis-ci.com/idekerlab/cellmaps_image_embedding


.. |c| image:: https://readthedocs.org/projects/cellmaps-image-embedding/badge/?version=latest
        :target: https://cellmaps-image-embedding.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status


.. |d| image:: https://zenodo.org/badge/618547854.svg
        :target: https://zenodo.org/doi/10.5281/zenodo.10607452
        :alt: Zenodo DOI badge


Generate embeddings from ImmunoFluorescent image data from `Human Protein Atlas <https://www.proteinatlas.org/>`__

* Free software: MIT license
* Documentation: https://cellmaps-image-embedding.readthedocs.io.

Dependencies
------------

* `cellmaps_utils <https://pypi.org/project/cellmaps-utils>`__
* `tqdm <https://pypi.org/project/tqdm>`__
* `numpy <https://pypi.org/project/numpy>`__
* `pandas>=0.23.1 <https://pypi.org/project/pandas>`__
* `torch <https://pypi.org/project/torch>`__
* `torchvision <https://pypi.org/project/torchvision>`__
* `opencv-python <https://pypi.org/project/opencv-python>`__
* `mlcrate <https://pypi.org/project/mlcrate>`__
* `scikit-image <https://pypi.org/project/scikit-image>`__
* `scikit-learn>=0.19.0 <https://pypi.org/project/scikit-learn>`__
* `Pillow <https://pypi.org/project/Pillow>`__

Compatibility
-------------

* Python 3.8 - 3.11

Installation
------------

.. code-block::

   git clone https://github.com/idekerlab/cellmaps_image_embedding
   cd cellmaps_image_embedding
   pip install -r requirements_dev.txt
   make dist
   pip install dist/cellmaps_image_embedding*whl


Run **make** command with no arguments to see other build/deploy options including creation of Docker image

.. code-block::

   make

Output:

.. code-block::

   clean                remove all build, test, coverage and Python artifacts
   clean-build          remove build artifacts
   clean-pyc            remove Python file artifacts
   clean-test           remove test and coverage artifacts
   lint                 check style with flake8
   test                 run tests quickly with the default Python
   test-all             run tests on every Python version with tox
   coverage             check code coverage quickly with the default Python
   docs                 generate Sphinx HTML documentation, including API docs
   servedocs            compile the docs watching for changes
   testrelease          package and upload a TEST release
   release              package and upload a release
   dist                 builds source and wheel package
   install              install the package to the active Python's site-packages
   dockerbuild          build docker image and store in local repository
   dockerpush           push image to dockerhub


Before running tests, please install: ``pip install -r requirements_dev.txt``.

For developers
-------------------------------------------


To deploy development versions of this package
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Below are steps to make changes to this code base, deploy, and then run
against those changes.

#. Make changes

   Modify code in this repo as desired

#. Build and deploy

.. code-block::

    # From base directory of this repo cellmaps_image_embedding
    pip uninstall cellmaps_image_embedding -y ; make clean dist; pip install dist/cellmaps_image_embedding*whl



Needed files
------------

The output directory for the image downloads is required (see `Cell Maps Image Downloader <https://github.com/idekerlab/cellmaps_imagedownloader/>`__). Optionally, a path to the image embedding model can be provided.

Usage
-----

For information invoke :code:`cellmaps_image_embeddingcmd.py -h`


**Example usage**

.. code-block::

   cellmaps_image_embeddingcmd.py ./cellmaps_image_embedding_outdir --inputdir ./cellmaps_imagedownloader_outdir


Via Docker
~~~~~~~~~~~~~~~~~~~~~~

**Example usage**


.. code-block::

   Coming soon...

Cite
-------

If you find this tool useful, please cite:

Lenkiewicz, J., Churas, C., Hu, M., Qian, G., Jain, M., Levinson, M. A., ... & Schaffer, L. V. (2025). Cell Mapping Toolkit: An end-to-end pipeline for mapping subcellular organization. Bioinformatics, 41(6), btaf205.

Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
.. _NDEx: http://www.ndexbio.org
