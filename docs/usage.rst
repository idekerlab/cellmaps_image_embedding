=====
Usage
=====

Generates image embeddings of immunofluorescent labeled images from the `Human Protein Atlas`_ that were downloaded by the cellmaps_imagedownloader_ package.

In a project
--------------

To use cellmaps_image_embedding in a project::

    import cellmaps_image_embedding


On the command line
---------------------

For information invoke :code:`cellmaps_image_embeddingcmd.py -h`

**Usage**

.. code-block::

  cellmaps_image_embeddingcmd.py [outdir] [--inputdir IMAGEDOWNLOADER_OUT_DIR] [OPTIONS]

**Arguments**

- ``outdir``
    The directory where the output will be written to.

*Required*

- ``--inputdir``
    Directory with blue, red, yellow, and green image directories (output of cellmaps_image_downloader package).

*Optional*

- ``--model_path``
    URL or path to model file for image embedding. If left unspecified then the winning Densenet model
    from `Human Protein Atlas Kaggle Challenge`_ is used

- ``--provenance``
    Path to a JSON file containing FAIRSCAPE provenance metadata (name, project, organization, etc.). This file becomes
    mandatory when the directory specified by ``--inputdir`` does not include ``ro-crate-metadata.json`` and lets the
    tool run on inputs that were not packaged as a RO-Crate.

- ``--name``
    Name of this run, needed for FAIRSCAPE. If unset, name value from the directory specified by --inputdir will be used.

- ``--organization_name``
    Name of the organization running this tool, needed for FAIRSCAPE. If unset, the organization name specified in --inputdir directory will be used.

- ``--project_name``
    Name of the project running this tool, needed for FAIRSCAPE. If unset, the project name specified in --input directory will be used.

- ``--fold``
    Image node attribute file fold to use. Default is 1.

- ``--fake_embedder``
    If set, generate fake embedding. This mode is only for testing/troubleshooting; the resulting ``image_emd.tsv`` contains
    intentionally random vectors and is **not** suitable for downstream scientific analysis.

- ``--dimensions``
    Dimensions of generated embedding vector. Default is 1024.

- ``--suffix``
    Suffix for image files. Default is .jpg.

- ``--logconf``
    Path to the Python logging configuration file in the specified format. Setting this overrides the -v parameter which uses the default logger.

- ``--verbose``, ``-vv``
    Increases verbosity of logger to standard error for log messages in this module. Logging levels: -v = ERROR, -vv = WARNING, -vvv = INFO, -vvvv = DEBUG, -vvvvv = NOTSET. Default is no logging.

- ``--version``
    Display the version of the package.

**Example usage**

The output directory for the image downloads is required (see `Cell Maps Image Downloader <https://github.com/idekerlab/cellmaps_imagedownloader/>`__). Optionally, a path to the image embedding model can be provided.

.. code-block::

   cellmaps_image_embeddingcmd.py ./cellmaps_image_embedding_outdir --inputdir ./cellmaps_imagedownloader_outdir --fold 1

Via Docker
---------------

**Example usage**


.. code-block::

   Coming soon...

.. _Human Protein Atlas: https://www.proteinatlas.org/
.. _cellmaps_imagedownloader: https://cellmaps-imagedownloader.readthedocs.io
.. _Human Protein Atlas Kaggle Challenge: https://github.com/CellProfiling/densenet
