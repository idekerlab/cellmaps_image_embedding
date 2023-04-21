#! /usr/bin/env python

import os
import csv
import time
import logging
import random
import subprocess
import cellmaps_image_embedding
from cellmaps_utils import logutils
from cellmaps_image_embedding.exceptions import CellMapsImageEmbeddingError

logger = logging.getLogger(__name__)


class CellmapsImageEmbeddingRunner(object):
    """
    Class to run algorithm
    """
    def __init__(self, outdir=None,
                 imagedir=None,
                 image_gene_node_attributes=None,
                 pythonbinary='/opt/conda/bin/python',
                 predict='/opt/densenet/predict/predict_d121.py',
                 model_path='/opt/densenet/models/model.pth',
                 suffix='jpg'):
        """
        Constructor

        :param exitcode: value to return via :py:meth:`.CellmapsImageEmbeddingRunner.run` method
        :type int:
        """
        logger.debug('In constructor')
        self._outdir = outdir
        self._imagedir = imagedir
        self._image_gene_node_attributes = image_gene_node_attributes
        self._start_time = int(time.time())
        self._dimensions = 1024
        self._pythonbinary = pythonbinary
        self._predict = predict
        self._model_path = model_path
        self._suffix = suffix

    def _run_cmd(self, cmd):
        """
        Runs hidef command as a command line process
        :param cmd_to_run: command to run as list
        :type cmd_to_run: list
        :return: (return code, standard out, standard error)
        :rtype: tuple
        """
        p = subprocess.Popen(cmd,
                             stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE)

        out, err = p.communicate()

        return p.returncode, out, err

    def _get_image_id_list(self):
        """
        Looks at red directory under image directory to
        get a list of image ids which are the file names
        in that directory with last ``_`` and everything to
        the right of it removed from the file name
        :return:
        """
        image_set = set()
        red_image_dir = os.path.join(self._imagedir, 'red')
        for entry in os.listdir(red_image_dir):
            if not entry.endswith(self._suffix):
                continue
            if not os.path.isfile(os.path.join(red_image_dir, entry)):
                continue
            image_set.add(entry[: entry.rfind('_')])
        return list(image_set)

    def _get_image_sublist(self, image_id_list=None,
                           chunk_size=3):
        """
        Gets list of image ids as chunks of smaller
        lists

        :param image_id_list: Original full list
        :type image_id_list: list
        :param chunk_size: Size of chunk
        :type chunk_size: 3
        :return: image ids
        :rtype: list
        """
        for i in range(0, len(image_id_list), chunk_size):
            yield image_id_list[i:i + chunk_size]

    def run(self):
        """
        Runs cellmaps_image_embedding


        :return:
        """
        logger.debug('In run method')
        if self._outdir is None:
            raise CellMapsImageEmbeddingError('outdir must be set')

        if not os.path.isdir(self._outdir):
            os.makedirs(self._outdir, mode=0o755)

        logutils.setup_filelogger(outdir=self._outdir,
                                     handlerprefix='cellmaps_image_embedding')
        logutils.write_task_start_json(outdir=self._outdir,
                                          start_time=self._start_time,
                                          data={'image_gene_node_attributes': str(self._image_gene_node_attributes),
                                                'imagedir': self._imagedir},
                                          version=cellmaps_image_embedding.__version__)
        exit_status = 99
        try:
            if self._imagedir is None:
                raise CellMapsImageEmbeddingError('imagedir must be set')

            if self._image_gene_node_attributes is None:
                raise CellMapsImageEmbeddingError('image_gene_node_attributes must be set')

            image_id_list = self._get_image_id_list()

            counter = 1
            for image_id_sublist in self._get_image_sublist(image_id_list=image_id_list):
                print(image_id_sublist)


            """ This is the fake version
            uniq_genes = set()
            with open(self._image_gene_node_attributes, 'r') as f:
                reader = csv.reader(f, delimiter='\t')
                for row in reader:
                    uniq_genes.add(row[0])

            embedding_file = os.path.join(self._outdir, 'image_emd.tsv')
            with open(embedding_file, 'w') as f:
                headerline = ['']
                for x in range(1, 1025):
                    headerline.append(str(x))
                f.write('\t'.join(headerline) + '\n')
                for gene in uniq_genes:
                    embedding = [gene]
                    for cntr in range(self._dimensions):
                        embedding.append(str(random.random()))
                    f.write('\t'.join(embedding) + '\n')
            """
            exit_status = 0
        finally:
            logutils.write_task_finish_json(outdir=self._outdir,
                                            start_time=self._start_time,
                                            status=exit_status)

        return exit_status

