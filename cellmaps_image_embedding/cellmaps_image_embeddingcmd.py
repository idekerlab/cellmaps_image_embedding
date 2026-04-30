#! /usr/bin/env python

import argparse
import json
import os
import sys
import logging
import logging.config
from cellmaps_utils import logutils
from cellmaps_utils import constants
import cellmaps_image_embedding
from cellmaps_image_embedding.runner import CellmapsImageEmbedder, EmbeddingGenerator
from cellmaps_image_embedding.runner import FakeEmbeddingGenerator
from cellmaps_image_embedding.runner import DensenetEmbeddingGenerator

logger = logging.getLogger(__name__)


def _parse_arguments(desc, args):
    """
    Parses command line arguments

    :param desc: description to display on command line
    :type desc: str
    :param args: command line arguments usually :py:func:`sys.argv[1:]`
    :type args: list
    :return: arguments parsed by :py:mod:`argparse`
    :rtype: :py:class:`argparse.Namespace`
    """
    parser = argparse.ArgumentParser(description=desc,
                                     formatter_class=constants.ArgParseFormatter)
    parser.add_argument('outdir', help='Output directory')
    parser.add_argument('--inputdir', required=True,
                        help='Directory with rocrate where blue, red, '
                             'yellow, and green image directories reside')
    parser.add_argument('--model_path', type=str,
                        default='https://github.com/CellProfiling/densenet/releases/download/v0.1.0/external_crop512_focal_slov_hardlog_class_densenet121_dropout_i768_aug2_5folds_fold0_final.pth',
                        help='URL or path to model file for image embedding')
    parser.add_argument('--provenance',
                        help='Path to file containing provenance '
                             'information about input files in JSON format. '
                             'This is required if inputdir does not contain '
                             'ro-crate-metadata.json file.')
    parser.add_argument('--name',
                        help='Name of this run, needed for FAIRSCAPE. If '
                             'unset, name value from specified '
                             'by --inputdir directory or provenance file will be used')
    parser.add_argument('--organization_name',
                        help='Name of organization running this tool, needed '
                             'for FAIRSCAPE. If unset, organization name specified '
                             'in --inputdir directory or provenance file will be used')
    parser.add_argument('--project_name',
                        help='Name of project running this tool, needed for '
                             'FAIRSCAPE. If unset, project name specified '
                             'in --input directory or provenance file will be used')

    parser.add_argument('--fold', default=EmbeddingGenerator.DEFAULT_FOLD, type=int,
                        help='Image node attribute file fold to use')
    parser.add_argument('--device', default='cpu', type=str,
                        help='Device to run inference on.')
    parser.add_argument('--fake_embedder', action='store_true',
                        help='If set, generate fake embedding')
    parser.add_argument('--dimensions', default=EmbeddingGenerator.DIMENSIONS, type=int,
                        help='Dimensions of generated embedding vector')
    parser.add_argument('--suffix', default=EmbeddingGenerator.SUFFIX,
                        help='Suffix for image files')
    parser.add_argument('--logconf', default=None,
                        help='Path to python logging configuration file in '
                             'this format: https://docs.python.org/3/library/'
                             'logging.config.html#logging-config-fileformat '
                             'Setting this overrides -v parameter which uses '
                             ' default logger. (default None)')
    parser.add_argument('--skip_logging', action='store_true',
                        help='If set, output.log, error.log '
                             'files will not be created')
    parser.add_argument('--verbose', '-v', action='count', default=1,
                        help='Increases verbosity of logger to standard '
                             'error for log messages in this module. Messages are '
                             'output at these python logging levels '
                             '-v = WARNING, -vv = INFO, '
                             '-vvv = DEBUG, -vvvv = NOTSET (default ERROR '
                             'logging)')
    parser.add_argument('--version', action='version',
                        version=('%(prog)s ' +
                                 cellmaps_image_embedding.__version__))

    return parser.parse_args(args)


def main(args):
    """
    Main entry point for program

    :param args: arguments passed to command line usually :py:func:`sys.argv[1:]`
    :type args: list

    :return: return value of :py:meth:`cellmaps_image_embedding.runner.CellmapsImageEmbedder.run`
             or ``2`` if an exception is raised
    :rtype: int
    """
    desc = """
Version {version}

Generates image embeddings from immunofluorescent labeled images
from the Human Protein Atlas that were downloaded by the
cellmaps_imagedownloader package using Densenet code taken from:
https://github.com/CellProfiling/densenet

To use set --inputdir to output directory created by cellmaps_imagedownloader
with red, blue, green, yellow directories containing images and FAIRSCAPE ro-crate
configuration file (ro-crate-metadata.json)

The generated embeddings are stored in image_emd.tsv under the output directory
specified when running this tool.

    """.format(version=cellmaps_image_embedding.__version__)
    theargs = _parse_arguments(desc, args[1:])
    theargs.program = args[0]
    theargs.version = cellmaps_image_embedding.__version__

    if theargs.provenance is not None:
        with open(theargs.provenance, 'r') as f:
            json_prov = json.load(f)
    else:
        json_prov = None

    try:
        logutils.setup_cmd_logging(theargs)
        if theargs.fake_embedder is True:
            gen = FakeEmbeddingGenerator(theargs.inputdir,
                                         fold=theargs.fold,
                                         dimensions=theargs.dimensions)
        else:
            gen = DensenetEmbeddingGenerator(os.path.abspath(theargs.inputdir),
                                             dimensions=theargs.dimensions,
                                             outdir=os.path.abspath(theargs.outdir),
                                             model_path=theargs.model_path,
                                             suffix=theargs.suffix,
                                             fold=theargs.fold,
                                             device=theargs.device)
        return CellmapsImageEmbedder(outdir=theargs.outdir,
                                     inputdir=theargs.inputdir,
                                     embedding_generator=gen,
                                     skip_logging=theargs.skip_logging,
                                     name=theargs.name,
                                     project_name=theargs.project_name,
                                     organization_name=theargs.organization_name,
                                     provenance=json_prov,
                                     input_data_dict=theargs.__dict__).run()
    except Exception as e:
        logger.exception('Caught exception: ' + str(e))
        return 2
    finally:
        logging.shutdown()


if __name__ == '__main__':  # pragma: no cover
    sys.exit(main(sys.argv))
