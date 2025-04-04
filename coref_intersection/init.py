import logging

import neuralcoref
from allennlp.predictors.predictor import Predictor

path = "app/models/coref-spanbert-large-2021.03.10.tar.gz"
predictor = Predictor.from_path(path)
logging.info(neuralcoref.__version__)
