from allennlp.predictors.predictor import Predictor
import neuralcoref
path = "app/models/coref-spanbert-large-2021.03.10.tar.gz"
predictor = Predictor.from_path(path)
print(neuralcoref.__version__)