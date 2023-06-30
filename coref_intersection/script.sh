# pip install neuralcoref --no-binary neuralcoref
# python -m spacy download en_core_web_sm
python -m spacy validate
python init.py
# rm -rf app/models/coref-spanbert-large-2021.03.10.tar.gz
rm -rf app/models/en_core_web_sm-2.1.0.tar.gz
