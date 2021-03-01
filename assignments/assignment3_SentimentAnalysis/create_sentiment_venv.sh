#!/usr/bin/env bash

VENVNAME=sentiment_environment

python3 -m venv $VENVNAME
source $VENVNAME/bin/activate
pip install --upgrade pip

pip install ipython
pip install jupyter
python -m ipykernel install --user --name=$VENVNAME

test -f requirements_sentiment.txt && pip install -r requirements_sentiment.txt
python -m spacy download en_core_web_sm

deactivate
echo "build $VENVNAME"
