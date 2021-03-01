# Assignment 2: Colloation
This script calculates collocates for a specific keyword by computing how often each word collocates with the keyword across a text corpus. A measure of the strength of association between the keyword and its collocates (MI) is calculated, and the results are saved as a single file containing the keyword, collocate, raw frequency, and the MI. 

The data used for this assignment is provided in the data-folder. 

### How to run the script: ###

Clone the repository:
```
git clone https://github.com/sofieditmer/LanguageAnalytics.git cds-language-sofie
```
From the terminal, navigate to the directory:
```
cd cds-language-sofie
```
Now you can create a virtual environment and activate it in order to be able to run the script:
```
bash create_vision_venv.sh

source lang101/bin/activate
```
Run the script, by specifying the required parameters:

-p: path to the text corpus

Example:
```
python3 Assignment2_collocation.py -p data/100_english_novels/corpus
