# Assignment 2: Colloation
This script calculates collocates for a specific keyword by computing how often each word collocates with the keyword across a text corpus. A measure of the strength of association between the keyword and its collocates (MI) is calculated, and the results are saved as a single file containing the keyword, collocate, raw frequency, and the MI. 

The data used for this assignment is provided in the data-folder. 

### How to run the script: ###

1. Clone the repository <br>
```
git clone https://github.com/sofieditmer/cds-language.git cds-language-sofie
```

2. From the terminal, navigate to the directory <br>
```
cd cds-language-sofie
```

3. Now you can create a virtual environment called "lang101" and activate it in order to be able to run the script <br>
```
bash create_lang_venv.sh

source lang101/bin/activate
```

4. Navigate to the directory containing the script <br>
```
cd assignments/assignment2_StringProcessing
```

5. Run the script and specify the required parameters:

-p: path to the text corpus

Example:
```
python3 Assignment2_collocation_au617836.py -p data/100_english_novels/corpus
```
