# CS494_Text_Classigication_Project
The project to build a simple text classifier and data set to distinguish Tumblr and Reddit posts. For CS494 class project


Tutorial followed for 'tutorial3.py': https://realpython.com/python-keras-text-classification/

Run tutorial with `python3 tutorial3.py`


Be aware: reddit/twitter classifier download datasets from HuggingFace. Datasets are very large, and will require space, and time to download. Once initially run, they will not need to be downloaded again.


How to run:

Install Pyenv and install Python 3.11.15
Using Pyenv, create a venv with Python 3.11.15
install requirements

Run Reddit/Twitter classifier with `python3 reddit_twitter_classifier.py`

I may be forgetting a step but I've been working on this for... so many hours now

Requires: 
- python                   3.11.15
- datasets                 2.19.2
- huggingface_hub          1.13.0
- keras                    3.14.0
- numpy                    1.26.4
- pandas                   2.1.4
- scikeras                 0.13.0
- scikit-learn             1.8.0
- scipy                    1.17.1
- tensorflow               2.21.0

I'm sure there's other things that it needs, but this is all I can remember. There are version control issues, so versions are important. May have time to write a make file for this. I am working on linux and dont know how Mac reacts to anything, so I can just verify on linux.