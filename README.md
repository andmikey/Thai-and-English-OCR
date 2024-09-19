# LT2926: Assignment 1 - Thai and English OCR

Data location: /scratch/lt2326-2926-h24/ThaiOCR  

## Task
Your task is to construct a system in Python that trains a model to recognize individual Thai characters and English characters as images and to analyze the performance of your models.  However, you will do this as command line scripts and not as Jupyter notebooks (I am assuming most of you know how to use the Python argument parser). How you will design and organize these command-line scripts is up to you (what arguments they take on the command line, etc), but you must deliver the following:

(4 points) Write a script to generate training, test, and validation samples. Consider how to do this without actually copying or moving any of the files. Allow the sets to be constrained by choice of language (Thai, English, or both). 

(12 points) Write a script to train a model on a training set generated from the previous script. You can use any neural network structure you like, including extremely trivial ones. You can use a validation set however you like, or none at all. The restrictions are as follows:

- The model trained will have as input a single image in the format of the dataset and produce as output a character corresponding to the identifiers in the ThaiOCR dataset or corresponding Unicode values.
- The model must be trained in PyTorch and possible to run on mltgpu or mltgpu-2.  You may use HuggingFace tools if you like (not sure why you would, but just saying it in preparation for the next requirement).
- You may not use any pretrained models that can already recognize Thai or English characters to any degree.
- There must be at least one hidden layer and nonlinearity in the model.
- The script must save the trained model to a specified file.
- The script must allow for batching and for an arbitrary number of training epochs.
- Training must involve exactly one GPU when run on the MLT servers.

(4 points) Write a script that uses the trained model on the held-out test set and calculates precision, recall, F1 (harmonic mean of precision and recall), and accuracy.  You can also include any other statistics or useful analysis output you feel like.

(4 points) Write documentation as a Markdown readme file describing how to run the scripts in enough detail, to, well, run the scripts and take advantage of any command-line arguments you have provided.  Write up any interesting challenges or observations you found and any design decisions you made.

(4 points) In the same readme file, write up an analysis of your own experiments using your scripts containing at least the following experiments (you can choose any character resolution as long as you specify in the analysis, unless otherwise specified):

Training data | Testing data
-- | --
Thai normal text, 200dpi | Thai normal text, 200dpi
Thai normal text, 400dpi | Thai normal text, 200dpi (yes, different resolution, figure out the logistics of this)
Thai normal text, 400 dpi | Thai bold text, 400dpi
Thai bold text | Thai normal text
All Thai styles | All Thai styles
Thai and English normal text jointly | Thai and English normal text jointly.
All Thai and English styles jointly. | All Thai and English styles jointly.

(2 points) The readme should contain one more analysis using your code involving English data of your choice with selected errors for some simple qualitative error analysis (whatever subjective observations or "wild hunches" you would like to include).

The bonus (25 points)

Examine the ThaiOCR's test dataset. It's quite different. What does the different structure imply about the test task? Attempt to do this task with another test script, further documentation, and further analysis. (You can use the test data to train a possibly necessary "intermediate" task, if you like and choose that route).  You will need to train models on all the ThaiOCR "trainig" data to do the test task, however.  You may not use pretrained OCR models of any kind, but you may use Thai language models and other image processing tools/techniques that are not specifically OCR as part of your pipeline.


## Guideline PDF
Run through Google Translate:

1
Documents are 4 types:
1. .Newpaper
2. .Maxgazine
3. .Journal
4. .Book
Each type has 1 text file and 2 subdirectories: Image and Txt
1.Text file will give you the image content as follows:
nb_001sg.bmp;2;9;144;TXT
Image file name nb_001sg.bmp
Zone 2
X position start 9
Y position start 144
TXT image type
nb_001sg.bmp;2;1066;214;TXT
Image file name nb_001sg.bmp
Zone 2
X position end 1066
Y position end 214
TXT image type
2.Image can be divided into 4 groups:
- 200dpi_BW scanned at 200 dpi resolution in black and white
- 300dpi_BW Scan at 200 dpi resolution in black and white
- 200dpi_Gray Scan at 200 dpi resolution in grayscale
- 300dpi_Gray Scan at 200 dpi resolution in grayscale
3. Txt will be a text file that is the same as the image file and divided by zone, such as

H nb_001z1.txt is the image file nb_001sg.bmp in zone H 1

H nb_001z2.txt is the image file nb_001sg.bmp in zone H 2