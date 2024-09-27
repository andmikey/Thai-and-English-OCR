# Thai and English OCR 

This repository details my solution for assignment 1 of LT2926 at the University of Gothenburg (Machine learning for statistical NLP: advanced), building a system for Thai and English OCR. 

## Instructions for use 

Tl;dr if you want to run all the experiments required for this assignment, run `bash /home/gusandmich@GU.GU.SE/assignment_1/runs/do_all_runs.sh`. **This will overwrite all existing runs in that folder.**

Otherwise read below for instructions on how to use the individual scripts. 

### (Optional) Create the conda environment 

While waiting for the main server environment to be set up with the packages I needed, I set up a Conda environment with the necessary packages. You can install and use it as below:

```sh
conda env create -f setup_files/environment.yml --prefix /scratch/gusandmich/
conda activate /scratch/gusandmich/assignment_1_scratch/
``` 

Note I had to put it in `/scratch/gusandmich` rather than `/home/gusandmich` because `/home` is much too slow; loading all the packages from disk took too long when the Conda env lived there. 

### Generate training data

The [training data generation script](./assignment_code/generate_training_data.py) allows setting the languages, dpis and styles of generated data; as well as the train/test/validation split. 

For example, to generate a dataset of normal Thai and English text at 300 DPI, with 70% of data for train and 15% for validation and test respectively:

```sh
python3 assignment_code/generate_training_data.py
    --language Thai --language English # Default: all languages
    --dpi 300 # Default: all DPIs
    --style normal # Default: all styles
    --train_proportion 0.7 --validation_proportion 0.15 --test_proportion 0.15 # Default: 60/20/20
    --output_path some_training_data_folder
    --logging_path results.log
```

This will create three files in `some_training_data_folder`: `training_set.txt`, `validation_set.txt`, `testing_set.txt`; and log the size and path of each dataset to `result.log`.

### Train a model

Use the [model training script](./assignment_code/train_model.py) to train a model on a given training set, and optionally report performance on a validation set:

```sh
python3 assignment_code/train_model.py 
    --train-data some_training_data_folder/training_set.txt 
    --validation-data some_training_data_folder/validation_set.txt 
    --save_dir some_results_folder
    --logging_path results.log
    --batches 1 # Default 1
    --epochs 10 # Default 100
```

This will save:
- A training log to `results.log`, giving performance on train, test, and (optionally) validation
- The trained model to `some_results_folder/model.pth`. 


### Evaluate a trained model

Use the [model evaluation script](./assignment_code/evaluate_model.py) to evaluate a model on a given testing set:

```sh
python3 assignment_code/evaluate_model.py 
    --test-data some_training_data_foldertesting_set.txt 
    --model_path some_results_folder/model.pth 
    --logging_path results.log
```

This will log the results of model evaluation (precision, recall, F1, accuracy) to `results.log`. 

## Comments on challenges / decisions
### Generating data
I generate the train/test/val datasets by writing a file in the format:

```
language,dpi,style,class_index,image_path
Thai,200,bold,199,/scratch/lt2326-2926-h24/ThaiOCR/ThaiOCR-TrainigSet/Thai/199/200/bold/KKTS212_200_31_20_199.bmp
```

This means the dataset definition uses up very little space and does not require copying any images around. 

### Image format
The provided images are in BMP, so we can't load them with [decode_image](https://pytorch.org/vision/main/generated/torchvision.io.decode_image.html#torchvision.io.decode_image) (BMP is not one of the supported formats). Instead, I use PIL to load the image and then convert it to a tensor. 

All the images come in slightly different shapes, even in the same DPI, which makes training tricky because the model should (generally) expect to receive everything in the same input size. I dealt with this by [resizing all images](https://pytorch.org/vision/main/generated/torchvision.transforms.Resize.html), regardless of DPI, to 64x64 images. 

### Architecture
I chose to use [LeNet 5](https://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf) as the architecture for this task.

The model outputs a vector that's equivalent in length to the number of classes: `output[i]` will get the probability of class `i`, where `i` is the numerical class assignment defined in [any of the training dataset descriptions](/scratch/lt2326-2926-h24/ThaiOCR/ThaiOCR-TrainigSet/English/20110202-List-Code-Character-OCR-Training-Database.txt). The output is softmaxed in order to get a probability distribution on the outputs. The most probable class can then by chosen by returning the highest-probability index.  

## Experiment results

## Bonus section
### Understanding the task

The dataset here is whole *pages* rather than individual characters. The task is now twofold:
- Extract individual words or characters from the pages
- Run OCR to convert those words/characters into readable form

### Summary of what I did and didn't do 

I have implemented code to do the following: 
- Load the images from the test set and segment them into the labelled 'zones'.
- Extract characters from each zone using OpenCV's contour-finding functionality. 
    - I disregard contours that are too small.  
    - I tested my method manually in a Jupyter notebook and it seems to capture letters reasonably well. 
    - I added some vertical padding around each letter to correctly capture diacritics.
- Do a placeholder 'matching' between extracted characters and the provided labels (the end result is entirely incorrect, see below for more details). 
- Train a character-level model on all the provided training data.
- Use the character-level model to predict the correct label for each segment of text, compare this to the 'matched' label, and generate a final performance score. 

What I have not done: 
- Correctly match extracted characters to the labels. I decided was too much work and wasn't aligned with what I want to learn from the course, so didn't want to dedicate too much time to it. If I had more time my approach would be:
    - Segment out the image into 'rows'. My idea for this is to reduce the image to a set of intensity peaks along the y-axis (you can do this with e.g. OpenCV's reduce method). Choose the top of each peak as the midpoint of the row. 
    - Run OpenCV's contours method and assign each character to the row it belongs to. I'd do this by taking the midpoint of each character's bounding box and assigning it to the closest row along the y-axis.
    - Sort the characters within each row left-to-right (since both Thai and English are read left-to-right) according to their x-coordinates. 
    - Finally, read all the characters left-to-right in row order and use this to match against the provided labels.   
- Use word-level (rather than character-level) methods to extract words. This would be a much larger project. My thinking here is: [...]

### Formatting
Last letter (b/g) indicates if b/w or grayscale. Dims are for grayscale only. Seems to work well enough for the bw too, though, from testing. 