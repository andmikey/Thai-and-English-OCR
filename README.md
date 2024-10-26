# Thai and English OCR 

This repository details my solution for assignment 1 of LT2926 at the University of Gothenburg (Machine learning for statistical NLP: advanced), building a system for Thai and English OCR. 

My writeups for each question are available in the below files:
- [Main assignment experiment results](./main_assignment.md)
- [Short qualitative analysis of experiment results](./notebooks/qualitative_error_analysis.ipynb)
- [Bonus question](./bonus_question.md) and an [example of the OCR pipeline](./notebooks/results_of_bonus_task.ipynb).

I received a mark of 55/30 for this assignment (30/30 in the core questions and 25/25 in the bonus question) and found out that OCR is pretty hard, actually. 

## Instructions for use 

Tl;dr:
- If you want to run all the experiments required for the non-bonus part of this assignment, run `bash runs/do_all_runs.sh`. 
- If you want to run the script for the bonus part of the assignment, run `bash runs/run_bonus_task.sh`. 
- The results of all my experiment runs (logs, saved models, dataset contents, etc) are saved in `/home/gusandmich@GU.GU.SE/assignment_1_run_results/runs`. 

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
- A training log to `results.log`, giving performance on train, test, and (optionally) validation.
- A plot of the per-epoch training loss (average across the batches in the epoch) to `some_results_folder/training_log.png`.
- The trained model to `some_results_folder/model.pth`. 


### Evaluate a trained model

Use the [model evaluation script](./assignment_code/evaluate_model.py) to evaluate a model on a given testing set:

```sh
python3 assignment_code/evaluate_model.py 
    --test-data some_training_data_folder/testing_set.txt 
    --model_path some_results_folder/model.pth 
    --logging_path results.log
```

This will log the results of model evaluation (precision, recall, F1, accuracy) to `results.log`. 

### Run the bonus task

Use the scripts above to generate a training dataset and train a model on that training dataset.

Then use the bonus task run script to run an end-to-end OCR pipeline on all the images specified by the DPI and type (Book/Journal): 

```sh
python3 assignment_code/bonus_task.py
    -d '200' -t Journal # 200 DPI journal images
    --model_path some_results_folder/model.pth # Trained model
    --output_path some_results_folder/output # For each segment, writes out a segment_name_PREDICTED.txt and segment_name_ACTUAL.txt file here
    --write_images # Write out intermediate images - the original image, row thresholding, row segmentation, and character segmentation
    --img_save_path some_results_folder/images # Save intermediate images here
```
