# Thai and English OCR 

## Instructions for use 
### Create the conda environment 
`conda env create -f setup_files/environment.yml --prefix /scratch/gusandmich/conda_envs`
`conda activate assignment_1`
I've put mine in /scratch because it's too slow on the NFS: conda activate /scratch/gusandmich/assignment_1_scratch/ 

Set NUM_BATCHES and NUM_EPOCHS and export them into your env 

## Comments on challenges / decisions
### Saving the training data
Wrote paths to a file. Means you don't copy the data anywhere, fast to retrieve. 

### Image format
BMP, so can't load with [decode_image](https://pytorch.org/vision/main/generated/torchvision.io.decode_image.html#torchvision.io.decode_image) (not a supported format). 

Could either upscale the training data then test on actual data, or train on training data and downscale the testing data. 

Input shapes?
PIL -> Tensorflow is loaded as one-channel array. From visual inspection, true is white, false is black. Convert to array of longs - 0 is black, 1 is white.

Thai:
Images all come as different shapes, even in the same DPI, which is really annoying! 
Just reshape everything to 64x64. 

### Architecture
Went with LeNet 5. 

## Experiment results

## Bonus section
### Formatting
Last letter (b/g) indicates if b/w or grayscale. Dims are for grayscale only. Seems to work well enough for the bw too, though, from testing. 