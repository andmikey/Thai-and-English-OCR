# Thai and English OCR 

## Instructions for use 
### Create the conda environment 
`conda env create -f setup_files/environment.yml`
`conda activate assignment_1`

## Comments on challenges / decisions
### Image format
Use `identify` to check formats. 

`identify -verbose /scratch/lt2326-2926-h24/ThaiOCR/ThaiOCR-TrainigSet/English/046/200/bold/ASEB211_200_30_08_046.bmp`:
```
  Filename: ASEB211_200_30_08_046.bmp
  Mime type: image/bmp
  Class: PseudoClass
  Geometry: 4x2+0+0
  Units: PixelsPerCentimeter
  Colorspace: sRGB
  Type: Bilevel
  Depth: 1-bit
  Channels: 4.0
  Channel depth:
    Red: 1-bit
    Green: 1-bit
    Blue: 1-bit
```

`identify -verbose /scratch/lt2326-2926-h24/ThaiOCR/ThaiOCR-TrainigSet/English/046/400/bold/ASEB411_400_30_08_046.bmp`:
```
  Format: BMP3 (Microsoft Windows bitmap image (V3))
  Mime type: image/bmp
  Class: PseudoClass
  Geometry: 6x5+0+0
  Units: PixelsPerCentimeter
  Colorspace: sRGB
  Type: Bilevel
  Depth: 1-bit
  Channels: 4.0
  Channel depth:
    Red: 1-bit
    Green: 1-bit
    Blue: 1-bit
```

BMP so will need to upscale/downscale in PyTorch. 

Could either upscale the training data then test on actual data, or train on training data and downscale the testing data. 

## Experiment results

## Bonus section