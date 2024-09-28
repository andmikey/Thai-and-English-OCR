# Bonus question
## Understanding the task

The dataset here is whole *pages* rather than individual characters. The task is now twofold:
- Extract individual words or characters from the pages
- Run OCR to convert those words/characters into readable form

### Summary of what I did and didn't do 

I have implemented code to do the following: 
- Load the images from the test set and segment them into the labelled 'zones'.
- Split each zone into text lines. 
    - I use OpenCV's morphologyEx to do this. It works reasonably well on pages that have *just* text, but works quite poorly on pages that have elements other than text (e.g. lines, diagrams).
- Extract characters from each zone using OpenCV's contour-finding functionality. 
    - I disregard contours that are too small.  
    - I tested my method manually in a Jupyter notebook and it seems to capture letters reasonably well. 
    - I added some vertical padding around each letter to correctly capture diacritics.
- Do a placeholder 'matching' between extracted characters and the provided labels (the end result is not very good, see below for more details). 
- Train a character-level model on all the provided training data.
- Use the character-level model to predict the correct label for each segment of text, compare this to the 'matched' label, and generate a final performance score. 

What I have not done: 
- Correctly match extracted characters to the labels. I decided getting this working was too much work and wasn't aligned with what I want to learn from the course, so didn't want to dedicate too much time to it. If I wanted to do this in practice I would just use an existing OCR tool that could handle it for me. If I wanted to implement something myself, I think there's two approaches here. One would be to fine-tune my existing method by playing with the parameters (kernel size etc) until it got reasonable performance on a wide range of pages. Another method I thought of would be:
    - Segment out the image into 'rows'. My idea for this is to reduce the image to a set of intensity peaks along the y-axis (you can do this with e.g. OpenCV's reduce method). Choose the top of each peak as the midpoint of the row. 
    - Run OpenCV's contours method and assign each character to the row it belongs to. I'd do this by taking the midpoint of each character's bounding box and assigning it to the closest row along the y-axis.
    - Sort the characters within each row left-to-right (since both Thai and English are read left-to-right) according to their x-coordinates. 
    - Finally, read all the characters left-to-right in row order and use this to match against the provided labels.   
- Use word-level (rather than character-level) methods to extract words. This would be a much larger project. My thinking here is: [...]

### Formatting
Last letter (b/g) indicates if b/w or grayscale. Dims are for grayscale only. Seems to work well enough for the bw too, though, from testing. 