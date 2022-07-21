# image_captioning

About Caption:
This project use data from: M. Hodosh, P. Young and J. Hockenmaier (2013) "Framing Image Description as a Ranking Task: Data, Models and Evaluation Metrics", Journal of Artifical Intellegence Research, Volume 47, pages 853-899
http://www.jair.org/papers/paper3994.html

inside that:
1. Flickr8k.token.txt - the raw captions of the Flickr8k Dataset . The first column is the ID of the caption which is "image address # caption number"
2. Flickr_8k.trainImages.txt - The training images used in our experiments
3. Flickr_8k.testImages.txt - The test images used in our experiments

and Images folder contains over 8000 pictures.

there are 3 Python file that handle:
1. caption_encode preprocess raw data into cleaned and convert it into vector for input model; caculate matrix to encode word by word into vector using glove model
2. image_encode encode image to vector using inception model (a CNN model instance)
3. model.py building model, train and test new picture
