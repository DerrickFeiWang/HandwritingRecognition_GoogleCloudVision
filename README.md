# Printed and handwritten text extraction from images using Tesseract and Google Cloud Vision API

Text extraction from image files is an useful technique for document digitalization. There are several well developed OCR engines for printed text extraction, such as Tesseract and EasyOCR [1]. However, for handwritten text extraction, it's more challenging because of large variations in handwriting from person to person. Tesseract and EasyOCR can't achieve satisfying results unless the texts are hand-printed. In this post, I will describe how to use Tesseract to extract printed texts, and use Google Cloud Vision API to extract handwritten texts.
<br>
<br>
The example text image file is from the IAM handwriting dataset [2]. It has a printed text session, and handwritten session for the same text content.

![image](https://user-images.githubusercontent.com/44976640/89544581-ea719f00-d7c7-11ea-8544-42941970d1d4.png)
<br>
<br>
The following major tools are used:<br>

**OpenCV**, For finding structures in the images to automatically break the images into printed segments and handwritten segments<br>
**Google Cloud Vision API**, For extract text from handwriting segment<br>
**Tesseract and Pytesseract**, For extract text from printed segment<br>


## Step 1, Page Segmentation.
We will use OpenCV to find lines between sections, and use the coordinates of the lines to break the image into segments. OpenCV has a function called getStructuringElement(). We can define the structure type as a rectangle ("MORPH_RECT"), minimum width (200) ,and height (1) of the rectangle to find horizontal lines.

```python
def findHorizontalLines(img):
    img = cv2.imread(img) 
    
    #convert image to greyscale
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    # set threshold to remove background noise
    thresh = cv2.threshold(gray,30, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    
    # define rectangle structure (line) to look for: width 100, hight 1. This is a 
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (200,1))
    
    # Find horizontal lines
    lineLocations = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=1)
    
    return lineLocations
```
![image](https://user-images.githubusercontent.com/44976640/89545670-4557c600-d7c9-11ea-9532-a4f0de039877.png)


We can then utilize the coordinates of the horizontal lines to break the images into segments as shown below.
![image](https://user-images.githubusercontent.com/44976640/89546221-f8282400-d7c9-11ea-8063-44dc2936d92a.png)

From the above segmentation results, we can see that the segments containing the printed and handwritten texts that we are interested in are segment 2 and 3.

## Step 2. Extract Printed Text

In this step, we will use Tesseract OCR engine to extract printed text from an image segment. If you don't already have Tesseracct installed on your machine, you can download the installation file from [here](http://digi.bib.uni-mannheim.de/tesseract/tesseract-ocr-setup-4.00.00dev.exe). 

You will also need to install the pytesseract library in order to call Tesseract engine from Python.

```Python
pip install pytesseract
```
Now we can use Tesseract OCR with Python to extract text from the image segments.

```Python
# tell pytesseract where the engine is installed
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract'
def extractTextFromImg(segment):
    text = pytesseract.image_to_string(segment, lang='eng')         
    text = text.encode("gbk", 'ignore').decode("gbk", "ignore")        
    return text  
```
![image](https://user-images.githubusercontent.com/44976640/89548299-74236b80-d7cc-11ea-9376-bfd6f310ac23.png)

Tesseract OCR doesn't work well on handwritten texts. When passing the handwritten segment into Tesseract, we get very poor reading results. See below.
![image](https://user-images.githubusercontent.com/44976640/89548753-06c40a80-d7cd-11ea-9079-6fe2c5832801.png)

For handwritten text, we will use Google Cloud Vision API to get better results.

## Step 3. Extract handwritten text using Google Cloud Vision API
In order to use the Google Cloud Vision API, you will need to login to your google account, create a project or select an existing project, then enable Cloud Vision API. You will also need to create a service account key and save its json file to your local drive following the instruction on [Google Cloud](https://cloud.google.com/vision/docs/before-you-begin).  

Now we can specify the location of json file that has the service account key, and use the following Python script to feed the handwritten image to Google Cloud Vision API to extract text from it. 

```Python
import os
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "C:\GoogleCloudPlatform\my-key.json"

def CloudVisionTextExtractor(handwritings):
    # convert image from numpy to bytes for submittion to Google Cloud Vision
    _, encoded_image = cv2.imencode('.png', handwritings)
    content = encoded_image.tobytes()
    image = vision.types.Image(content=content)
    
    # feed handwriting image segment to the Google Cloud Vision API
    client = vision.ImageAnnotatorClient()
    response = client.document_text_detection(image=image)
    
    return response

def getTextFromVisionResponse(response):
    texts = []
    for page in response.full_text_annotation.pages:
        for i, block in enumerate(page.blocks):  
            for paragraph in block.paragraphs:       
                for word in paragraph.words:
                    word_text = ''.join([symbol.text for symbol in word.symbols])
                    texts.append(word_text)

    return ' '.join(texts)
```

![image](https://user-images.githubusercontent.com/44976640/89550439-3b38c600-d7cf-11ea-8cf9-fca98ee14fc5.png)

From the above results, we can see that Google Cloud Vision API has done a much better job in extracting texts from image files than Tesseract.


