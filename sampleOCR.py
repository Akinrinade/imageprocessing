import pytesseract as pt
#import tensorflow as ts
import PyPDF2
import numpy as np
import pdf2image
import cv2
import tempfile
from matplotlib import pyplot as plt
import os
from skimage import io, color
from scipy.fftpack import dct, idct
from skimage import filters, morphology, measure
from scipy.ndimage.morphology import binary_fill_holes
from skimage import transform

""""
sample = pdf2image.convert_from_path('C:\\Users\\bisib\\PycharmProjects\\SampleOCR\\sample_data\\Scanned_20200928-1633.pdf')
for page in sample:
    page.save('out.jpg', 'JPEG')
pt.pytesseract.tesseract_cmd ="C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
img = cv2.imread('sample_data/20200928_115921.jpg')
#imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
treated=cv2.fastNlMeansDenoisingColored(img,None,10,10,7,21)
treated= cv2.fastNlMeansDenoisingColored(treated,None,10,10,7,21)
plt.subplot(121),plt.imshow(img)
plt.subplot(122),plt.imshow(treated)
plt.show()


print(pt.image_to_string(imgray))
"""
"""
fhandle = open(r'sample_data/Scanned_20200928-1633.pdf', 'rb')
pdfReader = PyPDF2.PdfFileReader(fhandle)
pagehandle = pdfReader.getPage(0)
print(pagehandle.extractText())

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

"""
#cv2.imwrite('C:/Temp/person-masked.jpg', masked)           # Save

from PIL import ImageFilter, Image
pt.pytesseract.tesseract_cmd ="C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
files = os.listdir('testingdata')
path = os.getcwd()
for file in files:


    filepath = os.path.join(path, 'testingdata', file)
    if str(file).endswith('.pdf'):
        sample = pdf2image.convert_from_path(filepath)
        for page in sample:
            page.save('out.jpg', 'JPEG')
            filepath = os.path.join(path, 'out.jpg')

    # first of many steps of image preprocessing
    im = Image.open(filepath)
    # blur
    im1 = im.filter(ImageFilter.BLUR)
    # filter
    im2 = im.filter(ImageFilter.MinFilter(3))
    im3 = im.filter(ImageFilter.MinFilter)
    plt.subplot(121),plt.imshow(im)
    plt.subplot(122),plt.imshow(im3)
    plt.show()
    # read text
    image_data = pt.image_to_data(im3)
    txt1 = pt.image_to_string(im3)
    #txt_original = pt.image_to_string(im)
    filename = str(file)+'.txt'
    textpath = os.path.join(path, 'testingoutput', filename)
    with open(textpath, 'w') as txtfile:
        txtfile.write(txt1)





