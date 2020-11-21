import pytesseract as pt
#import tensorflow as ts
import PyPDF2
import numpy as np
import pdf2image
import cv2
from imutils.object_detection import non_max_suppression
import tempfile

import sklearn as sklearn
from matplotlib import pyplot as plt
import os

from skimage import filters, morphology, measure, transform
from scipy.ndimage.morphology import binary_fill_holes
from sklearn import cluster
from scipy.fftpack import dct, idct

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

from PIL import ImageFilter, Image, ImageEnhance, ImageOps

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
    orig = im.copy()
    orig = np.array(orig)
    im = ImageEnhance.Contrast(im).enhance(1.2)
    imm = ImageEnhance.Contrast(im).enhance(1.2)
    imm = ImageOps.grayscale(imm)

    frequencies = dct(dct(imm, axis=0), axis=1)
    frequencies[:2, :2] = 0
    gray = idct(idct(frequencies, axis=1), axis=0)

    gray = (gray - gray.min()) / (gray.max() - gray.min())  # renormalize to range [0:1]
    plt.subplot(121), plt.imshow(im)
    plt.subplot(122), plt.imshow(gray)
    plt.show()
    # blur
    im1 = imm.filter(ImageFilter.BLUR)

    # filter
    im2 = imm.filter(ImageFilter.MinFilter(3))
    im3 = imm.filter(ImageFilter.MinFilter)
    plt.subplot(121),plt.imshow(im)
    plt.subplot(122),plt.imshow(im3)
    plt.show()
    #open cv
    openCVim = np.array(im3)
    treated = cv2.fastNlMeansDenoising(openCVim, None, 10, 10, 7)

    thresh = cv2.threshold(treated, 150, 255, cv2.THRESH_BINARY)[1]
    plt.subplot(121), plt.imshow(treated)
    plt.subplot(122), plt.imshow(thresh)
    plt.show()
    #treated = cv2.fastNlMeansDenoising(treated, None, 10, 10, 7)
    #treated = cv2.cvtColor(treated, cv2.COLOR_RGB2GRAY)
    kernel = np.ones((5, 5), np.uint8)
    treated = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)



    treated = cv2.bitwise_not(treated)
    plt.subplot(121), plt.imshow(openCVim)
    plt.subplot(122), plt.imshow(treated)
    plt.show()
    treated2 = cv2.morphologyEx(treated, cv2.MORPH_CLOSE, kernel)
    plt.subplot(121), plt.imshow(treated)
    plt.subplot(122), plt.imshow(treated2)
    plt.show()

    # find contours


    mask = np.ones((treated2.shape[0],treated2.shape[1]), np.uint8)
    masked = cv2.bitwise_or(treated2, treated2, mask=mask)
    transposed = cv2.cvtColor(masked,cv2.COLOR_GRAY2RGB)
    transposed = cv2.bitwise_not(transposed)
    plt.subplot(121), plt.imshow(masked)
    plt.subplot(122), plt.imshow(transposed)
    plt.show()
    # read text
    image_data = pt.image_to_data(treated)
    txt1 = pt.image_to_string(treated)
    #txt_original = pt.image_to_string(im)
    filename = str(file)+'.txt'
    textpath = os.path.join(path, 'testingoutput', filename)
    with open(textpath, 'w') as txtfile:
        txtfile.write(txt1)





