import cv2
from matplotlib import pyplot as plt

# Reading the image
img = cv2.imread(r'C:\Users\Dell\Desktop\face recognition\image.png')

# Converting image to grayscale
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Loading the Haar Cascade classifier
haar_cascade = cv2.CascadeClassifier(r'C:\Users\Dell\Downloads\haarcascade_frontalface_default.xml')

# Applying face detection
faces_rect = haar_cascade.detectMultiScale(gray_img, 1.1, 9)

# Drawing rectangles around detected faces
for (x, y, w, h) in faces_rect:
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Display the image using matplotlib
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.axis('off')  # Hide axes
plt.show()
