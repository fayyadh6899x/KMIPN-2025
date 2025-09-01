from PIL import Image
img = Image.open("20250901_091836.jpg")
print(img._getexif())
