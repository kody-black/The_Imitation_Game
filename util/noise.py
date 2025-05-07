import numpy as np
from PIL.ImageDraw import Draw
from PIL import Image
import random


def create_dots(img):
    image = Image.fromarray(img)
    draw = Draw(image)
    w, h = image.size
    width = random.randint(1, 5)
    number = 50
    for _ in range(number):
        x1 = random.randint(0, w)
        y1 = random.randint(0, h)
        draw.line(((x1, y1), (x1 + random.randint(-2, 2), y1 + random.randint(-2, 2))), fill=255, width=width)
    image = np.array(image)
    return image


def create_lines(img):
    image = Image.fromarray(img)
    draw = Draw(image)
    w, h = image.size
    width = random.randint(3, 7)
    number = 1
    for _ in range(number):
        x1 = random.randint(0, w)
        y1 = random.randint(0, h)
        x2 = random.randint(0, w)
        y2 = random.randint(0, h)
        draw.line(((x1, y1), (x2, y2)), fill=255, width=width)
    image = np.array(image)
    return image



def create_arcs(img, thickness=2, num=1):
    height, width = img.shape[:2]
    for _ in range(num):
        x = np.arange(width)
        amplitude = random.randint(0, height)
        frequency = random.randint(width // 10, width)
        y = (np.sin(x / frequency) + 1) * (amplitude / 2)

        for i in range(width):
            for j in range(-thickness // 2, thickness // 2 + 1):
                if 0 <= int(y[i]) + j < height:
                    img[int(y[i]) + j, i] = 255
    return img


def create_polynomial(img, thickness=2, num=1):
    height, width = img.shape[:2]
    for _ in range(num):
        # Define the degree of the polynomial (from 2 to 5)
        degree = random.randint(2, 5)
        # Generate random coefficients for the polynomial equation
        coeffs = [random.uniform(-1, 1) for _ in range(degree + 1)]
        # Define a vertical offset to move the curve up or down randomly
        vertical_offset = random.uniform(-0.5 * height, 0.5 * height)

        # Generate x values
        x = np.linspace(0, width, num=width)
        # Calculate y values using the polynomial equation
        y = np.zeros_like(x)
        for power, coeff in enumerate(coeffs):
            y += coeff * (x / width) ** power
        y = height / 2 + y * height / 4 + vertical_offset  # Scale and offset the curve

        # Draw the curve on the image
        for i in range(width):
            for j in range(-thickness // 2, thickness // 2 + 1):
                if 0 <= int(y[i]) + j < height:
                    img[int(y[i]) + j, i] = 255
    return img