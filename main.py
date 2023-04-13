from lib import kernels
from lib import image_processing
from typing import Final
import cv2 as cv

SCALE_BY: Final = 2
IMAGE_PATHS: Final = [
    'raw_images/small.jpg',
    'raw_images/medium.jpg',
    'raw_images/large.jpg',
]
SAVE_FOLDER: Final = 'processed_images'


def main():
    raw_img = cv.imread('raw_images/small.jpg')
    # processed_image = image_processing.convolve_img(
    #     img=raw_img, kernel=kernels.RELIEF, processes=4)
    #

    processed_image = image_processing.grayscale_img(img=raw_img)
    cv.imwrite('foo/bar.jpg', processed_image)


if __name__ == '__main__':
    main()
