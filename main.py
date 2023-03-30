from tools import kernels
from tools import image_processing
from typing import Final
import tools
import cv2 as cv

SCALE_BY: Final = 2
IMAGE_PATHS: Final = [
    'raw_images/small.jpg',
    'raw_images/medium.jpg',
    'raw_images/large.jpg',
]
SAVE_FOLDER: Final = 'processed_images'


def main():
    raw_img = cv.imread('raw_images/very_small.png')
    processed_image = image_processing.convolve_img(
        img=raw_img, kernel=kernels.RELIEF,
    )

    cv.imwrite('foo/bar.jpg', processed_image)

    # raw_images = image_processing.read_images(paths=IMAGE_PATHS)
    #
    # for processes_available in (1, 2, 4, 6, 8, 10, 12, 14, 16):
    #     image_processing.convolve_images_multiprocess(
    #         images=raw_images,
    #         kernel=kernels.RELIEF,
    #         processes=processes_available,
    #     )
    # print('-' * 40)
    #
    # for processes_available in (1, 2, 4, 6, 8, 10, 12, 14, 16):
    #     image_processing.scale_images_multiprocess(
    #         images=raw_images,
    #         scale_by=SCALE_BY,
    #         processes=processes_available,
    #     )
    # print('-' * 40)


if __name__ == '__main__':
    main()
