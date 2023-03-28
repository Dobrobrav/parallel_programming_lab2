from tools import kernels
from tools import image_processing as ip
from typing import Final

SCALE_BY: Final = 2
IMAGE_PATHS: Final = [
    'raw_images/small.jpg',
    'raw_images/medium.jpg',
    'raw_images/large.jpg',
]
SAVE_FOLDER: Final = 'processed_images'


def main():
    raw_images = ip.read_images(paths=IMAGE_PATHS)

    for processes_available in (1, 2, 4, 6, 8, 10, 12, 14, 16):
        ip.convolve_images_multiprocess(images=raw_images,
                                        kernel=kernels.RELIEF,
                                        processes=processes_available)
    print('-' * 40)

    for processes_available in (1, 2, 4, 6, 8, 10, 12, 14, 16):
        ip.scale_images_multiprocess(images=raw_images,
                                     scale_by=SCALE_BY,
                                     processes=processes_available)
    print('-' * 40)


if __name__ == '__main__':
    main()
