import cv2 as cv
import itertools
from tools import execution_time, kernels
from tools import image_processors as ip
from concurrent import futures
from typing import Final

IMAGE_PATHS: Final = [
    'raw_images/small.jpg',
    'raw_images/medium.jpg',
    'raw_images/large.jpg',
]
SAVE_FOLDER: Final = 'processed_images'


def main():
    raw_images = read_images(paths=IMAGE_PATHS)
    for processes_available in (1, 2, 4, 6, 8, 10, 12, 14, 16):
        process_pictures_multiprocess(images=raw_images,
                                      kernel=kernels.RELIEF,
                                      processes=processes_available)
    # save_pictures(pictures=processed_pictures, folder=SAVE_FOLDER)


def read_images(paths: list[str]) -> list[ip.NDArray3D]:
    images = [cv.imread(path) for path in paths]
    return images


@execution_time.PrintExecutionTime
def process_pictures_multiprocess(images: list[ip.NDArray3D],
                                  kernel: ip.NDArray2D,
                                  processes: int) -> list[ip.NDArray3D]:
    print(f"Processes: {processes}")
    with futures.ProcessPoolExecutor(max_workers=processes) as executor:
        results = executor.map(
            ip.convolve_img, images, itertools.repeat(kernel)
        )
    return list(results)


def save_pictures(pictures: list[ip.NDArray3D], folder: str):
    for img, name in zip(pictures, ('small', 'medium', 'large')):
        img_path = f"{folder}/processed_{name}.jpg"
        cv.imwrite(img_path, img)


if __name__ == '__main__':
    main()
