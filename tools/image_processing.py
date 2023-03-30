import itertools
import time

import cv2 as cv
import numpy as np
from concurrent import futures
from typing import TypeAlias
from tools import execution_time

NDArray2D: TypeAlias = np.ndarray[np.ndarray[int]]
NDArray3D: TypeAlias = np.ndarray[np.ndarray[np.ndarray[int]]]


@execution_time.PrintExecutionTime
def scale_images_multiprocess(images: list[NDArray3D],
                              scale_by: int,
                              processes: int) -> list[NDArray3D]:
    print(f"Processes: {processes}")
    with futures.ProcessPoolExecutor(max_workers=processes) as executor:
        results = executor.map(
            scale_img, images, itertools.repeat(scale_by)
        )
    return list(results)


def save_pictures(pictures: list[NDArray3D], folder: str):
    for img, name in zip(pictures, ('small', 'medium', 'large')):
        img_path = f"{folder}/processed_{name}.jpg"
        cv.imwrite(img_path, img)


@execution_time.PrintExecutionTime
def convolve_img(img: NDArray3D,
                 kernel: NDArray2D) -> NDArray3D:
    blue, green, red = _get_channels(img)

    blue_convolved = _convolve_channel(blue, kernel)
    green_convolved = _convolve_channel(green, kernel)
    red_convolved = _convolve_channel(red, kernel)

    convolved_img = _merge_channels(
        blue_convolved, green_convolved, red_convolved
    )

    return convolved_img


def scale_img(img: NDArray3D, scale_by: int = 2):
    b, g, r = _get_channels(img)
    b_scaled = _scale_channel(b, scale_by)
    g_scaled = _scale_channel(g, scale_by)
    r_scaled = _scale_channel(r, scale_by)
    img_scaled = _merge_channels(b_scaled, g_scaled, r_scaled)

    return img_scaled


def read_images(paths: list[str]) -> list[NDArray3D]:
    images = [cv.imread(path) for path in paths]
    return images


def _get_channels(img: NDArray3D) -> tuple[NDArray2D, NDArray2D, NDArray2D]:
    blue_channel = img[:, :, 0]
    green_channel = img[:, :, 1]
    red_channel = img[:, :, 2]

    return blue_channel, green_channel, red_channel


def _convolve_channel(channel: NDArray2D,
                      kernel: NDArray2D) -> NDArray2D:
    res = _multiply_by_kernel_multiprocess(channel, kernel)
    # res = _multiply_by_kernel(channel, kernel)
    _process_matrix(res)
    return res


def _multiply_by_kernel_multiprocess(matrix: NDArray2D, kernel: NDArray2D) -> NDArray2D:
    x_size = len(kernel)
    y_size = len(kernel[0])
    x_len = len(matrix)
    y_len = len(matrix[0])
    res_matrix_x_len = x_len - x_size + 1
    res_matrix_y_len = y_len - y_size + 1

    new = np.empty((res_matrix_x_len, res_matrix_y_len), dtype='int16')
    packs_of_areas = _get_packs_of_areas(matrix, kernel)

    with futures.ProcessPoolExecutor(max_workers=1) as executor:
        packs_of_results = executor.map(
            _multiply_pack_of_areas,
            packs_of_areas, itertools.repeat(kernel),
        )

    for i, pack_of_results in enumerate(packs_of_results):
        for j, result in enumerate(pack_of_results):
            new[i, j] = result

    return new


def _get_packs_of_areas(matrix: NDArray2D,
                        kernel: NDArray2D) -> list[list[NDArray2D]]:
    x_size = len(kernel)
    y_size = len(kernel[0])
    x_len = len(matrix)
    y_len = len(matrix[0])
    res_matrix_x_len = x_len - x_size + 1
    res_matrix_y_len = y_len - y_size + 1

    packs_of_areas = []
    for i in range(res_matrix_x_len):
        pack_of_areas = []
        for j in range(res_matrix_y_len):
            pack_of_areas.append(np.array(matrix[i:i + x_size, j:j + y_size]))
        packs_of_areas.append(pack_of_areas)

    return packs_of_areas


def _multiply_pack_of_areas(pack_of_areas: list[NDArray2D],
                            kernel: NDArray2D) -> list[NDArray2D]:
    pack_of_results = []
    for area in pack_of_areas:
        pack_of_results.append(_multiply_by_kernel(area, kernel))

    return pack_of_results


def _multiply_by_kernel(matrix: NDArray2D, kernel: NDArray2D) -> NDArray2D:
    x_size = len(kernel)
    y_size = len(kernel[0])
    x_len = len(matrix)
    y_len = len(matrix[0])
    res_matrix_x_len = x_len - x_size + 1
    res_matrix_y_len = y_len - y_size + 1
    new = np.empty((res_matrix_x_len, res_matrix_y_len), dtype='int16')

    for i in range(res_matrix_x_len):
        for j in range(res_matrix_y_len):
            # areas_to_multiply.append(matrix[i:i + x_size, j:j + y_size])
            res = _multiply_area(matrix[i:i + x_size, j:j + y_size], kernel)
            new[i, j] = res

    return new


def _process_matrix(matrix: NDArray2D):
    _negatives_to_zeros(matrix)


def _negatives_to_zeros(matrix: NDArray2D):
    for row in matrix:
        for j, val in enumerate(row):
            if val < 0:
                row[j] = 0


def _multiply_area(matrix_1: NDArray2D,
                   matrix_2: NDArray2D) -> NDArray2D:
    x_size = len(matrix_1)
    y_size = len(matrix_2)
    new = np.empty((x_size, y_size), dtype='int16')

    for i in range(x_size):
        for j in range(y_size):
            new[i, j] = matrix_1[i, j] * matrix_2[i, j]

    return _sum_matrix_values(new)


def _sum_matrix_values(matrix: NDArray2D) -> float:
    return sum(sum(row) for row in matrix)


def _merge_channels(b: NDArray2D,
                    g: NDArray2D,
                    r: NDArray2D) -> NDArray3D:
    new = np.empty((len(b), len(b[0]), 3), dtype='int16')

    for i in range(len(new)):
        for j in range(len(new[0])):
            values = (b[i, j], g[i, j], r[i, j])
            new[i, j] = np.array(values, dtype='int16')

    return new


def _scale_channel(channel: NDArray2D, scale_by: int):
    channel_x = len(channel)
    channel_y = len(channel[0])
    x = channel_x // scale_by
    y = channel_y // scale_by
    new = np.empty((x, y), dtype='int16')
    for i in range(x):
        for j in range(y):
            new[i, j] = _get_avg(
                channel[i * scale_by: (i + 1) * scale_by, j * scale_by: (j + 1) * scale_by]
            )

    return new


def _get_avg(matrix: NDArray2D) -> int:
    res = round(_sum_matrix_values(matrix) / len(matrix) ** 2)
    return res
