import itertools
import cv2 as cv
import numpy as np
from concurrent import futures
from typing import TypeAlias, Final
from lib import tools

NDArray2D: TypeAlias = np.ndarray[np.ndarray[int]]
NDArray3D: TypeAlias = np.ndarray[np.ndarray[np.ndarray[int]]]

SLISE_SIZE: Final = 1000


@tools.PrintExecutionTime
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


@tools.PrintExecutionTime
def convolve_img(img: NDArray3D,
                 kernel: NDArray2D,
                 processes: int = 1) -> NDArray3D:
    blue, green, red = _get_channels(img)

    blue_convolved = _convolve_channel(blue, kernel, processes=processes)
    green_convolved = _convolve_channel(green, kernel, processes=processes)
    red_convolved = _convolve_channel(red, kernel, processes=processes)

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

    return blue_channel, green_channel, red_channel  # type: ignore


def _convolve_channel(channel: NDArray2D,
                      kernel: NDArray2D,
                      processes: int) -> NDArray2D:
    res = _multiply_matrix_by_kernel(
        matrix=channel, kernel=kernel, processes=processes,
    )
    _post_process_matrix(res)
    return res


def _multiply_matrix_by_kernel(matrix: NDArray2D,
                               kernel: NDArray2D,
                               processes: int) -> NDArray2D:
    kernel_x = len(kernel)
    kernel_y = len(kernel[0])
    matrix_x = len(matrix)
    matrix_y = len(matrix[0])

    new_matrix_x = matrix_x - kernel_x + 1
    new_matrix_y = matrix_y - kernel_y + 1

    packs_of_areas = _get_packs_of_areas(
        matrix=matrix, kernel_x=kernel_x, kernel_y=kernel_y, slice_size=SLISE_SIZE)
    packs_of_results = _multiply_packs_of_areas_by_kernel(
        packs=packs_of_areas, kernel=kernel, processes=processes,
    )
    # print(tools.is_ok_2d(packs_of_results))
    processed_matrix = _get_matrix_from_packs_of_results(
        packs=packs_of_results, x=new_matrix_x, y=new_matrix_y,
    )

    return processed_matrix


def _get_packs_of_areas(matrix: NDArray2D,
                        kernel_x: int,
                        kernel_y: int,
                        slice_size: int) -> list[list[NDArray2D]]:
    areas = _get_areas_from_matrix(matrix, kernel_x, kernel_y)
    packs = [[]]
    for area in areas:
        if len(packs[-1]) == slice_size:  # if pack is full, start new pack
            packs.append([])
        packs[-1].append(area)  # add pixel to the latest started pack

    return packs


def _get_areas_from_matrix(matrix: NDArray2D,
                           area_x: int,
                           area_y: int) -> list[NDArray2D]:
    areas = []
    for i in range(len(matrix) - area_x + 1):
        for j in range(len(matrix) - area_y + 1):
            areas.append(np.array(matrix[i:i + area_x, j:j + area_y]))

    return areas


def _multiply_packs_of_areas_by_kernel(packs: list[list[NDArray2D]],
                                       kernel: NDArray2D,
                                       processes: int = 1) -> list[list[int]]:
    with futures.ProcessPoolExecutor(max_workers=processes) as executor:
        packs_of_results = executor.map(
            _multiply_pack_of_areas_by_kernel,
            packs, itertools.repeat(kernel),
        )

    lst_packs_of_results = list(packs_of_results)
    return lst_packs_of_results


def _get_matrix_from_packs_of_results(packs: list[list[int]],
                                      x: int,
                                      y: int) -> NDArray2D:
    ndarray = np.empty(shape=(x * y), dtype='int32')

    _fill_ndarray_with_packed_results(ndarray, packs=packs)
    matrix = ndarray.reshape((x, y))

    return matrix


def _fill_ndarray_with_packed_results(ndarray: np.ndarray[int],
                                      packs: list[list[int]]) -> None:
    results_iter = itertools.chain(*packs)

    for i, _ in enumerate(ndarray):
        ndarray[i] = next(results_iter)


def _multiply_pack_of_areas_by_kernel(pack: list[NDArray2D],
                                      kernel: NDArray2D) -> list[int]:
    pack_of_results = [
        _multiply_area_by_kernel(area=area, kernel=kernel)
        for area in pack
    ]

    return pack_of_results


def _post_process_matrix(matrix: NDArray2D):
    _negatives_to_zeros(matrix)


def _negatives_to_zeros(matrix: NDArray2D):
    for row in matrix:
        for j, val in enumerate(row):
            if val < 0:
                row[j] = 0


def _multiply_area_by_kernel(area: NDArray2D,
                             kernel: NDArray2D) -> int:
    x_size = len(area)
    y_size = len(area[0])
    new = np.empty((x_size, y_size), dtype='int16')

    for i in range(x_size):
        for j in range(y_size):
            new[i, j] = area[i, j] * kernel[i, j]

    return _sum_matrix_values(new)


def _sum_matrix_values(matrix: NDArray2D) -> int:
    return sum(sum(row) for row in matrix)


def _merge_channels(brown: NDArray2D,
                    green: NDArray2D,
                    red: NDArray2D) -> NDArray3D:
    new = np.empty((len(brown), len(brown[0]), 3), dtype='int16')

    for i in range(len(new)):
        for j in range(len(new[0])):
            new[i, j, 0] = brown[i, j]
            new[i, j, 1] = green[i, j]
            new[i, j, 2] = red[i, j]

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
