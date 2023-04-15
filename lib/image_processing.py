import itertools
import time

import cv2 as cv
import numpy as np
from concurrent import futures
from typing import TypeAlias, Final, Iterator, Iterable, Callable
from lib import tools

NDArray2D: TypeAlias = np.ndarray[np.ndarray[int]]
NDArray3D: TypeAlias = np.ndarray[np.ndarray[np.ndarray[int]]]

CONVOLVE_SLICE_SIZE: Final = 10_000
EROSION_SLICE_SIZE: Final = 10_000
GS_SLICE_SIZE: Final = 100_000

GS_PROCESSES: Final = 6
ERODE_PROCESSES: Final = 6


def threshold_img(img: NDArray2D,
                  threshold: float) -> NDArray2D:
    filter_func = lambda x: 0 if x < threshold else 1
    processed_img = np.apply_along_axis(func1d=filter_func, axis=1, arr=img)

    return processed_img


# @tools.PrintExecutionTime
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


# @tools.PrintExecutionTime
def convolve_img(img: NDArray3D,
                 kernel: NDArray2D,
                 processes: int | None = None) -> NDArray3D:
    channels = _get_channels(img)

    convolved_channels = _convolve_channels(channels, kernel, processes=processes)

    convolved_img = _merge_channels(convolved_channels)

    return convolved_img


def erode_img(img: NDArray2D,
              kernel: NDArray2D,
              step: int) -> NDArray2D:
    for _ in range(step):
        img = _erode_img_step_1(img, kernel)

    return img


def _erode_img_step_1(img: NDArray2D,
                      kernel: NDArray2D) -> NDArray2D:
    return _process_2d_img(
        processor=_erode_pack_of_areas, img=img,
        kernel=kernel, processes=ERODE_PROCESSES
    )


def _process_2d_img(processor: Callable,
                    img: NDArray2D,
                    kernel: NDArray2D,
                    processes: int | None) -> NDArray2D:
    kernel_x = len(kernel)
    kernel_y = len(kernel[0])
    matrix_x = len(img)
    matrix_y = len(img[0])

    new_matrix_x = matrix_x - kernel_x + 1
    new_matrix_y = matrix_y - kernel_y + 1

    packs_of_areas = _get_packs_of_areas(
        matrix=img, area_x=kernel_x, area_y=kernel_y,
        slice_size=EROSION_SLICE_SIZE)

    packs_of_results = _process_packs_of_areas(
        processor=processor, packs=packs_of_areas,
        kernel=kernel, processes=processes)

    processed_img = _get_matrix_from_packs(
        packs=packs_of_results, x=new_matrix_x, y=new_matrix_y)

    return processed_img


def _process_packs_of_areas(processor: Callable,
                            packs: list[list[NDArray2D]],
                            kernel: NDArray2D,
                            processes: int | None) -> Iterator[list[int]]:
    with futures.ProcessPoolExecutor(max_workers=processes) as executor:
        packs_of_results = executor.map(
            processor,
            packs, itertools.repeat(kernel)
        )

    return packs_of_results


def _erode_pack_of_areas(pack: list[NDArray2D],
                         kernel: NDArray2D) -> list[int]:
    results = [_erode_area(area, kernel) for area in pack]
    return results


def _erode_area(area: NDArray2D, kernel: NDArray2D) -> int:
    return 0 if area < kernel else 1


@tools.PrintExecutionTime
def _convolve_channels(channels: tuple[NDArray2D, NDArray2D, NDArray2D],
                       kernel: NDArray2D,
                       processes: int | None) -> Iterator[NDArray2D]:
    with futures.ProcessPoolExecutor(max_workers=3) as executor:
        results = executor.map(
            _convolve_channel,
            channels, itertools.repeat(kernel), itertools.repeat(processes),
        )

    return results


def scale_img(img: NDArray3D,
              scale_by: int = 2) -> NDArray3D:
    b, g, r = _get_channels(img)
    b_scaled = _scale_channel(b, scale_by)
    g_scaled = _scale_channel(g, scale_by)
    r_scaled = _scale_channel(r, scale_by)
    img_scaled = _merge_channels(b_scaled, g_scaled, r_scaled)  # TODO: fix this!

    return img_scaled


def read_images(paths: list[str]) -> list[NDArray3D]:
    images = [cv.imread(path) for path in paths]
    return images


@tools.PrintExecutionTime
def grayscale_img(img: NDArray3D) -> NDArray2D:
    raw_packs = _get_packs_of_pixels(img, slice_size=GS_SLICE_SIZE)
    gs_packs = _grayscale_packs(raw_packs)

    x, y, _ = img.shape
    new_matrix = _get_matrix_from_packs(
        packs=gs_packs, x=x, y=y)

    return new_matrix


@tools.PrintExecutionTime
def _grayscale_packs(packs: list[list[np.ndarray]]) -> Iterator[list[int]]:
    with futures.ProcessPoolExecutor(max_workers=GS_PROCESSES) as executor:
        gs_packs = executor.map(
            _grayscale_pack,
            packs,
        )

    return gs_packs  # Плохо параллелится предположительно из-за того, что каждый пак надо скопировать


def _grayscale_pack(pack: list[np.ndarray]) -> list[int]:
    gs_pack = [_grayscale_pixel(pixel) for pixel in pack]

    # print(time.perf_counter() - start)
    return gs_pack


def _grayscale_pixel(pixel: np.ndarray) -> int:
    return sum(pixel) / len(pixel)


# @tools.PrintExecutionTime
def _get_channels(img: NDArray3D) -> tuple[NDArray2D, NDArray2D, NDArray2D]:
    blue_channel = img[:, :, 0]
    green_channel = img[:, :, 1]
    red_channel = img[:, :, 2]

    return blue_channel, green_channel, red_channel  # type: ignore


def _convolve_channel(channel: NDArray2D,
                      kernel: NDArray2D,
                      processes: int | None) -> NDArray2D:
    res = _process_2d_img(
        processor=_multiply_pack_of_areas_by_kernel, img=channel,
        kernel=kernel, processes=processes
    )

    return res


# @tools.PrintExecutionTime
def _get_packs_of_areas(matrix: NDArray2D,
                        area_x: int,
                        area_y: int,
                        slice_size: int) -> list[list[NDArray2D]]:
    areas: list[NDArray2D] = _get_areas(matrix, area_x, area_y)
    packs_of_areas: list[list[NDArray2D]] = _get_packs(areas, slice_size)

    return packs_of_areas


@tools.PrintExecutionTime
def _get_packs_of_pixels(img: NDArray3D,
                         slice_size: int) -> list[list[np.ndarray]]:
    pixels: NDArray2D = _reduce_dimension(img)
    packs_of_pixels: list[list[NDArray2D]] = _get_packs(pixels, slice_size)

    return packs_of_pixels


def _reduce_dimension(ndarray_3d: np.ndarray) -> np.ndarray:
    x, y, z = ndarray_3d.shape
    return ndarray_3d.reshape((x * y, z))


def _get_packs(units: Iterable, slice_size: int) -> list[list]:
    packs = [[]]

    for unit in units:
        if len(packs[-1]) == slice_size:  # if pack is full, start new pack
            packs.append([])
        packs[-1].append(unit)  # add pixel to the latest started pack

    return packs


def _get_areas(matrix: NDArray2D,
               area_x: int,
               area_y: int) -> list[NDArray2D]:
    # print(tools.total_elements(matrix))

    areas = []
    for i in range(len(matrix) - area_x + 1):
        for j in range(len(matrix[0]) - area_y + 1):
            areas.append(np.array(matrix[i:i + area_x, j:j + area_y]))

    return areas


# @tools.PrintExecutionTime
def _multiply_packs_of_areas_by_kernel(packs: list[list[NDArray2D]],
                                       kernel: NDArray2D,
                                       processes: int | None) -> Iterator[list[int]]:
    # print(tools.total_elements(packs))

    with futures.ProcessPoolExecutor(max_workers=processes) as executor:
        packs_of_results = executor.map(
            _multiply_pack_of_areas_by_kernel,
            packs, itertools.repeat(kernel),
        )

    return packs_of_results


@tools.PrintExecutionTime
def _get_matrix_from_packs(packs: Iterator[list[int]],
                           x: int,
                           y: int) -> NDArray2D:
    ndarray = np.empty(shape=(x * y), dtype='int32')

    _fill_ndarray_with_packed_results(ndarray, packs=packs)
    matrix = ndarray.reshape((x, y))

    return matrix


def _fill_ndarray_with_packed_results(ndarray: np.ndarray[int],
                                      packs: Iterator[list[int]]) -> None:
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


def _negatives_to_zeros(matrix: NDArray2D):
    for row in matrix:
        for j, val in enumerate(row):
            if val < 0:
                row[j] = 0


def _multiply_area_by_kernel(area: NDArray2D,
                             kernel: NDArray2D) -> int:
    x_size = len(area)
    y_size = len(area[0])
    new = np.empty((x_size, y_size), dtype='int32')

    for i in range(x_size):
        for j in range(y_size):
            new[i, j] = area[i, j] * kernel[i, j]

    sum_value = _sum_matrix_values(new)

    if sum_value < 0:
        sum_value = -sum_value

    return sum_value


def _sum_matrix_values(matrix: NDArray2D) -> int:
    return sum(sum(row) for row in matrix)


# @tools.PrintExecutionTime
def _merge_channels(
        channels: tuple[NDArray2D, NDArray2D, NDArray2D]
) -> NDArray3D:
    brown, green, red = channels

    new = np.empty((len(brown), len(brown[0]), 3), dtype='int32')

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
