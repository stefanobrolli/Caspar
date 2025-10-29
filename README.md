Caspar takes an image and weaves it into thread-like patterns,
simulating a knitted fabric using greyscale shading.

Usage: python Caspar.py image_name resolution_output_image number_of_pins opaqueness_of_threads

The program looks for the image in the 'Images' folder.

Requirements: numpy, imageio.v3, numba.

No numba? Comment decorators @njit(cache=True, fastmath=True) on all functions in canvas_functions.py.
Also comment 'from numba import njit'.
