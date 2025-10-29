#==============================================================#
# CASPAR — Computational Art                                   #
#--------------------------------------------------------------#
# Author : Stefano Brolli                                       #
# Purpose: Convert an input image into a greyscale knitted art  #
#==============================================================#

"""
Caspar takes an image and weaves it into thread-like patterns,
simulating a knitted fabric using greyscale shading.

Usage: python Caspar.py image_name resolution_output_image number_of_pins opaqueness_of_threads

The program looks for the image in the 'Images' folder.

Requirements: numpy, imageio.v3, numba.

No numba? Comment decorators @njit(cache=True, fastmath=True) on all functions in canvas_functions.py.
Also comment 'from numba import njit'.
"""

import sys
import time
import preparation_utilities as prep
import manipulate_canvas as manip

t_1 = time.time()

if len(sys.argv) < 5:
    
    print("Program usage: image_name resolution_output_image number_of_pins opaqueness_of_threads")
    print("The program looks for the image in the 'Images' folder.")
    exit(1)

input_image = sys.argv[1]   # Image we want to "knit"
size_out = int( sys.argv[2] )  # Size of the canvas
num_pins = int( sys.argv[3] )  # Number of pins
opaqueness = int( sys.argv[4] ) # Opaqueness of the threads

if num_pins < 3:

    print("The number of pins cannot be smaller than 3. Exiting...")
    exit(2)

if opaqueness < 1 or opaqueness > 255:

    print("Opaqueness must be an integer between 1 and 255.")
    exit(3)

image_start = prep.prepare("Images/" + input_image, size_out) # Prepare image (greyscale it, crop it)
canvas = manip.canvas(image_start, num_pins, opaqueness) # Create a white canvas
canvas.optimize() # Optimize pattern
canvas.print(pins = True)  # Print the Canvas. Also print the pins (pins = True).
t_2 = time.time()
print(f"Time taken to create the image = {t_2 - t_1} s.")