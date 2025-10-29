import numpy as np
import imageio.v3 as iio   
from numba import njit
import preparation_utilities as prep



def _print(canvas, x_pins, y_pins, name, pins):
    
    """
    Prints the canvas. The x and y positions of all the pins ('x_pins' and 'y_pins') are needed 
    to determine the position of the pins on the canvas. Needed in case pins = True, which colors 
    the pins of the image black.
    """
    
    if pins == True:
            
        for n in range(x_pins.shape[0]):

            x_pin = x_pins[n]
            y_pin = y_pins[n]

            canvas[canvas.shape[0] - y_pin - 1, x_pin] = 0

    iio.imwrite(name, canvas)


@njit(cache=True, fastmath=True)
def _add_thread(canvas, x_pins, y_pins, start_pin, end_pin, opaqueness):
    
    """
    Adds a thread from start_pin to end_pin with the set opaqueness. The x and y positions 
    of all the pins ('x_pins' and 'y_pins') are needed to determine the position of the 
    start and end pins ('start_pin', 'end_pin') on the canvas.
    """
    
    x_1 = x_pins[start_pin]
    y_1 = y_pins[start_pin]

    x_2 = x_pins[end_pin]
    y_2 = y_pins[end_pin]

    if x_1 == x_2: # Vertical thread

        y_start = min(y_1, y_2)
        y_end = max(y_1, y_2)

        for y in range(y_start, y_end):

            if canvas[canvas.shape[0] - y - 1, x_1] >= opaqueness:
                canvas[canvas.shape[0] - y - 1, x_1] -= opaqueness # Reduce brightness by the opaqueness
            else: # It is already black
                canvas[canvas.shape[0] - y - 1, x_1] = 0

    else:

        x_start = min(x_1, x_2)
        x_end = max(x_1, x_2)

        y_start = min(y_1, y_2)
        y_end = max(y_1, y_2)

        m = (y_2 - y_1) / (x_2 - x_1)

        if abs(m) < 1: # x axis longer: there are more pixels on the x axis. Iterate on x

            for x in range(x_start, x_end):

                y = round(m*(x - x_1) + y_1)
                
                if canvas[canvas.shape[0] - y - 1, x] >= opaqueness:
                    canvas[canvas.shape[0] - y - 1, x] -= opaqueness # Reduce brightness by the opaqueness
                else: # It is already black
                    canvas[canvas.shape[0] - y - 1, x] = 0

        else: # y axis longer: there are more pixels on the y axis. Iterate on y
                
            for y in range(y_start, y_end):

                x = round((y - y_1) / m + x_1)

                if canvas[canvas.shape[0] - y - 1, x] >= opaqueness: 
                    canvas[canvas.shape[0] - y - 1, x] -= opaqueness # Reduce brightness by the opaqueness
                else: # It is already black
                    canvas[canvas.shape[0] - y - 1, x] = 0


@njit(cache=True, fastmath=True)
def _restore_thread(canvas, saved_canvas, x_pins, y_pins, start_pin, end_pin):
    
    """
    Restore a thread on 'canvas' using the previous version of the canvas ('saved_canvas'). The x and y positions 
    of all the pins ('x_pins' and 'y_pins') are needed to determine the position of the 
    start and end pins ('start_pin', 'end_pin') on the canvas.
    """
    
    x_1 = x_pins[start_pin]
    y_1 = y_pins[start_pin]

    x_2 = x_pins[end_pin]
    y_2 = y_pins[end_pin]

    if x_1 == x_2:

        y_start = min(y_1, y_2)
        y_end = max(y_1, y_2)

        for y in range(y_start, y_end):

            canvas[canvas.shape[0] - y - 1, x_1] = saved_canvas[saved_canvas.shape[0] - y - 1, x_1]

    else:

        x_start = min(x_1, x_2)
        x_end = max(x_1, x_2)

        y_start = min(y_1, y_2)
        y_end = max(y_1, y_2)

        m = (y_2 - y_1) / (x_2 - x_1)

        if abs(m) < 1: 

            for x in range(x_start, x_end):

                y = round(m*(x - x_1) + y_1)
                canvas[canvas.shape[0] - y - 1, x] = saved_canvas[saved_canvas.shape[0] - y - 1, x]

        else:
                
            for y in range(y_start, y_end):

                x = round((y - y_1) / m + x_1)
                canvas[canvas.shape[0] - y - 1, x] = saved_canvas[saved_canvas.shape[0] - y - 1, x]


@njit(cache=True, fastmath=True)
def _compute_loss(canvas, image):

    """
    Compute the loss as the L^2 norm between the canvas 'canvas' and the
    image 'image' from scratch. Normalize by the number of pixels.
    Returns the loss.
    """

    N = image.shape[0] * image.shape[1]
    diff = canvas.astype(np.float32) - image.astype(np.float32)
    loss = 1/N*np.sum( diff * diff )
    
    return loss


@njit(cache=True, fastmath=True)
def _compute_loss_optimal(canvas, saved_canvas, x_pins, y_pins, loss, start_pin, end_pin, image):

    """
    Compute the loss of 'canvas' by only considering the pixels that change according to
    Loss_new = Loss_old - Loss_old thread + Loss_new_thread.

    'saved_canvas' should be equal to canvas.copy(), needed to restore the pixels.
    The x and y positions of all the pins ('x_pins' and 'y_pins') are needed to determine 
    the position of the start and end pins ('start_pin', 'end_pin') on the canvas.
    Returns the new loss.
    """

    x_1 = x_pins[start_pin]
    y_1 = y_pins[start_pin]

    x_2 = x_pins[end_pin]
    y_2 = y_pins[end_pin]

    old_loss_contrib = 0.
    new_loss_contrib = 0.

    if x_1 == x_2:

        y_start = min(y_1, y_2)
        y_end = max(y_1, y_2)

        for y in range(y_start, y_end):

            old_loss_contrib += ( np.float32(saved_canvas[canvas.shape[0] - y - 1, x_1]) - np.float32(image[image.shape[0] - y - 1, x_1]) )**2
            new_loss_contrib += ( np.float32(canvas[canvas.shape[0] - y - 1, x_1]) - np.float32(image[image.shape[0] - y - 1, x_1]) )**2

    else:

        x_start = min(x_1, x_2)
        x_end = max(x_1, x_2)

        y_start = min(y_1, y_2)
        y_end = max(y_1, y_2)

        m = (y_2 - y_1) / (x_2 - x_1)

        if abs(m) < 1: 

            for x in range(x_start, x_end):

                y = round(m*(x - x_1) + y_1)
                old_loss_contrib += ( np.float32(saved_canvas[canvas.shape[0] - y - 1, x]) - np.float32(image[image.shape[0] - y - 1, x]) )**2
                new_loss_contrib += ( np.float32(canvas[canvas.shape[0] - y - 1, x]) - np.float32(image[image.shape[0] - y - 1, x]) )**2

        else:
                
            for y in range(y_start, y_end):

                x = round((y - y_1) / m + x_1)
                old_loss_contrib += ( np.float32(saved_canvas[canvas.shape[0] - y - 1, x]) - np.float32(image[image.shape[0] - y - 1, x]) )**2
                new_loss_contrib += ( np.float32(canvas[canvas.shape[0] - y - 1, x]) - np.float32(image[image.shape[0] - y - 1, x]) )**2


    # Finally update the loss function
    N = image.shape[0]*image.shape[1]
    new_loss = loss - 1/N*old_loss_contrib + 1/N*new_loss_contrib
    
    return new_loss


@njit(cache=True, fastmath=True)
def _add_optimal(loss, canvas, saved_canvas, x_pins, y_pins, last_pin, image, opaqueness):

    """
    Try all possible threads on 'canvas' and add to both 'canvas' and 'saved_canvas' 
    the one that minimizes locally the loss function between 'canvas' and 'image'.
    The x and y positions of all the pins ('x_pins' and 'y_pins') are needed to determine 
    the position of the pins on the canvas. 
    Returns (locally_minimized_loss, best_pin). 
    """

    best_loss = loss
    best_pin = last_pin

    for pin in range(x_pins.shape[0]):

        if pin == last_pin:

            continue
            
        _add_thread(canvas, x_pins, y_pins, last_pin, pin, opaqueness)
        new_loss = _compute_loss_optimal(canvas, saved_canvas, x_pins, y_pins, loss, last_pin, pin, image)

        if new_loss < best_loss:
                
            best_pin = pin
            best_loss = new_loss
            
        _restore_thread(canvas, saved_canvas, x_pins, y_pins, last_pin, pin)

    _add_thread(canvas, x_pins, y_pins, last_pin, best_pin, opaqueness)
    _add_thread(saved_canvas, x_pins, y_pins, last_pin, best_pin, opaqueness)

    return best_loss, best_pin


@njit(cache=True, fastmath=True)
def _optimize(loss, canvas, saved_canvas, x_pins, y_pins, last_pin, image, opaqueness, max_threads, tol):

    """
    Iteratively calls _add_optimal(loss, canvas, saved_canvas, x_pins, y_pins, last_pin, image, opaqueness)
    until the improvement in loss is smaller than 'tol' or until 'max_threads' is reached.
    It also prints the evolution of the loss function during the 'knitting'.
    Returns (loss, last pin).
    """

    power = 0
    for n in range(max_threads):

        if n == power:
            
            print(f"Loss function({n} threads) =", loss)
            power += 1000
        
        last_loss = loss
        loss, last_pin = _add_optimal(loss, canvas, saved_canvas, x_pins, y_pins, last_pin, image, opaqueness)
        
        if abs(last_loss - loss) < tol:
            
            print(f"Optimization converged after {n + 1} threads. Loss =", loss)
            break
        
    return loss, last_pin