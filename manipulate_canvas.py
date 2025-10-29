import numpy as np
import imageio.v3 as iio
import preparation_utilities as prep
import canvas_functions as cf


class canvas:

    def __init__(self, image, num_pins, opaqueness):

        '''
        Initialize the canvas as white. Choose the input image (image), the number of pins 
        (num_pins), and the opaqueness of the threads (opaqueness). The opaqueness must be 
        an integer between 1 and 255.
        '''

        self.image = image
        self.canvas = np.full( (image.shape[1], image.shape[0]), 255, dtype=np.uint8 )
        self.saved_canvas = self.canvas.copy()
        self.loss = self.compute_loss()
        self.x_pins = np.zeros(num_pins, dtype = int) # Store x positions of the pins
        self.y_pins = np.zeros(num_pins, dtype = int) # Store y positions of the pins
        self.num_pins = num_pins
        self.opaqueness = opaqueness
        self.last_pin = 0

        radius = self.canvas.shape[0] / 2
        theta_step = 2.*np.pi/num_pins
        
        # Set the x and y positions of the pins
        for n in range(num_pins):   

            self.x_pins[n] = round( radius*(1 + np.cos(n*theta_step) ) )
            self.y_pins[n] = round( radius*(1 + np.sin(n*theta_step) ) )

            self.x_pins[ self.x_pins >= self.canvas.shape[1] ] = self.canvas.shape[1] - 1
            self.y_pins[ self.y_pins >= self.canvas.shape[0] ] = self.canvas.shape[0] - 1

            self.x_pins[ self.x_pins < 0 ] = 0
            self.y_pins[ self.y_pins < 0 ] = 0
        
    def print(self, name = "image_output.png", pins = False):
        
        """
        Print the image with name 'name', if pins = True, print the pins in black.
        """
        
        cf._print(self.canvas, self.x_pins, self.y_pins, name, pins)
    
    def compute_loss(self):
        
        """
        Computes the loss between the canvas and input image.
        """
        
        loss = cf._compute_loss(self.canvas, self.image)

        return loss

    def optimize(self, max_threads=1000000, tol=1e-10):
        
        """
        Iteratively adds the locally best thread until the improvement in loss 
        is smaller than 'tol' or until 'max_threads' is reached.
        """
        
        self.loss, self.last_pin = cf._optimize(self.loss, self.canvas, self.saved_canvas, self.x_pins, self.y_pins, self.last_pin, self.image, self.opaqueness, max_threads, tol)