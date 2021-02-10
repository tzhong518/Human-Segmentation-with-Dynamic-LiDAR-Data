#from . import backend as K
#from .utils.generic_utils import serialize_keras_object
#from .utils.generic_utils import deserialize_keras_object

import numpy as np

import keras.backend as K
from keras.initializers import Initializer


"""Initializer that generates the identity matrix.
    Only use for square 2D matrices.
    # Arguments
        gain: Multiplicative factor to apply to the identity matrix.
"""

class DeltaKernel(Initializer):
	def __init__(self, gain=1., stddev=0.):
		self.gain = gain
		self.stddev = stddev

	def __call__(self, shape, dtype=None):
		if( len(shape) != 4 ):
			raise ValueError( 'DeltaKernel can only be used for conv2d.' )
		else:
			if( shape[2] != shape[3] ):
				raise ValueError( 'Dimensions of input and output features should be same.' )
			else:
				if( self.stddev > 0 ):
					w = np.random.randn( shape[0], shape[1], shape[2], shape[3] ) * self.stddev
				else:
					w = np.zeros( shape )
				c0 = shape[0]//2
				c1 = shape[1]//2
				for channel in range(shape[3]):
					w[c0,c1,channel,channel] += self.gain
		'''
			if( K.image_data_format() == 'channels_last' ):
				if( shape[2] != shape[3] ):
					raise ValueError( 'Dimensions of input and output features should be same.' )
				else:
					w = np.zeros( shape )
					c0 = shape[0]//2
					c1 = shape[1]//2
					for channel in range(shape[3]):
						w[c0,c1,channel,channel] = gain

			else:
				if( shape[0] != shape[3] ):
					raise ValueError( 'Dimensions of input and output features should be same.' )
				else:
					w = np.zeros( shape )
					c1 = shape[1]//2
					c2 = shape[2]//2
					for channel in range(shape[3]):
						w[channel,c1,c1,channel] = gain
		'''		
		return w

	def get_config(self):
		return {
			'gain': self.gain,
			'stddev': self.stddev,
		}


if( __name__ == '__main__' ):
	from keras.layers import Convolution2D, Input
	from keras.models import Model
	
	#K.set_image_dim_ordering('th')

	#K.set_image_data_format('channels_first')
	#inp = Input(shape=(2,16,16))

	K.set_image_data_format('channels_last')
	inp = Input(shape=(16,16,2))
	
	print( K.image_data_format() )

	x = Convolution2D( 2, (3, 5), name='samp', use_bias=False, kernel_initializer=DeltaKernel(gain=1.4, std=0.001) ) (inp)
	model = Model(inputs=inp, outputs=x)
	
	layer = model.get_layer('samp')
	weights = layer.get_weights()
	print(weights[0].shape)
	for out_channel in range(weights[0].shape[3]):
		for in_channel in range(weights[0].shape[2]):
			print( weights[0][:,:,in_channel, out_channel] )
