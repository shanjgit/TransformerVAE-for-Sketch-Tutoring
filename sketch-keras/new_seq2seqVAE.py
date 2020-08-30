import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Concatenate, RepeatVector, Dense, LSTM, Bidirectional, Lambda, Conv2D, Flatten, Add
from tensorflow.compat.v1.keras.layers import CuDNNLSTM
from tensorflow.keras.activations import softmax, tanh
from tensorflow.keras import backend as K
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import Adam, SGD
import numpy as np
import random
from transformer_decoder import *


def get_default_hparams():
	""" Return default hyper-parameters """
	params_dict = {
		'max_seq_len' : 250,
		'data_set' : ['sketchrnn_bird.full.npz'],	#datasets to train on
		'img_H' : 64,	# image height
		'img_W' : 64,	# image width
		# Experiment Params:
		'is_training': True,  # train mode (relevant only for accelerated LSTM mode)
		'epochs': 100,	# how many times to go over the full train set (on average, since batches are drawn randomly)
		'save_every': None, # Batches between checkpoints creation and validation set evaluation. Once an epoch if None.
		'batch_size': 100,	# Minibatch size. Recommend leaving at 100.
		'accelerate_LSTM': False,  # Flag for using CuDNNLSTM layer, gpu + tf backend only
		# Loss Params:
		'optimizer': 'adam',  # adam or sgd
		'learning_rate': 0.001,
		'decay_rate': 0.9999,  # Learning rate decay per minibatch.
		'min_learning_rate': .00001,  # Minimum learning rate.
		'kl_tolerance': 0.2,  # Level of KL loss at which to stop optimizing for KL.
		'kl_weight': 0.5,  # KL weight of loss equation. Recommend 0.5 or 1.0.
		'kl_weight_start': 0.01,  # KL start weight when annealing.
		'kl_decay_rate': 0.99995,  # KL annealing decay rate per minibatch.
		'grad_clip': 1.0,  # Gradient clipping. Recommend leaving at 1.0.
		# Architecture Params:
		'z_size': 64,	# Size of latent vector z. Recommended 32, 64 or 128.
		'enc_rnn_size': 128,  # Units in encoder RNN.
		'dec_size': 256,  # Units in decoder.
		'use_recurrent_dropout': True,	# Dropout with memory loss. Recommended
		'recurrent_dropout_prob': 0.9,	# Probability of recurrent dropout keep.
		'num_mixture': 20,	# Number of mixtures in Gaussian mixture model.
		# Data pre-processing Params:
		'random_scale_factor': 0.15,  # Random scaling data augmentation proportion.
		'augment_stroke_prob': 0.10,  # Point dropping augmentation proportion.
		'opencv': True, # Use opencv to scale the image drawn with strokes
		'stroke_resize': True, # Resize the strokes to 255 x 255
		'normalizing_scale_factor': 1, # Normalizing scale factor of strokes
		'index_list': False, # Index list of clean data
		'num_layers': 2, # number of layers of decoder layer in decoder of transformer
		'num_heads': 8, # number of heads in decoder with transformer
	}

	return params_dict


class Seq2seqModel(object):

	def __init__(self, hps):
		# Hyper parameters
		self.hps = hps
		# Model
		self.model = self.build_model()
		# Print a model summary
		self.model.summary()

		# Optimizer
		if self.hps['optimizer'] == 'adam':
			self.optimizer = Adam(lr=self.hps['learning_rate'], clipvalue=self.hps['grad_clip'])
		elif self.hps['optimizer'] == 'sgd':
			self.optimizer = SGD(lr=self.hps['learning_rate'], momentum=0.9, clipvalue=self.hps['grad_clip'])
		else:
			raise ValueError('Unsupported Optimizer!')
		# Loss Function
		self.loss_func = self.model_loss()
		# Sample models, to be used when encoding\decoding specific strokes
		self.sample_models = {}
	
	def high_pass_filter(self, shape, dtype = None, partition_info = None):
		
		f = np.array([[[[-1]], [[-1]], [[-1]]], [[[-1]], [[9]], [[-1]]], [[[-1]], [[-1]], [[-1]]]])
		print(f.shape)
		print(shape)
		return K.constant(f, dtype = 'float32')

	def build_model(self):
		""" Create a Keras seq2seq VAE model for sketch-rnn """

		# Arrange inputs:
		self.encoder_input = Input(shape=(self.hps['img_H'], self.hps['img_W'], 1), name='encoder_input')
		self.decoder_input = Input(shape=(self.hps['max_seq_len'], 5), name='decoder_input')

		# Set recurrent dropout to fraction of units to *drop*, if in CuDNN accelerated mode, don't use dropout:
		recurrent_dropout = 1.0-self.hps['recurrent_dropout_prob'] if \
			(self.hps['use_recurrent_dropout'] and (self.hps['accelerate_LSTM'] is False)) else 0
		
		'''
		# Option to use the accelerated version of LSTM, CuDNN LSTM. Much faster, but no support for recurrent dropout:
		if self.hps['accelerate_LSTM'] and self.hps['is_training']:
			#lstm_layer_encoder = CuDNNLSTM(units=self.hps['enc_rnn_size'])
			lstm_layer_decoder = CuDNNLSTM(units=self.hps['dec_rnn_size'], return_sequences=True, return_state=True)
			self.hps['use_recurrent_dropout'] = False
			print('Using CuDNNLSTM - No Recurrent Dropout!')
		else:
			# Note that in inference LSTM is always selected (even in accelerated mode) so inference on CPU is supported
			#lstm_layer_encoder = LSTM(units=self.hps['enc_rnn_size'], recurrent_dropout=recurrent_dropout)
			lstm_layer_decoder = LSTM(units=self.hps['dec_rnn_size'], recurrent_dropout=recurrent_dropout,
									  return_sequences=True, return_state=True)
		'''
		# Encoder:
		conv_input = Input(shape=(self.hps['img_H'], self.hps['img_W'], 1), name='conv_input')
		conv2D_1 = Conv2D(filters = 4, kernel_size = (2, 2), activation = 'relu', padding = 'same', strides = (2, 2), name = 'conv2D_1')(conv_input)
		conv2D_2 = Conv2D(filters = 4, kernel_size = (2, 2), activation = 'relu', padding = 'same', strides = (1, 1), name = 'conv2D_2')(conv2D_1)
		conv2D_3 = Conv2D(filters = 8, kernel_size = (2, 2), activation = 'relu', padding = 'same', strides = (2, 2), name = 'conv2D_3')(conv2D_2)
		conv2D_4 = Conv2D(filters = 8, kernel_size = (2, 2), activation = 'relu', padding = 'same', strides = (1, 1), name = 'conv2D_4')(conv2D_3)
		conv2D_5 = Conv2D(filters = 8, kernel_size = (2, 2), activation = 'relu', padding = 'same', strides = (2, 2), name = 'conv2D_5')(conv2D_4)
		conv2D_6 = Conv2D(filters = 8, kernel_size = (2, 2), activation = 'tanh', padding = 'same', strides = (1, 1), name = 'conv2D_6')(conv2D_5)
		encoder_output = Flatten()(conv2D_6)
		# Dense layers to create the mean and stddev of the latent vector
		mu = Dense(units=self.hps['z_size'], kernel_initializer=RandomNormal(stddev=0.001))(encoder_output)
		sigma = Dense(units=self.hps['z_size'], kernel_initializer=RandomNormal(stddev=0.001))(encoder_output)
				
		self.conv_model = Model(conv_input, [mu, sigma])
		high_pass = Conv2D(filters = 1, kernel_size = (3, 3), kernel_initializer = self.high_pass_filter, padding = 'same', name = 'high_pass', trainable = False)(self.encoder_input)
		[self.mu, self.sigma] = self.conv_model(high_pass)
		
		# Latent vector - [batch_size]X[z_size]:
		self.batch_z = self.latent_z(self.mu, self.sigma)

		# Decoder:
		####self.decoder = Decoder(self.hps['num_layers'], self.hps['dec_size'], self.hps['num_heads'], self.hps['dec_size'] * 4, self.hps['max_seq_len'])
		#def __init__(          num_layers            , d_model             , num_heads            , dff,  maximum_position_encoding, rate=0.1, **kwargs):
		
		'''
		# Initial state for decoder:
		self.initial_state = Dense(units=2*self.decoder.units, activation='tanh', name='dec_initial_state',
							  kernel_initializer=RandomNormal(mean=0.0, stddev=0.001))
		initial_state = self.initial_state(self.batch_z)

		# Split to hidden state and cell state:
		init_h, init_c = (initial_state[:, :self.decoder.units], initial_state[:, self.decoder.units:])
		'''

		self.embedding = tf.keras.layers.Dense(self.hps['dec_size'])
		self.pos_encoding = positional_encoding(self.hps['max_seq_len'], self.hps['dec_size'])
		
		print(self.pos_encoding)
		print('num_layers', self.hps['num_layers'])
		self.dec_layers = [DecoderLayer(self.hps['dec_size'], self.hps['num_heads'], self.hps['dec_size'] * 4, 0.1)
							 for _ in range(self.hps['num_layers'])]

		self.wq = [tf.keras.layers.Dense(self.hps['dec_size']) for _ in range(self.hps['num_layers'])]
		self.wk = [tf.keras.layers.Dense(self.hps['dec_size']) for _ in range(self.hps['num_layers'])]
		self.wv = [tf.keras.layers.Dense(self.hps['dec_size']) for _ in range(self.hps['num_layers'])]


		self.dropout = tf.keras.layers.Dropout(0.1)



		self.dense = [tf.keras.layers.Dense(self.hps['dec_size']) for _ in range(self.hps['num_layers'])]

		self.dense1 = [tf.keras.layers.Dense(self.hps['dec_size'] * 4, activation='relu') for _ in range(self.hps['num_layers'])]
		self.dense2 = [tf.keras.layers.Dense(self.hps['dec_size']) for _ in range(self.hps['num_layers'])]
		self.ffn = [tf.keras.Sequential([
					tf.keras.layers.Dense(self.hps['dec_size'] * 4, activation='relu'),	# (batch_size, seq_len, dff)
					tf.keras.layers.Dense(self.hps['dec_size'])	# (batch_size, seq_len, d_model)
					]) for _ in range(self.hps['num_layers'])]
	 
		self.layernorm1 = [tf.keras.layers.LayerNormalization(epsilon=1e-6) for _ in range(self.hps['num_layers'])]
		self.layernorm2 = [tf.keras.layers.LayerNormalization(epsilon=1e-6) for _ in range(self.hps['num_layers'])]
		
		self.dropout1 = [tf.keras.layers.Dropout(0.1) for _ in range(self.hps['num_layers'])]
		self.dropout2 = [tf.keras.layers.Dropout(0.1) for _ in range(self.hps['num_layers'])]



		# Concatenate latent vector to expected output and	feed this as input to decoder:
		tile_z = RepeatVector(self.hps['max_seq_len'])(self.batch_z)
		decoder_full_input = Concatenate()([self.decoder_input, tile_z])

		# Retrieve decoder output tensors:
		#######decoder_output, attention = self.decoder(decoder_full_input, True)
		# self.final_state = [final_state1, final_state_2] todo: not used, remove when stable





		seq_len = tf.shape(decoder_full_input)[1]
		self.look_ahead_mask = create_look_ahead_mask(seq_len)
		attention_weights = {}

		decoder_embedded = self.embedding(decoder_full_input)	# (batch_size, target_seq_len, self.hps['dec_size'])
		decoder_embedded_new = decoder_embedded

		print(decoder_embedded)
		print(self.pos_encoding[:, :seq_len, :])
		decoder_embedded_new *= tf.math.sqrt(tf.cast(self.hps['dec_size'], tf.float32))
		decoder_embedded_new += self.pos_encoding[:, :seq_len, :]

		decoder_embedded_new = self.dropout(decoder_embedded_new, training=True)
		#decoder_output = decoder_pos_encode

		#x = decoder_output
		#training = True
		
		def split_heads(x, batch_size):
			depth = self.hps['dec_size'] // self.hps['num_heads']
			x = tf.reshape(x, (batch_size, -1, self.hps['num_heads'], depth))
			return tf.transpose(x, perm=[0, 2, 1, 3])
		
		q = [0 for _ in range(self.hps['num_layers'])]
		k = [0 for _ in range(self.hps['num_layers'])]
		v = [0 for _ in range(self.hps['num_layers'])]
		qs = [0 for _ in range(self.hps['num_layers'])]
		ks = [0 for _ in range(self.hps['num_layers'])]
		vs = [0 for _ in range(self.hps['num_layers'])]
		'''
		matmul_qk = [0 for _ in range(self.hps['num_layers'])]
		dk = [0 for _ in range(self.hps['num_layers'])]
		scaled_attention_logits = [0 for _ in range(self.hps['num_layers'])]
		'''
		scaled_attention = [0 for _ in range(self.hps['num_layers'])]
		attn_weights_block = [0 for _ in range(self.hps['num_layers'])]
		concat_attention = [0 for _ in range(self.hps['num_layers'])]
		attn = [0 for _ in range(self.hps['num_layers'])]
		out1 = [0 for _ in range(self.hps['num_layers'])]
		ffn_output = [0 for _ in range(self.hps['num_layers'])]
		self.decoder_output = [0 for _ in range(self.hps['num_layers']+1)]
		self.decoder_output[0] = decoder_embedded_new


		for i in range(self.hps['num_layers']):
			#attn, attn_weights_block = self.mha(decoder_output, decoder_output, decoder_output, look_ahead_mask)	# (batch_size, target_seq_len, d_model)

			batch_size = tf.shape(self.decoder_output[i])[0]

			q[i] = self.wq[i](self.decoder_output[i])	# (batch_size, seq_len, d_model)
			k[i] = self.wk[i](self.decoder_output[i])	# (batch_size, seq_len, d_model)
			v[i] = self.wv[i](self.decoder_output[i])	# (batch_size, seq_len, d_model)
			

			qs[i] = split_heads(q[i], batch_size)	# (batch_size, num_heads, seq_len_q, depth)
			ks[i] = split_heads(k[i], batch_size)	# (batch_size, num_heads, seq_len_k, depth)
			vs[i] = split_heads(v[i], batch_size)	# (batch_size, num_heads, seq_len_v, depth)
			
			# scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
			# attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
			scaled_attention[i], attn_weights_block[i] = scaled_dot_product_attention(qs[i], ks[i], vs[i], self.look_ahead_mask)
			'''
			###################### scaled_dot_product_attention ########################
			matmul_qk[i] = tf.matmul(qs[i], ks[i], transpose_b=True)	# (..., seq_len_q, seq_len_k)
	
			# scale matmul_qk
			dk[i] = tf.cast(tf.shape(ks[i])[-1], tf.float32)
			scaled_attention_logits[i] = matmul_qk[i] / tf.math.sqrt(dk[i])

			# add the mask to the scaled tensor.
			if self.look_ahead_mask is not None:
				scaled_attention_logits[i] += (self.look_ahead_mask * -1e9)	

			# softmax is normalized on the last axis (seq_len_k) so that the scores
			# add up to 1.
			attn_weights_block[i] = tf.nn.softmax(scaled_attention_logits[i], axis=-1)	# (..., seq_len_q, seq_len_k)

			scaled_attention[i] = tf.matmul(attn_weights_block[i], vs[i])	# (..., seq_len_q, depth_v)
			
			'''

			scaled_attention[i] = tf.transpose(scaled_attention[i], perm=[0, 2, 1, 3])	# (batch_size, seq_len_q, num_heads, depth)

			concat_attention[i] = tf.reshape(scaled_attention[i], 
											(batch_size, -1, self.hps['dec_size']))	# (batch_size, seq_len_q, d_model)
			attn[i] = self.dense[i](concat_attention[i])	# (batch_size, seq_len_q, d_model)
			



			attn[i] = self.dropout1[i](attn[i], training=True)
			out1[i] = self.layernorm1[i](attn[i] + self.decoder_output[i])
			
			#ffn_output[i] = self.ffn[i](out1[i])	# (batch_size, target_seq_len, d_model)
			ffn_output[i] = self.dense1[i](out1[i])	# (batch_size, target_seq_len, d_model)
			ffn_output[i] = self.dense2[i](ffn_output[i])

			ffn_output[i] = self.dropout2[i](ffn_output[i], training=True)
			self.decoder_output[i+1] = self.layernorm2[i](ffn_output[i] + out1[i])	# (batch_size, target_seq_len, d_model)
			
			#decoder_output, block = self.dec_layers[i](decoder_output, True, look_ahead_mask)
			attention_weights['decoder_layer{}_block'.format(i+1)] = attn_weights_block[i]









		# Number of outputs:
		# 3 pen state logits, 6 outputs per mixture model(mean_x, mean_y, std_x, std_y, corr_xy, mixture weight pi)
		n_out = (3 + self.hps['num_mixture'] * 6)

		# Output FC layer
		self.output = Dense(n_out, name='output')
		output = self.output(self.decoder_output[self.hps['num_layers']])

		# Build Keras model
		model_o = Model([self.encoder_input, self.decoder_input], output)

		

		return model_o

	def latent_z(self, mu, sigma):
		""" Return a latent vector z of size [batch_size]X[z_size] """

		def transform2layer(z_params):
			""" Auxiliary function to feed into Lambda layer.
			 Gets a list of [mu, sigma] and returns a random tensor from the corresponding normal distribution """
			mu, sigma = z_params
			sigma_exp = K.exp(sigma / 2.0)
			colored_noise = mu + sigma_exp*K.random_normal(shape=K.shape(sigma_exp), mean=0.0, stddev=1.0)
			return colored_noise

		# We cannot simply use the operations and feed to the next layer, so a Lambda layer must be used
		return Lambda(transform2layer)([mu, sigma])

	def calculate_kl_loss(self, *args, **kwargs):
		""" Function to calculate the KL loss term.
		 Considers the tolerance value for which optimization for KL should stop """
		# kullback Leibler loss between normal distributions
		kl_cost = -0.5*K.mean(1+self.sigma-K.square(self.mu)-K.exp(self.sigma))

		return K.maximum(kl_cost, self.hps['kl_tolerance'])

	def calculate_md_loss(self, y_true, y_pred):
		""" calculate reconstruction (mixture density) loss """
		# Parse the output tensor to appropriate mixture density coefficients:
		out = self.get_mixture_coef(y_pred)
		[o_pi, o_mu1, o_mu2, o_sigma1, o_sigma2, o_corr, o_pen, o_pen_logits] = out

		# Parse target vector to coordinates and pen states:
		[x1_data, x2_data] = [y_true[:, :, 0], y_true[:, :, 1]]
		pen_data = y_true[:, :, 2:5]

		# Get the density value of each mixture, estimated for the target coordinates:
		pdf_values = self.keras_2d_normal(x1_data, x2_data, o_mu1, o_mu2, o_sigma1, o_sigma2, o_corr)

		# Compute the GMM values (weighted sum of mixtures using pi values)
		gmm_values = pdf_values * o_pi
		gmm_values = K.sum(gmm_values, 2, keepdims=True)

		# gmm_loss is the loss wrt pen offset (L_s in equation 9 of https://arxiv.org/pdf/1704.03477.pdf)
		epsilon = 1e-6
		gmm_loss = -K.log(gmm_values + epsilon)  # avoid log(0)

		# Zero out loss terms beyond N_s, the last actual stroke
		fs = 1.0 - pen_data[:, :, 2]
		fs = K.expand_dims(fs)
		gmm_loss = gmm_loss * fs

		# pen_loss is the loss wrt pen state, (L_p in equation 9)
		pen_loss = categorical_crossentropy(pen_data, o_pen)
		pen_loss = K.expand_dims(pen_loss)

		# Eval mode, mask eos columns. todo: remove this?
		pen_loss = K.switch(tf.cast(K.learning_phase(), dtype = tf.bool), pen_loss, pen_loss * fs)

		# Total loss
		result = gmm_loss + pen_loss

		r_cost = K.mean(result)  # todo: Keras already averages over all tensor values, this might be redundant
		return r_cost

	def model_loss(self):
		"""" Wrapper function which calculates auxiliary values for the complete loss function.
		 Returns a *function* which calculates the complete loss given only the input and target output """
		# KL loss
		kl_loss = self.calculate_kl_loss
		# Reconstruction loss
		md_loss_func = self.calculate_md_loss

		# KL weight (to be used by total loss and by annealing scheduler)
		self.kl_weight = K.variable(self.hps['kl_weight_start'], name='kl_weight')
		kl_weight = self.kl_weight

		def seq2seq_loss(y_true, y_pred):
			""" Final loss calculation function to be passed to optimizer"""
			# Reconstruction loss
			md_loss = md_loss_func(y_true, y_pred)
			# Full loss
			model_loss = kl_weight*kl_loss() + md_loss
			return model_loss

		return seq2seq_loss

	def get_mixture_coef(self, out_tensor):
		""" Parses the output tensor to appropriate mixture density coefficients"""
		# This uses eqns 18 -> 23 of http://arxiv.org/abs/1308.0850.

		# Pen states:
		z_pen_logits = out_tensor[:, :, 0:3]
		# Process outputs into MDN parameters
		M = self.hps['num_mixture']
		dist_params = [out_tensor[:, :, (3 + M * (n - 1)):(3 + M * n)] for n in range(1, 7)]
		z_pi, z_mu1, z_mu2, z_sigma1, z_sigma2, z_corr = dist_params

		# Softmax all the pi's and pen states:
		z_pi = softmax(z_pi)
		z_pen = softmax(z_pen_logits)

		# Exponent the sigmas and also make corr between -1 and 1.
		z_sigma1 = K.exp(z_sigma1)
		z_sigma2 = K.exp(z_sigma2)
		z_corr = tanh(z_corr)

		r = [z_pi, z_mu1, z_mu2, z_sigma1, z_sigma2, z_corr, z_pen, z_pen_logits]
		return r

	def keras_2d_normal(self, x1, x2, mu1, mu2, s1, s2, rho):
		""" Returns the density values of each mixture, estimated for the target coordinates tensors
		This is the result of eq # 24 of http://arxiv.org/abs/1308.0850."""
		M = mu1.shape[2]  # Number of mixtures
		norm1 = tf.tile(K.expand_dims(x1), [1, 1, M]) - mu1
		norm2 = tf.tile(K.expand_dims(x2), [1, 1, M]) - mu2
		s1s2 = s1 * s2
		# eq 25
		z = K.square(norm1 / s1) + K.square(norm2 / s2) - 2.0 * (rho * norm1 * norm2) / s1s2
		neg_rho = 1.0 - K.square(rho)
		result = K.exp((-z) / (2 * neg_rho))
		denom = 2 * np.pi * s1s2 * K.sqrt(neg_rho)
		result = result / denom
		return result

	def compile(self):
		""" Compiles the Keras model. Includes metrics to differentiate between the two main loss terms """
		self.model.compile(experimental_run_tf_function=False, optimizer=self.optimizer, loss=self.loss_func,
						   metrics=[self.calculate_md_loss, self.calculate_kl_loss])
		print('Model Compiled!')

	def load_trained_weights(self, weights):
		""" Loads weights of a pre-trained model. 'weights' is path to h5 model\weights file"""
		self.model.load_weights(weights)
		#self.model = tf.keras.models.load_model(weights)
		print('Weights from {} loaded successfully'.format(weights))

	def make_sampling_models(self):
		""" Creates models for various input-output combinations
		 that are used when sampling and encoding\decoding specific strokes """
		models = {}
				
		'''
		# Model 1: Latent vector to initial state. Model gets batch_z and outputs initial_state
		batch_z = Input(shape=(self.hps['z_size'],))
		initial_state = self.initial_state(batch_z)
		models['z_to_init_model'] = Model(inputs=batch_z, outputs=initial_state)
		print('create z_to_init_model!')
		'''

		# Model 2: sample_output_model. Model that gets decoder input, initial state and batch_z.
		# Outputs final state and mixture parameters.

		# Inputs:



		decoder_input = Input(shape=(None, 5))
		batch_z = Input(shape=(self.hps['z_size'],))
		def repeat_vector(args):
			layer_to_repeat = args[0]
			sequence_layer = args[1]
			print('=============================================',layer_to_repeat, sequence_layer)
			print(K.shape(sequence_layer)[1])
			return RepeatVector(K.shape(sequence_layer)[1])(layer_to_repeat)

		tile_z = Lambda(repeat_vector, output_shape=(None, self.hps['z_size'])) ([batch_z, decoder_input])



		decoder_full_input = Concatenate()([decoder_input, tile_z])


		seq_len = tf.shape(decoder_full_input)[1]
		look_ahead_mask = 1 - tf.linalg.band_part(tf.ones((seq_len, seq_len), name='new_ones'), -1, 0, name='new_mask')#create_look_ahead_mask(seq_len)
		attention_weights = {}

		decoder_embedded = self.embedding(decoder_full_input)	# (batch_size, target_seq_len, self.hps['dec_size'])
		decoder_embedded_new = decoder_embedded
		decoder_embedded_new *= tf.math.sqrt(tf.cast(self.hps['dec_size'], tf.float32))
		decoder_embedded_new += self.pos_encoding[:, :seq_len, :]

		decoder_embedded_new = self.dropout(decoder_embedded_new, training=True)
		#decoder_output = decoder_pos_encode



		

		#x = decoder_output
		#training = True
		
		def split_heads(x, batch_size):
			depth = self.hps['dec_size'] // self.hps['num_heads']
			x = tf.reshape(x, (batch_size, -1, self.hps['num_heads'], depth))
			return tf.transpose(x, perm=[0, 2, 1, 3])
		
		q = [0 for _ in range(self.hps['num_layers'])]
		k = [0 for _ in range(self.hps['num_layers'])]
		v = [0 for _ in range(self.hps['num_layers'])]
		qs = [0 for _ in range(self.hps['num_layers'])]
		ks = [0 for _ in range(self.hps['num_layers'])]
		vs = [0 for _ in range(self.hps['num_layers'])]
		'''
		matmul_qk = [0 for _ in range(self.hps['num_layers'])]
		dk = [0 for _ in range(self.hps['num_layers'])]
		scaled_attention_logits = [0 for _ in range(self.hps['num_layers'])]
		scaled_attention_logits1 = [0 for _ in range(self.hps['num_layers'])]
		'''
		scaled_attention = [0 for _ in range(self.hps['num_layers'])]
		#scaled_attention1 = [0 for _ in range(self.hps['num_layers'])]
		attn_weights_block = [0 for _ in range(self.hps['num_layers'])]
		concat_attention = [0 for _ in range(self.hps['num_layers'])]
		attn = [0 for _ in range(self.hps['num_layers'])]
		out1 = [0 for _ in range(self.hps['num_layers'])]
		ffn_output = [0 for _ in range(self.hps['num_layers'])]
		decoder_output = [0 for _ in range(self.hps['num_layers']+1)]
		decoder_output[0] = decoder_embedded_new


		for i in range(self.hps['num_layers']):
			#attn, attn_weights_block = self.mha(decoder_output, decoder_output, decoder_output, look_ahead_mask)	# (batch_size, target_seq_len, d_model)

			batch_size = tf.shape(decoder_output[i])[0]
			

			q[i] = self.wq[i](decoder_output[i])	# (batch_size, seq_len, d_model)
			k[i] = self.wk[i](decoder_output[i])	# (batch_size, seq_len, d_model)
			v[i] = self.wv[i](decoder_output[i])	# (batch_size, seq_len, d_model)
			

			qs[i] = split_heads(q[i], batch_size)	# (batch_size, num_heads, seq_len_q, depth)
			ks[i] = split_heads(k[i], batch_size)	# (batch_size, num_heads, seq_len_k, depth)
			vs[i] = split_heads(v[i], batch_size)	# (batch_size, num_heads, seq_len_v, depth)
			
			# scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
			# attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
			scaled_attention[i], attn_weights_block[i] = scaled_dot_product_attention(qs[i], ks[i], vs[i], look_ahead_mask)
			
	
			scaled_attention[i] = tf.transpose(scaled_attention[i], perm=[0, 2, 1, 3])	# (batch_size, seq_len_q, num_heads, depth)

			concat_attention[i] = tf.reshape(scaled_attention[i], 
											(batch_size, -1, self.hps['dec_size']))	# (batch_size, seq_len_q, d_model)
			attn[i] = self.dense[i](concat_attention[i])	# (batch_size, seq_len_q, d_model)
			



			attn[i] = self.dropout1[i](attn[i], training=True)
			out1[i] = self.layernorm1[i](attn[i] + decoder_output[i])
			
			#ffn_output[i] = self.ffn[i](out1[i])	# (batch_size, target_seq_len, d_model)
			ffn_output[i] = self.dense1[i](out1[i])	# (batch_size, target_seq_len, d_model)
			ffn_output[i] = self.dense2[i](ffn_output[i])
			ffn_output[i] = self.dropout2[i](ffn_output[i], training=True)
			decoder_output[i+1] = self.layernorm2[i](ffn_output[i] + out1[i])	# (batch_size, target_seq_len, d_model)
			
			#decoder_output, block = self.dec_layers[i](decoder_output, True, look_ahead_mask)
			attention_weights['decoder_layer{}_block'.format(i+1)] = attn_weights_block[i]

		# x.shape == (batch_size, target_seq_len, self.hps['dec_size'])

		

		'''
		[decoder_output, final_state_1, final_state_2] = self.decoder(decoder_full_input,
																	  initial_state=[initial_h_input, initial_c_input])
		final_state = [final_state_1, final_state_2]
		'''
		# Apply original output layer
		model_output = self.output(decoder_output[self.hps['num_layers']])
		# Get mixture coef' based on output layer
		mixture_params = Lambda(self.get_mixture_coef)(model_output[:, -1:, :])
		'''
		models['sample_output_model'] = Model(inputs=[decoder_input, initial_h_input, initial_c_input, batch_z],
											  outputs=final_state + mixture_params)
		'''
		
		'''
		models['decoder_to_embed_model'] = Model(inputs = [decoder_input, batch_z], outputs = decoder_embedded)
		print('create decoder_to_embed_model!')
		'''

		

		models['sample_output_model'] = Model(inputs = [decoder_input, batch_z], outputs = mixture_params)
		print('create sample_output_model!')

		models['decoder_model'] = Model(inputs = [decoder_input, batch_z], outputs = decoder_output[self.hps['num_layers']])
		print('create decoder_model!')

		em_input = Input(shape = (None, 5+self.hps['z_size']))
		em_output = self.embedding(em_input)
		models['input_to_embed_dense'] = Model(inputs = em_input, outputs = em_output)
		print('input_to_embed_dense!')

		wq_input = [0 for _ in range(self.hps['num_layers'])]
		wk_input = [0 for _ in range(self.hps['num_layers'])]
		wv_input = [0 for _ in range(self.hps['num_layers'])]
		concat_input = [0 for _ in range(self.hps['num_layers'])]
		decoder_output_input = [0 for _ in range(self.hps['num_layers'])]
		tmp = [0 for _ in range(self.hps['num_layers'])]
		tmp1 = [0 for _ in range(self.hps['num_layers'])]
		tmp_out = [0 for _ in range(self.hps['num_layers'])]
		ffn_tmp = [0 for _ in range(self.hps['num_layers'])]
		ffn_tmp1 = [0 for _ in range(self.hps['num_layers'])]

		wq_output = [0 for _ in range(self.hps['num_layers'])]
		wk_output = [0 for _ in range(self.hps['num_layers'])]
		wv_output = [0 for _ in range(self.hps['num_layers'])]
		concat_output = [0 for _ in range(self.hps['num_layers'])]
		for i in range(self.hps['num_layers']):
			wq_input[i] = Input(shape = (None, self.hps['dec_size']))
			wk_input[i] = Input(shape = (None, self.hps['dec_size']))
			wv_input[i] = Input(shape = (None, self.hps['dec_size']))
			wq_output[i] = self.wq[i](wq_input[i])
			wk_output[i] = self.wk[i](wk_input[i])
			wv_output[i] = self.wv[i](wv_input[i])

			concat_input[i] = Input(shape = (None, self.hps['dec_size']))
			decoder_output_input[i] = Input(shape = (None, self.hps['dec_size']))
			#concat_output[i] = self.dense[i](concat_input[i] + decoder_output_input[i])
			tmp[i] = self.dense[i](concat_input[i])
			tmp[i] = self.dropout1[i](tmp[i], training=True)
			tmp1[i] = Add()([tmp[i], decoder_output_input[i]])
			tmp_out[i] = self.layernorm1[i](tmp1[i])# + decoder_output_input[i])

			#ffn_tmp[i] = self.ffn[i](tmp_out[i])
			ffn_tmp[i] = self.dense1[i](tmp_out[i])
			ffn_tmp[i] = self.dense2[i](ffn_tmp[i])
			ffn_tmp[i] = self.dropout2[i](ffn_tmp[i], training=True)
			ffn_tmp1[i] = Add()([ffn_tmp[i], tmp_out[i]])
			concat_output[i] = self.layernorm2[i](ffn_tmp1[i])# + tmp_out[i])

		for i in range(self.hps['num_layers']):
			models['wq_dense_'+str(i)] = Model(inputs = wq_input[i], outputs = wq_output[i])
			models['wk_dense_'+str(i)] = Model(inputs = wk_input[i], outputs = wk_output[i])
			models['wv_dense_'+str(i)] = Model(inputs = wv_input[i], outputs = wv_output[i])
			models['concat_attention_to_decoder_output_'+str(i)+'_model'] = Model(inputs = [concat_input[i], decoder_output_input[i]], outputs = concat_output[i])
			#models['concat_attention_to_decoder_output_'+str(i)+'_model'] = Model(inputs = concat_input[i], outputs = concat_output[i])
		

		# Model 3: Encoder model. Stroke input to latent vector
		models['encoder_model'] = Model(inputs=self.encoder_input, outputs=[self.mu, self.sigma])
		print('create encoder_model!')
		

		models['encoder_model_without_highpass'] = self.conv_model
		print('create encoder_model_without_highpass!')

		output_layer_input = Input(shape = (None, self.hps['dec_size']))
		models['output_layer_model'] = Model(output_layer_input, self.output(output_layer_input))
		print('create output_layer_model!')
	
		self.sample_models = models
		print('Created Sub Models!')


def sample(seq2seq_model, seq_len=250, temperature=1.0, greedy_mode=False, z=None):
	"""Samples a sequence from a pre-trained model."""

	hps = seq2seq_model.hps

	def adjust_temp(pi_pdf, temp):
		pi_pdf = np.log(pi_pdf) / temp
		pi_pdf -= pi_pdf.max()
		pi_pdf = np.exp(pi_pdf)
		pi_pdf /= pi_pdf.sum()
		return pi_pdf

	def get_pi_idx(x, pdf, temp=1.0, greedy=False):
		"""Samples from a pdf, optionally greedily."""
		if greedy:
			return np.argmax(pdf)
		pdf = adjust_temp(np.copy(pdf), temp)
		accumulate = 0
		for i in range(0, pdf.size):
			accumulate += pdf[i]
			if accumulate >= x:
				return i
		print('Error with sampling ensemble.')
		return -1

	def sample_gaussian_2d(mu1, mu2, s1, s2, rho, temp=1.0, greedy=False):
		""" Sample from a bivariate normal distribution """
		if greedy:
			return mu1, mu2
		mean = [mu1, mu2]
		s1 *= temp * temp
		s2 *= temp * temp
		cov = [[s1 * s1, rho * s1 * s2], [rho * s1 * s2, s2 * s2]]
		x = np.random.multivariate_normal(mean, cov, 1)
		return x[0][0], x[0][1]

	# Initial stroke:
	prev_x = np.zeros((1, 1, 5), dtype=np.float32)
	prev_x[0, 0, 2] = 1

	# If latent vector is not specified:
	if z is None:
		z = np.random.randn(1, hps['z_size'])

	# Get the model that gets batch_z and outputs initial_state:
	'''
	z_to_init_model = seq2seq_model.sample_models['z_to_init_model']
	# Get value of initial state:
	prev_state = z_to_init_model.predict(z)
	# Split to hidden and cell states:
	prev_state = [prev_state[:, :seq2seq_model.decoder.units], prev_state[:, seq2seq_model.decoder.units:]]
	'''

	# Get the model that gets decoder input, initial state and batch_z. outputs final state and mixture parameters.
	sample_output_model = seq2seq_model.sample_models['sample_output_model']
	output_layer_model = seq2seq_model.sample_models['output_layer_model']
	input_to_embed_dense = seq2seq_model.sample_models['input_to_embed_dense']
	wq_dense = [seq2seq_model.sample_models['wq_dense_'+str(i)] for i in range(hps['num_layers'])]
	wk_dense = [seq2seq_model.sample_models['wk_dense_'+str(i)] for i in range(hps['num_layers'])]
	wv_dense = [seq2seq_model.sample_models['wv_dense_'+str(i)] for i in range(hps['num_layers'])]
	concat_attention_to_decoder_output = [seq2seq_model.sample_models['concat_attention_to_decoder_output_'+str(i)+'_model']  for i in range(hps['num_layers'])]
	# Initialize strokes matrix
	strokes = np.zeros((seq_len, 5), dtype=np.float32)
	mixture_params = []

	for i in range(seq_len-1):
		
		'''
		# Arrange inputs
		feed = {
			'decoder input': prev_x,
			'initial_state': prev_state,
			'batch_z': z
		}
		# Get decoder states and mixture parameters:
		model_outputs_list = sample_output_model.predict([feed['decoder input'],
														  feed['initial_state'][0],
														  feed['initial_state'][1],
														  feed['batch_z']])
		'''
		feed = {
			'decoder input': np.concatenate((prev_x, np.expand_dims(strokes[:i], axis = 0)), axis = 1),
			'batch_z': z
		}
		# Get decoder states and mixture parameters:
		### model_output = sample_output_model.predict([feed['decoder input'], feed['batch_z']])
		
		def repeat_vector(args):
			layer_to_repeat = args[0]
			sequence_layer = args[1]
			return RepeatVector(K.shape(sequence_layer)[1])(layer_to_repeat)

		tile_z = Lambda(repeat_vector, output_shape=(None, hps['z_size'])) ([feed['batch_z'], feed['decoder input']])



		decoder_full_input = Concatenate()([feed['decoder input'], tile_z])


		seq_len_1 = tf.shape(decoder_full_input)[1]
		look_ahead_mask = 1 - tf.linalg.band_part(tf.ones((seq_len_1, seq_len_1), name='new_ones'), -1, 0, name='new_mask')#create_look_ahead_mask(seq_len)
		attention_weights = {}

		decoder_embedded = input_to_embed_dense.predict(decoder_full_input)	# (batch_size, target_seq_len, self.hps['dec_size'])
		decoder_embedded_new = decoder_embedded
		decoder_embedded_new *= tf.math.sqrt(tf.cast(hps['dec_size'], tf.float32))
		decoder_embedded_new += seq2seq_model.pos_encoding[:, seq_len_1, :]

		#decoder_embedded_new = self.dropout(decoder_embedded_new, training=True)
		#decoder_output = decoder_pos_encode



		

		#x = decoder_output
		#training = True
		
		def split_heads(x, batch_size):
			depth = hps['dec_size'] // hps['num_heads']
			x = tf.reshape(x, (batch_size, -1, hps['num_heads'], depth))
			return tf.transpose(x, perm=[0, 2, 1, 3])
		
		q = [0 for _ in range(hps['num_layers'])]
		k = [0 for _ in range(hps['num_layers'])]
		v = [0 for _ in range(hps['num_layers'])]
		qs = [0 for _ in range(hps['num_layers'])]
		ks = [0 for _ in range(hps['num_layers'])]
		vs = [0 for _ in range(hps['num_layers'])]
		
		scaled_attention = [0 for _ in range(hps['num_layers'])]
		#scaled_attention1 = [0 for _ in range(self.hps['num_layers'])]
		attn_weights_block = [0 for _ in range(hps['num_layers'])]
		concat_attention = [0 for _ in range(hps['num_layers'])]
		attn = [0 for _ in range(hps['num_layers'])]
		out1 = [0 for _ in range(hps['num_layers'])]
		ffn_output = [0 for _ in range(hps['num_layers'])]
		decoder_output = [0 for _ in range(hps['num_layers']+1)]
		decoder_output[0] = decoder_embedded_new


		for j in range(hps['num_layers']):
			#attn, attn_weights_block = self.mha(decoder_output, decoder_output, decoder_output, look_ahead_mask)	# (batch_size, target_seq_len, d_model)

			batch_size = tf.shape(decoder_output[j])[0]
			

			q[j] = wq_dense[j].predict(decoder_output[j])	# (batch_size, seq_len, d_model)
			k[j] = wk_dense[j].predict(decoder_output[j])	# (batch_size, seq_len, d_model)
			v[j] = wv_dense[j].predict(decoder_output[j])	# (batch_size, seq_len, d_model)
			

			qs[j] = split_heads(q[j], batch_size)	# (batch_size, num_heads, seq_len_q, depth)
			ks[j] = split_heads(k[j], batch_size)	# (batch_size, num_heads, seq_len_k, depth)
			vs[j] = split_heads(v[j], batch_size)	# (batch_size, num_heads, seq_len_v, depth)
			
			# scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
			# attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
			scaled_attention[j], attn_weights_block[j] = scaled_dot_product_attention(qs[j], ks[j], vs[j], look_ahead_mask)
			
	
			scaled_attention[j] = tf.transpose(scaled_attention[j], perm=[0, 2, 1, 3])	# (batch_size, seq_len_q, num_heads, depth)

			concat_attention[j] = tf.reshape(scaled_attention[j], 
											(batch_size, -1, hps['dec_size']))	# (batch_size, seq_len_q, d_model)
			
			'''
			attn[i] = self.dense[i](concat_attention[i])	# (batch_size, seq_len_q, d_model)
			attn[i] = self.dropout1[i](attn[i], training=True)
			out1[i] = self.layernorm1[i](attn[i] + decoder_output[i])
			
			#ffn_output[i] = self.ffn[i](out1[i])	# (batch_size, target_seq_len, d_model)
			ffn_output[i] = self.dense1[i](out1[i])	# (batch_size, target_seq_len, d_model)
			ffn_output[i] = self.dense2[i](ffn_output[i])
			ffn_output[i] = self.dropout2[i](ffn_output[i], training=True)
			decoder_output[i+1] = self.layernorm2[i](ffn_output[i] + out1[i])	# (batch_size, target_seq_len, d_model)
			'''
			decoder_output[j+1] = concat_attention_to_decoder_output[j].predict([concat_attention[j], decoder_output[j]])
			#decoder_output, block = self.dec_layers[i](decoder_output, True, look_ahead_mask)
			

		# Apply original output layer
		model_output = output_layer_model.predict(decoder_output[hps['num_layers']])
		# Get mixture coef' based on output layer
		model_output = Lambda(seq2seq_model.get_mixture_coef)(model_output[:, -1:, :])


		##################################



		# Parse output list:
		'''
		next_state = model_outputs_list[:2]
		mixture_params_val = model_outputs_list[2:]
		[o_pi, o_mu1, o_mu2, o_sigma1, o_sigma2, o_corr, o_pen, _] = mixture_params_val
		'''

		[o_pi, o_mu1, o_mu2, o_sigma1, o_sigma2, o_corr, o_pen, _] = model_output


		idx = get_pi_idx(random.random(), o_pi[0][0], temperature, greedy_mode)

		idx_eos = get_pi_idx(random.random(), o_pen[0][0], temperature, greedy_mode)
		eos = [0, 0, 0]
		eos[idx_eos] = 1

		next_x1, next_x2 = sample_gaussian_2d(o_mu1[0][0][idx], o_mu2[0][0][idx],
											  o_sigma1[0][0][idx], o_sigma2[0][0][idx],
											  o_corr[0][0][idx], np.sqrt(temperature), greedy_mode)

		strokes[i, :] = [next_x1, next_x2, eos[0], eos[1], eos[2]]

		params = [
			o_pi[0][0], o_mu1[0][0], o_mu2[0][0], o_sigma1[0][0], o_sigma2[0][0], o_corr[0][0],
			o_pen[0][0]
		]

		mixture_params.append(params)

		'''
		prev_x = np.zeros((1, 1, 5), dtype=np.float32)
		prev_x[0][0] = np.array(strokes[i, :], dtype=np.float32)
		# prev_x[0][0] = np.array(
		#	 [next_x1, next_x2, eos[0], eos[1], eos[2]], dtype=np.float32)
		prev_state = next_state
		'''

	return strokes, mixture_params