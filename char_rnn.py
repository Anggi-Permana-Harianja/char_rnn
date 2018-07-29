'''A clean, no frills character level generative language level'''

import os
import sys
import random
import time
import utils #utils.py

import tensorflow as tf

'''write helper codes'''

def vocab_encode(text, vocab):
	return [vocab.index(x) + 1 for x in text if x in vocab]

def vocab_decode(array, vocab):
	return ''.join([vocab[x - 1] for x in array])

def read_data(filename, vocab, window, overlap):
	lines = [line.strip() for line in open(filename, 'r').readlines()]

	while True:
		random.shuffle(lines)

		for text in lines:
			#here we use vocab_encode function
			text = vocab_encode(text, vocab)
			'''create chunks as batch '''
			for start in range(0, len(text) - window, overlap):
				chunk = text[start : start + window]
				#fill the blank with [0] if overspace
				chunk += [0] * (window - len(chunk))

				yield chunk

def read_batches(stream, batch_size):
	batch = []

	for element in stream:
		batch.append(element)

		if len(batch) == batch_size:
			yield batch
			batch = []

	yield batch

'''create a RNN class '''
class CharRNN(object):
	def __init__(self, model):
		self.model = model
		self.path = 'data/' + model + '.txt'

		if 'trump' in model:
			self.vocab = ("$%'()+,-./0123456789:;=?ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                    " '\"_abcdefghijklmnopqrstuvwxyz{|}@#➡📈")
		else:
			self.vocab = (" $%'()+,-./0123456789:;=?ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                    "\\^_abcdefghijklmnopqrstuvwxyz{|}")

		#placeholder
		self.sequence = tf.placeholder(tf.int32, [None, None])
		
		self.temp = tf.constant(1.5)
		self.hidden_size = [128, 256]
		self.batch_size = 64 #this will be used in read_batches function
		self.learning_rate = 0.0003
		self.skip_step = 1
		self.number_steps = 50 #for RNN unrolled
		self.length_generated = 200

		#create a shared variable, a trainable one
		self.global_step = tf.Variable(0, dtype = tf.int32, trainable = False, name = 'global_step')

	def create_rnn(self, sequence):
		#create layers of GRU cells as big as self.hidden_size
		layers = [tf.nn.rnn_cell.GRUCell(size) for size in self.hidden_size]
		#create multi cells of GRU
		cells = tf.nn.rnn_cell.MultiRNNCell(layers)

		batch = tf.shape(sequence)[0]

		zero_states = cells.zero_state(batch, dtype = tf.float32)

		self.in_state = tuple([tf.placeholder_with_default(state, [None, state.shape[1]]) for state in zero_states])
		'''this line to calculate the real length of sequence
		   all sequence are padded to be of the same length, which is num_steps   
		'''
		length = tf.reduce_sum(tf.reduce_max(tf.sign(sequence), 2), 1)

		#run the dynamic RNN
		self.output, self.out_state = tf.nn.dynamic_rnn(cells, sequence, length, self.in_state)

	def create_model(self):
		sequence = tf.one_hot(self.sequence, len(self.vocab))
		
		#run create_rnn function
		self.create_rnn(sequence)

		#logits layer
		self.logits = tf.layers.dense(self.output, len(self.vocab), None)

		#loss function
		loss = tf.nn.softmax_cross_entropy_with_logits(logits = self.logits[ :, : -1], labels = sequence[ :, 1 : ])
		self.loss = tf.reduce_sum(loss)

		'''sample the next character from maxwell-boltzmann distribution
		   with temperature temp. it works equally well without tf.exp
		'''
		self.sample = tf.multinomial(tf.exp(self.logits[ :, -1] / self.temp), 1)[ :, 0]
		self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss, global_step = self.global_step)

	def train(self):
		saver = tf.train.Saver()
		start = time.time()
		minimal_loss = None

		#start session
		with tf.Session() as sess:
			#run tensorboard
			writer = tf.summary.FileWriter('graphs/gist', sess.graph)
			#start variable initializer
			sess.run(tf.global_variables_initializer())

			#checkpoint
			checkpoint = tf.train.get_checkpoint_state(os.path.dirname('checkpoints/' + self.model + '/checkpoint'))
			if checkpoint and checkpoint.model_checkpoint_path:
				saver.restore(sess, checkpoint.model_checkpoint_path)

			iteration = self.global_step.eval()
			#run read_data to get batch
			stream = read_data(self.path, self.vocab, self.number_steps, overlap = self.number_steps // 2)
			data = read_batches(stream, self.batch_size)

			'''read batch '''
			while True:
				#slide the window batch
				batch = next(data)
				'''get the loss '''
				batch_loss, _ = sess.run([self.loss, self.optimizer], {self.sequence: batch})

				if (iteration + 1) % self.skip_step == 0:
					print('iteration {} loss {} time {}'.format(iteration, batch_loss, time.time() - start))

					#generate one character based on online inference model
					self.online_inference(sess)

					start = time.time()
					checkpoint_name = 'checkpoints' + self.model + '/char-rnn'

					if minimal_loss is None:
						saver.save(sess, checkpoint_name, iteration)
					else:
						saver.save(sess, checkpoint_name, iteration)
						minimal_loss = batch_loss

				iteration += 1

	def online_inference(self, sess):
		'''generate sequence one character at a time, based on the previous character '''
		for seed in ['Hillary', 'I', 'R', 'T', '@', 'N', 'M', '.', 'G', 'A', 'W']:
			sentence = seed
			state = None

			for _ in range(self.length_generated):
				batch = [vocab_encode(sentence[-1], self.vocab)]
				feed = {self.sequence: batch}

				#for the first decoder step, the state is None
				if state is not None:
					for i in range(len(state)):
						feed.update({self.in_state[i]: state[i]})
				#get next state
				index, state = sess.run([self.sample, self.out_state], feed)
				sentence += vocab_decode(index, self.vocab)

			print('\t' + sentence)

'''run the main program '''
def main():
	#create model 
	model = 'trump_tweets'
	utils.safe_mkdir('checkpoints')
	utils.safe_mkdir('checkpoints/' + model)

	#create RNN object
	used_RNN = CharRNN(model)
	used_RNN.create_model()
	used_RNN.train()

if __name__ == '__main__':
	main()

