__author__ = 'Frederic Godin  (frederic.godin@ugent.be / www.fredericgodin.com)'
import theano
import theano.tensor as T
import numpy
import lasagne
import argparse

import DCNN
import dataUtils
import networks
import utils



parser = argparse.ArgumentParser(description='Train a DCNN on the binary Stanford Sentiment dataset as specified in the Kalchbrenner \'14 paper. All the default values are taken from the paper or the Matlab code.')
# training settings
parser.add_argument("--learning_rate",type=float, default=0.1, help='Learning rate')
parser.add_argument("--n_epochs",type=int,default=500,help="Number of epochs")
parser.add_argument("--valid_freq",type=int,default=10,help="Number of batches processed until we validate.")
parser.add_argument("--adagrad_reset",type=int,default=5,help="Resets the adagrad cumulative gradient after x epochs. If the value is 0, no reset will be executed.")
# input output
parser.add_argument("--vocab_size",type=int, default=15448, help='Vocabulary size')
parser.add_argument("--output_classes",type=int, default=2, help='Number of output classes')
parser.add_argument("--batch_size",type=int, default=4, help='Batch size')
# network paras
parser.add_argument("--word_vector_size",type=int, default=48, help='Word vector size')
parser.add_argument("--filter_size_conv_layers", nargs="+", type=int, default=[7,5],help="List of sizes of filters at layer 1 and 2, default=[10,7]")
parser.add_argument("--nr_of_filters_conv_layers", nargs="+", type=int, default=[6,14],help="List of number of filters at layer 1 and 2, default=[6,12]")
parser.add_argument("--activations",nargs='+', type=str,default=["tanh","tanh"],help="List of activation functions behind first and second conv layers, default [tanh, tanh]. Possible values are \"linear\", \"tanh\", \"rectify\" and \"sigmoid\". ")
parser.add_argument("--L2",nargs='+',type=float,default=[0.0001/2,0.00003/2,0.000003/2,0.0001/2],help="Fine-grained L2 regularization. 4 values are needed for 4 layers, namly for the embeddings layer, 2 conv layers and a final/output dense layer.")
parser.add_argument("--ktop",type=int,default=4,help="K value of top pooling layer DCNN")
parser.add_argument("--dropout_value", type=float,default=0.5,help="Dropout value after penultimate layer")

args = parser.parse_args()
hyperparas = vars(args)
print("Hyperparameters: "+str(hyperparas))

if len(hyperparas['filter_size_conv_layers'])!= 2 or len(hyperparas['nr_of_filters_conv_layers'])!=2 or len(hyperparas['activations'])!=2 or len(hyperparas["L2"])!=4 :
    raise Exception('Check if the input --filter_size_conv_layers, --nr_of_filters_conv_layers and --activations are lists of size 2, and the --L2 field needs a value list of 4 values.')


#######################
# LOAD  TRAINING DATA #
#######################
print('Loading the training data')

# load data, taken from Kalchbrenner matlab files
# we order the input according to length and pad all sentences until the maximum length
# at training time however, we will use the "length" array to shrink that matrix following the largest sentence within a batch
# in practice, this means that batches are padded with 1 or 2, or aren't even padded at all.
kalchbrenner_path = "./data/binarySentiment/"
new_train_x_indexes, new_train_y, train_lengths = dataUtils.read_and_sort_matlab_data(kalchbrenner_path+"train.txt",kalchbrenner_path+"train_lbl.txt")
new_dev_x_indexes, new_dev_y, dev_lengths = dataUtils.read_and_sort_matlab_data(kalchbrenner_path+"valid.txt",kalchbrenner_path+"valid_lbl.txt")
new_test_x_indexes, new_test_y, test_lengths = dataUtils.read_and_sort_matlab_data(kalchbrenner_path+"test.txt",kalchbrenner_path+"test_lbl.txt")

# train data on GPU
train_indexes_shared = T.cast(theano.shared(new_train_x_indexes,borrow=True), 'int32')
train_labels_shared = T.cast(theano.shared(new_train_y,borrow=True), 'int32')
n_train_batches = len(train_lengths) / hyperparas['batch_size']

#dev data on GPU
new_dev_x_indexes_extended = dataUtils.pad_to_batch_size(new_dev_x_indexes,hyperparas['batch_size'])
dev_indexes_shared = T.cast(theano.shared(new_dev_x_indexes_extended,borrow=True), 'int32')
new_dev_y_extended = dataUtils.pad_to_batch_size(new_dev_y,hyperparas['batch_size'])
dev_labels_shared = T.cast(theano.shared(new_dev_y_extended,borrow=True), 'int32')
n_dev_batches = new_dev_x_indexes_extended.shape[0] / hyperparas['batch_size']
n_dev_samples = len(new_dev_y)
dataUtils.extend_lenghts(dev_lengths,hyperparas['batch_size'])

# test data on GPU
new_test_x_indexes_extended = dataUtils.pad_to_batch_size(new_test_x_indexes,hyperparas['batch_size'])
test_indexes_shared = T.cast(theano.shared(new_test_x_indexes_extended,borrow=True), 'int32')
new_test_y_extended = dataUtils.pad_to_batch_size(new_test_y,hyperparas['batch_size'])
test_labels_shared = T.cast(theano.shared(new_test_y_extended,borrow=True), 'int32')
n_test_batches = new_test_x_indexes_extended.shape[0] / hyperparas['batch_size']
n_test_samples = len(new_test_y)
dataUtils.extend_lenghts(test_lengths,hyperparas['batch_size'])

######################
# BUILD ACTUAL MODEL #
######################
print('Building the model')

# allocate symbolic variables for the data
indexes = T.ivector('indexes')  # index to a [mini]batch
length = T.iscalar('length')
X_batch = T.imatrix('x')
y_batch = T.ivector('y')

# define/load the network
output_layer = networks.buildDCNNPaper(batch_size=hyperparas['batch_size'],vocab_size=hyperparas['vocab_size'],embeddings_size=hyperparas['word_vector_size'],filter_sizes=hyperparas['filter_size_conv_layers'],nr_of_filters=hyperparas['nr_of_filters_conv_layers'],activations=hyperparas['activations'],ktop=hyperparas['ktop'],dropout=hyperparas["dropout_value"],output_classes=hyperparas['output_classes'],padding='last')

# Kalchbrenner uses a fine-grained L2 regularization in the Matlab code, default values taken from Matlab code
# Training objective
l2_layers = []
for layer in lasagne.layers.get_all_layers(output_layer):
    if isinstance(layer,(DCNN.embeddings.SentenceEmbeddingLayer,DCNN.convolutions.Conv1DLayerSplitted,lasagne.layers.DenseLayer)):
        l2_layers.append(layer)
loss_train = lasagne.objectives.aggregate(lasagne.objectives.categorical_crossentropy(lasagne.layers.get_output(output_layer,X_batch),y_batch),mode='mean')+lasagne.regularization.regularize_layer_params_weighted(dict(zip(l2_layers,hyperparas["L2"])),lasagne.regularization.l2)

# validating/testing
loss_eval = lasagne.objectives.categorical_crossentropy(lasagne.layers.get_output(output_layer,X_batch,deterministic=True),y_batch)
pred = T.argmax(lasagne.layers.get_output(output_layer, X_batch, deterministic=True),axis=1)
correct_predictions = T.eq(pred, y_batch)

# In the matlab code, Kalchbrenner works with a adagrad reset mechanism, if the para --adagrad_reset has value 0, no reset will be applied
all_params = lasagne.layers.get_all_params(output_layer)
updates, accumulated_grads = utils.adagrad(loss_train, all_params, hyperparas['learning_rate'])
#updates = lasagne.updates.adagrad(loss_train, all_params, hyperparas['learning_rate'])


train_model = theano.function(inputs=[indexes,length], outputs=loss_train,
        updates=updates,
        givens={
            X_batch: train_indexes_shared[indexes,0:length],
            y_batch: train_labels_shared[indexes]
        })

valid_model = theano.function(inputs=[indexes,length], outputs=correct_predictions,
        givens={
            X_batch: dev_indexes_shared[indexes,0:length],
            y_batch: dev_labels_shared[indexes]})

test_model = theano.function(inputs=[indexes,length], outputs=correct_predictions,
        givens={
            X_batch: test_indexes_shared[indexes,0:length],
            y_batch: test_labels_shared[indexes]})


###############
# TRAIN MODEL #
###############
print('Started training')
print('Because of the default high validation frequency, only improvements are printed.')

train_error = 0
best_validation_accuracy = 0
epoch = 0
while (epoch < hyperparas['n_epochs']):
    epoch = epoch + 1
    permutation = numpy.random.permutation(n_train_batches)
    batch_counter = 0
    train_accuracy=0
    for minibatch_index in permutation:
        train_accuracy += train_model(range(minibatch_index*hyperparas['batch_size'],(minibatch_index+1)*hyperparas['batch_size']),train_lengths[(minibatch_index+1)*hyperparas['batch_size']-1])

        if batch_counter>0 and batch_counter % hyperparas["valid_freq"] == 0:
            accuracy_valid=[]
            for minibatch_dev_index in range(n_dev_batches):
                accuracy_valid.append(valid_model(range(minibatch_dev_index*hyperparas['batch_size'],(minibatch_dev_index+1)*hyperparas['batch_size']),dev_lengths[(minibatch_dev_index+1)*hyperparas['batch_size']-1]))

            #dirty code to correctly asses validation accuracy, last results in the array are predictions for the padding rows
            this_validation_accuracy = numpy.concatenate(accuracy_valid)[0:n_dev_samples].sum()/float(n_dev_samples)

            if this_validation_accuracy > best_validation_accuracy:
                print("Train loss, "+str( (train_accuracy/hyperparas["valid_freq"]))+", validation accuracy: "+str(this_validation_accuracy)+"%")
                best_validation_accuracy = this_validation_accuracy

                # test it
                accuracy_test= []
                for minibatch_test_index in range(n_test_batches):
                    accuracy_test.append(test_model(range(minibatch_test_index*hyperparas['batch_size'],(minibatch_test_index+1)*hyperparas['batch_size']),test_lengths[(minibatch_test_index+1)*hyperparas['batch_size']-1]))
                this_test_accuracy = numpy.concatenate(accuracy_test)[0:n_test_samples].sum()/float(n_test_samples)
                print("Test accuracy: "+str(this_test_accuracy)+"%")

            train_accuracy=0
        batch_counter+=1

    if hyperparas["adagrad_reset"] > 0:
        if epoch % hyperparas["adagrad_reset"] == 0:
            utils.reset_grads(accumulated_grads)

    print("Epoch "+str(epoch)+" finished.")



