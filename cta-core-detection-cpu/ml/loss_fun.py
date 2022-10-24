#
# Loss functions
# author: Luca Giancardo  
#

import numpy as np
from tensorflow.keras import backend as K
import tensorflow as tf
import  tensorflow.python as tfp
from sklearn.metrics import roc_auc_score, accuracy_score
from tensorflow.keras.callbacks import Callback

def euclidean_distance(vects):
    x, y = vects
    return K.sqrt(K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True), K.epsilon()))

def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return(shape1[0], 1)

def compute_accuracy(y_true, y_pred):
    pred = y_pred.ravel() < 0.5
    return np.mean(pred == y_true)

def accuracy(y_true, y_pred):
    return K.mean(K.equal(y_true, K.cast(y_pred < 0.5, y_true.dtype)))

def contrastive_loss(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    margin = 1

    # return K.mean( (1 - y_true)*K.square(y_pred) + \
    #                 y_true*K.square(K.maximum(0., margin - y_pred)) \
    #                 )
    return K.mean( (1 - y_true)*y_pred + \
                    y_true*K.maximum(0., margin - y_pred) \
                    )


def quantile_loss(q, y_true, y_pred):
    e = y_true - y_pred
    return K.mean(K.maximum(q*e, (q - 1)*e), axis=-1)

def epsilonMSE(epsilon, y_true, y_pred):
    '''
    This loss function computes epsilon-margin MSE.
    If |y_true - y_pred| < margin, then the loss is 0, otherwise the 
    '''
    absDiff = K.abs(y_true - y_pred)
#     absDiffMargin = tf.where(tf.less(tf.cast(absDiff, tf.float32), tf.cast(epsilon, tf.float32)), tf.cast(0, tf.float32), absDiff-epsilon)
    return K.mean(K.square(K.maximum(0.0, absDiff-epsilon)))




# def weighted_binary_crossentropy( pos_weight ):
#     '''
#     w1 is  the weight for the positive classes.
#     A value pos_weight > 1 decreases the false negative count, hence increasing the recall. Conversely setting pos_weight < 1 decreases the false positive count and increases the precision
#     Use like so:  model.compile(loss=weighted_binary_crossentropy(), optimizer="adam", metrics=["accuracy"])
#     '''
#     def loss(y_true, y_pred):
#         # avoid absolute 0
#         y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
#         w = K.variable(pos_weight, dtype=tf.float32)

#         res = y_true * (-K.log(y_pred)) * w + (1 - y_true) * (-K.log(1 - y_pred))

#         return res
  
#     return loss



def weighted_binary_crossentropy(zero_weight, one_weight):

    def my_loss(y_true, y_pred):

        # Original binary crossentropy (see losses.py):
        # K.mean(K.binary_crossentropy(y_true, y_pred), axis=-1)

        # Calculate the binary crossentropy
        b_ce =  tf.keras.losses.binary_crossentropy(y_true[:,1], y_pred[:,1])


        # Apply the weights
        weight_vector = y_true[:,1] * one_weight + (1 - y_true[:,1]) * zero_weight
        weighted_b_ce = weight_vector * b_ce

        # Return the mean error
        return tf.math.reduce_mean(weighted_b_ce)

        # # test 2
        # b_ce = tf.keras.losses.binary_crossentropy(y_true[:,1], y_pred[:,1])

        # return K.mean(b_ce)


    return my_loss

class RocCallback(Callback):
    def __init__(self,training_data,validation_data, weigthsFilePrefix=None, maxBatchSize=None, outputsLblLst=['out'], ignoreLblLst=[]):
        """
        training_data: full training samples
        validation_data: full validation samples
        weigthsFilePrefix: string with weights filename to be saved. It will save the weigths any time the AUC improves. Set to None to stop saving
        maxBatchSize: maximum number per batch when predicting  
        outputsLblLst: array of output labels. The number of outputs is inferred from this array
        ignoreLblLst: array of output names to ignore
        """
        self.x = training_data[0]
        self.y = training_data[1]
        self.x_val = validation_data[0]
        self.y_val = validation_data[1]
        # create dummy array if one output only
        if len(outputsLblLst)==1:
            self.y = [self.y]
            self.y_val = [self.y_val]

        self.maxBatchSize = maxBatchSize

        self.outputsLblLst=outputsLblLst
        self.nOutputs=len(outputsLblLst)

        self.ignoreLblLst = ignoreLblLst
        
        #init dictionaries
        self.val_roc_rep = {} # dictionary containing the list of ROCs values
        self.val_acc_rep = {} # dictionary containing the list of Accuracies values
        self.best_weights_acc_file = {} # dictionary containing the best weights
        self.best_weights_auc_file = {} # dictionary containing the best weights
        for lbl in outputsLblLst:
            self.val_roc_rep[lbl] = []
            self.val_acc_rep[lbl] = []
            self.best_weights_acc_file[lbl] = None
            self.best_weights_auc_file[lbl] = None

        # save weigths
        if weigthsFilePrefix==None:
            self.saveWeights = False
        else:
            self.saveWeights = True
            self.weigthsFilePrefix = weigthsFilePrefix


    def on_train_begin(self, logs={}):
        return

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def _roc_acc_comp( self, yPred, yGt, epoch, lblStr, isTraining=True ):
        roc = roc_auc_score(yGt, yPred)      
        acc = accuracy_score(yGt, yPred>0.5)      

        if isTraining:
            print('\rroc-auc_'+lblStr+'_train: %s' % str(round(roc,4)),end=100*' '+'\n')
            # log 
            tf.summary.scalar('roc-auc_'+lblStr+'_train', roc, step=epoch)
            # do not store training results
        else:
            print('\rroc-auc_'+lblStr+'_val: %s' % str(round(roc,4)),end=100*' '+'\n')
            # log
            tf.summary.scalar('roc-auc_'+lblStr+'_val', roc, step=epoch)
            self.val_roc_rep[lblStr].append(roc)
            self.val_acc_rep[lblStr].append(acc)






    def on_epoch_end(self, epoch, logs={}):        
        # Training data
        y_pred = self.model.predict(self.x, batch_size=self.maxBatchSize)
        y_pred_val = self.model.predict(self.x_val, batch_size=self.maxBatchSize)

        # creat dummy array dimension if there is one output only
        if len(self.outputsLblLst) == 1:
            y_pred = [y_pred]
            y_pred_val = [y_pred_val]

        assert(len( y_pred )== self.nOutputs)
        # compute ROCs and ACC for all outputs
        for lbl, lblId in zip(self.outputsLblLst, range(self.nOutputs)):
            if lbl in self.ignoreLblLst:
                continue # ignore unwanted labels (like the segmentation maps)

            self._roc_acc_comp( y_pred[lblId][:,1], self.y[lblId][:,1], epoch, lbl, isTraining=True )
            self._roc_acc_comp( y_pred_val[lblId][:,1], self.y_val[lblId][:,1], epoch, lbl, isTraining=False )

        # save weights
        if self.saveWeights and epoch > 1:
            for lbl in self.outputsLblLst:
                if lbl in self.ignoreLblLst:
                    continue # ignore unwanted labels (like the segmentation maps)
                # if current roc_val or acc_val is better than everything else, save
                lastAuc = self.val_roc_rep[lbl][-1]
                if  lastAuc > np.max(self.val_roc_rep[lbl][:-1]):
                    print('saving weights (AUC)')
                    fName = self.weigthsFilePrefix + '.' + str(epoch) + '.'+lbl+'_auc-' + str(round(lastAuc,4)) + '.hdf5'
                    self.model.save_weights( fName )
                    self.best_weights_auc_file[lbl] = fName

                lastAcc = self.val_acc_rep[lbl][-1]
                if  lastAcc > np.max(self.val_acc_rep[lbl][:-1]):
                    print('saving weights (ACC)')
                    fName = self.weigthsFilePrefix + '.' + str(epoch) + '.'+lbl+'_acc-' + str(round(lastAcc,4)) + '.hdf5'
                    self.model.save_weights( fName )
                    self.best_weights_acc_file[lbl] = fName
        elif self.saveWeights and epoch == 1:
            # save first epoch by default
            print('saving weights (first epoch)')
            fName = self.weigthsFilePrefix + '.' + str(epoch) + '.hdf5'
            self.model.save_weights( fName )
            self.best_weights_acc_file[lbl] = fName
            self.best_weights_auc_file[lbl] = fName



        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return

