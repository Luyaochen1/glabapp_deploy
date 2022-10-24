#
# Network architectures definitions
# author: Luca Giancardo  
#

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras

# pylint: disable=import-error
from tensorflow.keras.models import Model
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, MaxPooling3D, Conv3D, BatchNormalization, Flatten, Lambda, Dropout, Dense, Concatenate, Multiply, Add, InputLayer, LayerNormalization
from tensorflow.keras.layers.experimental import SyncBatchNormalization
from tensorflow.keras import backend as K

# custom layer
from tensorflow.python.keras.layers.pooling import Pooling3D
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.nn_ops import _get_sequence

from skimage.transform import resize


#=== Deafult parameters
depth = 4 
activation = 'relu' 
nFilters = 16 # # filters in 1st hidden layer
#==


class NormLayer(layers.Layer):

  def __init__(self, name,  **kwargs):
    super(NormLayer, self).__init__(name=name,  **kwargs)

    self.minVar = tf.Variable(0.001,trainable=True, name='minVar')
    self.maxVar = tf.Variable(1.,trainable=True, name='maxVar')

  def call(self, inputs):
    res = tf.clip_by_value(inputs,self.minVar,self.maxVar)
    res = (res - self.minVar) / (self.maxVar-self.minVar)
    res = tf.clip_by_value(res,0,1)

    # tf.summary.scalar('minVar', self.minVar)
    # tf.summary.scalar('maxVar', self.maxVar)

    return res




def leNet( input_layer, \
        depth=depth, \
        activation=activation, \
        nFilters=nFilters):
    
    x = input_layer
    for _ in range(depth):
        # Inception module
        x = Conv3D(nFilters, kernel_size=(3, 3, 3), padding='same', activation=activation)(x)
        x = MaxPooling3D((2, 2, 2), strides=(2, 2, 2), padding='same')(x)


    return x

def leNetWithInputLayer(input_shape, **args):
    inputs = Input(shape=input_shape)

    outputs = leNet(inputs, **args)

    return Model(inputs, outputs)

    

def sNetAutoPreprocessingLeNet(input_shape,
                            depthBefore=4,
                            depthAfter=4,
                            activation='relu',
                            n_classes=3,
                            nFilters=32,
                            globAvgPool=False,
                            addDenseLayerNeurons=15):


    base_network = leNetWithInputLayer(input_shape,
                                 depth=depthBefore,
                                 activation=activation,
                                 nFilters=nFilters)

    input_left = Input(shape=(input_shape))
    input_right = Input(shape=(input_shape))
    # input_left = InputLayer(input_shape=(input_shape))
    # input_right = InputLayer(shape=(input_shape))

    nL = NormLayer(name='normLayer1')

    # Re-use same instance of base network (weights are shared across both branches)
    processed_left = base_network(nL(input_left))
    processed_right = base_network(nL(input_right))
    # processed_left = base_network(input_left)
    # processed_right = base_network(input_right)


    # Merging features from two hemispheres
    # l1
    l1_distance_layer = Lambda(lambda tensors: K.abs(tensors[0] - tensors[1]))
    merged_layer = l1_distance_layer([processed_left, processed_right])

    # Inception modules after merge
    x = leNet(merged_layer,
            depth=depthAfter,
            activation=activation,
            nFilters=nFilters)

    if globAvgPool:
        x = layers.GlobalAveragePooling3D()(x)

    # Flatten
    x = Flatten()(x)

    # additional dense layer
    if addDenseLayerNeurons > 0:
        x = Dense(units=addDenseLayerNeurons, activation='tanh')(x)


    # Prediction layer
    prediction = Dense(units=n_classes, activation='softmax')(x)


    return Model(inputs=[input_left, input_right], outputs=prediction)


def sNetAutoPreprocessingLeNetSiamese(input_shape,
                            depthNet=4,
                            activation='relu',
                            n_classes=3,
                            nFilters=32,
                            globAvgPool=False,
                            addDenseLayerNeurons=15):

    #=== create shared net
    inputShared = Input(shape=input_shape)
    xShared = leNet( inputShared,
                    depth=depthNet,
                     activation=activation,
                     nFilters=nFilters)

    if globAvgPool:
        xShared = layers.GlobalAveragePooling3D()(xShared)

    # Flatten
    xShared = Flatten()(xShared)

    # additional dense layer
    if addDenseLayerNeurons > 0:
        xShared = Dense(units=addDenseLayerNeurons, activation='tanh')(xShared)
    # model
    sharedMod = Model(inputShared, xShared)
    #===


    input_left = Input(shape=(input_shape))
    input_right = Input(shape=(input_shape))


    nL = NormLayer(name='normLayer1')

    # Re-use same instance of base network (weights are shared across both branches)
    processed_left = sharedMod(nL(input_left))
    processed_right = sharedMod(nL(input_right))


    # Merging features from two hemispheres
    L1_layer = Lambda(lambda tensors:K.abs(tensors[0] - tensors[1]))
    L1_distance = L1_layer([processed_left, processed_right])
    
    # Add a dense layer with a sigmoid unit to generate the similarity score
    prediction = Dense(n_classes,activation='softmax')(L1_distance)


    return Model(inputs=[input_left, input_right], outputs=prediction)


def sNetAutoPreprocessingLeNetSiameseDistOut(input_shape,
                            depthNet=4,
                            activation='relu',
                            n_classes=3,
                            nFilters=32,
                            globAvgPool=False,
                            addDenseLayerNeurons=15):

    #=== create shared net
    inputShared = Input(shape=input_shape)
    xShared = leNet( inputShared,
                    depth=depthNet,
                     activation=activation,
                     nFilters=nFilters)

    if globAvgPool:
        xShared = layers.GlobalAveragePooling3D()(xShared)

    # Flatten
    xShared = Flatten()(xShared)

    # additional dense layer
    if addDenseLayerNeurons > 0:
        xShared = Dense(units=addDenseLayerNeurons, activation='tanh')(xShared)
    # model
    sharedMod = Model(inputShared, xShared)
    #===


    input_left = Input(shape=(input_shape))
    input_right = Input(shape=(input_shape))


    nL = NormLayer(name='normLayer1')

    # Re-use same instance of base network (weights are shared across both branches)
    processed_left = sharedMod(nL(input_left))
    processed_right = sharedMod(nL(input_right))


    def curr_distance(tensors):
        return K.sum(K.abs(tensors[0] - tensors[1]), axis=1, keepdims=True)
        # return K.sqrt(K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True), K.epsilon()))

    def curr_distance_output_shape(shapes):
        shape1, shape2 = shapes
        return(shape1[0], 1)

    # # Merging features from two hemispheres
    # L1_layer_red = Lambda(lambda tensors: K.sum(K.abs(tensors[0] - tensors[1])), output_shape=eucl_dist_output_shape)
    # L1_distance = L1_layer_red([processed_left, processed_right])

    
    # # Output similarity score directly
    # prediction = L1_distance


    # Flatten
    processed_left = Flatten()(processed_left)
    processed_right = Flatten()(processed_right)

    distance = Lambda(curr_distance, output_shape=curr_distance_output_shape)([processed_left, processed_right])
    model = Model(inputs=[input_left, input_right], outputs=distance)
    return model

    # return Model(inputs=[input_left, input_right], outputs=prediction)

def sNetAutoPreprocessingLeNetWithSkipConn(input_shape,
                                        depthBefore=4,
                                        depthAfter=4,
                                        activation='relu',
                                        n_classes=3,
                                        nFilters=32,
                                        globAvgPool=False,
                                        addDenseLayerNeurons=15):


    base_network = leNetWithInputLayer(input_shape,
                                 depth=depthBefore,
                                 activation=activation,
                                 nFilters=nFilters)

    input_left = Input(shape=(input_shape))
    input_right = Input(shape=(input_shape))
    # input_left = InputLayer(input_shape=(input_shape))
    # input_right = InputLayer(shape=(input_shape))

    nL = NormLayer(name='normLayer1')

    # Re-use same instance of base network (weights are shared across both branches)
    processed_left = base_network(nL(input_left))
    processed_right = base_network(nL(input_right))



    # Merging features from two hemispheres
    # l1
    l1_distance_layer = Lambda(lambda tensors: K.abs(tensors[0] - tensors[1]))
    merged_layer = l1_distance_layer([processed_left, processed_right])

    #==== Inception modules after merge
    # symmetric branch
    x = leNet(merged_layer,
            depth=depthAfter,
            activation=activation,
            nFilters=nFilters)
    # non symmetric branches (shared weights between them)
    nonSymNet = leNetWithInputLayer(processed_left.shape[1:],
                                depth=depthAfter,
                                activation=activation,
                                nFilters=nFilters)
    xLeft = nonSymNet(processed_left)
    xRight = nonSymNet(processed_right)

    # concat
    x = Concatenate(axis=-1)([x, xLeft, xRight])

    if globAvgPool:
        x = layers.GlobalAveragePooling3D()(x)


     #====
    # Flatten
    x = Flatten()(x)

    # additional dense layer
    if addDenseLayerNeurons > 0:
        x = Dense(units=addDenseLayerNeurons, activation='tanh')(x)


    # Prediction layer
    prediction = Dense(units=n_classes, activation='softmax')(x)


    return Model(inputs=[input_left, input_right], outputs=prediction)



def vggNet( input_layer, \
        depth=depth, \
        activation=activation, \
        nFilters=nFilters, \
        nConv=2, 
        maxPool=True,
        layerNorm=False, 
        batchNorm=False,
        batchNormSync=False,
        usePca=False,
        pcaComp=5):
    
    pcaLayer = PCALayer(pcaComp)

    x = input_layer
    for _ in range(depth):
        for _ in range(nConv):
            x = Conv3D(nFilters, kernel_size=(3, 3, 3), padding='same', activation=activation)(x)

        if usePca:
            x = pcaLayer(x)


        if maxPool:
            x = MaxPooling3D((2, 2, 2), strides=(2, 2, 2), padding='same')(x)


        if layerNorm:
            x = LayerNormalization()(x)

        if batchNorm:
            x = BatchNormalization()(x)

        if batchNormSync:
            x = SyncBatchNormalization()(x)

    return x

def vggNetWithInputLayer(input_shape, **args):
    inputs = Input(shape=input_shape)

    outputs = vggNet(inputs, **args)

    return Model(inputs, outputs)



def sNetAutoPreprocessingVggNetWithSkipConn(input_shape,
                                        depthBefore=4,
                                        depthAfter=4,
                                        activation='relu',
                                        n_classes=3,
                                        nFilters=32,
                                        nConv=3,
                                        globAvgPool=False,
                                        addDenseLayerNeurons=15):


    base_network = vggNetWithInputLayer(input_shape,
                                 depth=depthBefore,
                                 activation=activation,
                                 nFilters=nFilters,
                                 nConv=nConv)

    input_left = Input(shape=(input_shape))
    input_right = Input(shape=(input_shape))
    # input_left = InputLayer(input_shape=(input_shape))
    # input_right = InputLayer(shape=(input_shape))

    nL = NormLayer(name='normLayer1')

    # Re-use same instance of base network (weights are shared across both branches)
    processed_left = base_network(nL(input_left))
    processed_right = base_network(nL(input_right))


    # Merging features from two hemispheres
    # l1
    l1_distance_layer = Lambda(lambda tensors: K.abs(tensors[0] - tensors[1]))
    merged_layer = l1_distance_layer([processed_left, processed_right])

    #==== Inception modules after merge
    # symmetric branch
    x = vggNet(merged_layer,
            depth=depthAfter,
            activation=activation,
            nFilters=nFilters,
            nConv=nConv)
    # non symmetric branches (shared weights between them)
    nonSymNet = vggNetWithInputLayer(processed_left.shape[1:],
                                depth=depthAfter,
                                activation=activation,
                                nFilters=nFilters,
                                nConv=nConv)
    xLeft = nonSymNet(processed_left)
    xRight = nonSymNet(processed_right)

    # concat
    x = Concatenate(axis=-1)([x, xLeft, xRight])

    if globAvgPool:
        x = layers.GlobalAveragePooling3D()(x)


     #====
    # Flatten
    x = Flatten()(x)

    # additional dense layer
    if addDenseLayerNeurons > 0:
        x = Dense(units=addDenseLayerNeurons, activation='tanh')(x)


    # Prediction layer
    prediction = Dense(units=n_classes, activation='softmax')(x)


    return Model(inputs=[input_left, input_right], outputs=prediction)


def sNetAutoPreprocessingVggNetWithSkipConnBN(input_shape,
                                        depthBefore=4,
                                        depthAfter=4,
                                        activation='relu',
                                        n_classes=3,
                                        nFilters=32,
                                        nConv=3,
                                        globAvgPool=False,
                                        addDenseLayerNeurons=15):
    """
    Add batch normalization
    """
    base_network = vggNetWithInputLayer(input_shape,
                                 depth=depthBefore,
                                 activation=activation,
                                 nFilters=nFilters,
                                 nConv=nConv)

    input_left = Input(shape=(input_shape))
    input_right = Input(shape=(input_shape))
    # input_left = InputLayer(input_shape=(input_shape))
    # input_right = InputLayer(shape=(input_shape))

    nL = NormLayer(name='normLayer1')

    # Re-use same instance of base network (weights are shared across both branches)
    processed_left = base_network(nL(input_left))
    processed_right = base_network(nL(input_right))


    # Merging features from two hemispheres
    # l1
    l1_distance_layer = Lambda(lambda tensors: K.abs(tensors[0] - tensors[1]))
    merged_layer = l1_distance_layer([processed_left, processed_right])

    #==== Inception modules after merge
    # symmetric branch
    x = vggNet(merged_layer,
            depth=depthAfter,
            activation=activation,
            nFilters=nFilters,
            nConv=nConv)
    # non symmetric branches (shared weights between them)
    nonSymNet = vggNetWithInputLayer(processed_left.shape[1:],
                                depth=depthAfter,
                                activation=activation,
                                nFilters=nFilters,
                                nConv=nConv)
    xLeft = nonSymNet(processed_left)
    xRight = nonSymNet(processed_right)

    # BN
    xLeft = BatchNormalization()(xLeft)
    xRight = BatchNormalization()(xRight)
    x = BatchNormalization()(x)

    # concat
    x = Concatenate(axis=-1)([x, xLeft, xRight])

    if globAvgPool:
        x = layers.GlobalAveragePooling3D()(x)


     #====
    # Flatten
    x = Flatten()(x)

    # additional dense layer
    if addDenseLayerNeurons > 0:
        x = Dense(units=addDenseLayerNeurons, activation='tanh')(x)


    # Prediction layer
    prediction = Dense(units=n_classes, activation='softmax')(x)


    # # Prediction layer
    # y_LVO = Dense(units=1, activation='sigmoid', name='LVO')(merged_layer_LVO)
    # y_infarct = Dense(units=1, activation='sigmoid', name='infarctVolume')(merged_layer_infarctVol)
    # y_penumbra = Dense(units=1, activation='sigmoid', name='penumbraVolume')(merged_layer_penumbraVol)

    # return Model(inputs=[input_left, input_right], outputs=[y_LVO, y_infarct, y_penumbra])

    return Model(inputs=[input_left, input_right], outputs=prediction)





def sNetAutoPreprocessingVggNetWithSkipConnBNmt(input_shape,
                                        depthBefore=4,
                                        depthAfter=4,
                                        activation='relu',
                                        n_classes=2,
                                        nFilters=32,
                                        nConv=3,
                                        globAvgPool=False,
                                        addDenseLayerNeurons=15,
                                        tasksLst=['core','pen']):
    """
    Add Multitask
    """
    base_network = vggNetWithInputLayer(input_shape,
                                 depth=depthBefore,
                                 activation=activation,
                                 nFilters=nFilters,
                                 nConv=nConv)

    input_left = Input(shape=(input_shape))
    input_right = Input(shape=(input_shape))
    # input_left = InputLayer(input_shape=(input_shape))
    # input_right = InputLayer(shape=(input_shape))

    nL = NormLayer(name='normLayer1')

    # Re-use same instance of base network (weights are shared across both branches)
    processed_left = base_network(nL(input_left))
    processed_right = base_network(nL(input_right))


    # Merging features from two hemispheres
    # l1
    l1_distance_layer = Lambda(lambda tensors: K.abs(tensors[0] - tensors[1]))
    merged_layer = l1_distance_layer([processed_left, processed_right])

    #==== Inception modules after merge
    # symmetric branch
    x = vggNet(merged_layer,
            depth=depthAfter,
            activation=activation,
            nFilters=nFilters,
            nConv=nConv)
    # non symmetric branches (shared weights between them)
    nonSymNet = vggNetWithInputLayer(processed_left.shape[1:],
                                depth=depthAfter,
                                activation=activation,
                                nFilters=nFilters,
                                nConv=nConv)
    xLeft = nonSymNet(processed_left)
    xRight = nonSymNet(processed_right)

    # BN
    xLeft = BatchNormalization()(xLeft)
    xRight = BatchNormalization()(xRight)
    x = BatchNormalization()(x)

    # concat
    x = Concatenate(axis=-1)([x, xLeft, xRight])

    if globAvgPool:
        x = layers.GlobalAveragePooling3D()(x)


     #====
    # Flatten
    x = Flatten()(x)

    # prediction output lst
    predOutLst = [None] * len(tasksLst)
    for tID in range(len(tasksLst)):

        # additional dense layer
        if addDenseLayerNeurons > 0:
            tmpLayer = Dense(units=addDenseLayerNeurons, activation='tanh',name=tasksLst[tID]+'_dense')(x)
            # Prediction layer
            predOutLst[tID] = Dense(units=n_classes, activation='softmax', name=tasksLst[tID]+'_output')(tmpLayer)

    return Model(inputs=[input_left, input_right], outputs=predOutLst)




def sNetVggNetWithSkipConnLN(input_shape,
                            depthBefore=4,
                            depthAfter=4,
                            activation='relu',
                            n_classes=2,
                            nFilters=32,
                            nConv=3,
                            globAvgPool=False,
                            addDenseLayerNeurons=15,
                            tasksLst=['core','pen']):
    """
    Add Multitask
    """
    base_network = vggNetWithInputLayer(input_shape,
                                 depth=depthBefore,
                                 activation=activation,
                                 nFilters=nFilters,
                                 nConv=nConv,
                                 layerNorm=True )

    input_left = Input(shape=(input_shape))
    input_right = Input(shape=(input_shape))
    # input_left = InputLayer(input_shape=(input_shape))
    # input_right = InputLayer(shape=(input_shape))



    # Re-use same instance of base network (weights are shared across both branches)
    processed_left = base_network(input_left)
    processed_right = base_network(input_right)


    # Merging features from two hemispheres
    # l1
    l1_distance_layer = Lambda(lambda tensors: K.abs(tensors[0] - tensors[1]))
    merged_layer = l1_distance_layer([processed_left, processed_right])

    #==== Inception modules after merge
    # symmetric branch
    x = vggNet(merged_layer,
            depth=depthAfter,
            activation=activation,
            nFilters=nFilters,
            nConv=nConv,
            layerNorm=True)
    # non symmetric branches (shared weights between them)
    nonSymNet = vggNetWithInputLayer(processed_left.shape[1:],
                                depth=depthAfter,
                                activation=activation,
                                nFilters=nFilters,
                                nConv=nConv,
                                layerNorm=True)
    xLeft = nonSymNet(processed_left)
    xRight = nonSymNet(processed_right)

    # BN
    xLeft =  LayerNormalization ()(xLeft)
    xRight = LayerNormalization ()(xRight)
    x = LayerNormalization ()(x)

    # concat
    x = Concatenate(axis=-1)([x, xLeft, xRight])

    if globAvgPool:
        x = layers.GlobalAveragePooling3D()(x)


     #====
    # Flatten
    x = Flatten()(x)

    # prediction output lst
    predOutLst = [None] * len(tasksLst)
    for tID in range(len(tasksLst)):

        # additional dense layer
        if addDenseLayerNeurons > 0:
            tmpLayer = Dense(units=addDenseLayerNeurons, activation='tanh',name=tasksLst[tID]+'_dense')(x)
            # Prediction layer
            predOutLst[tID] = Dense(units=n_classes, activation='softmax', name=tasksLst[tID]+'_output')(tmpLayer)

    return Model(inputs=[input_left, input_right], outputs=predOutLst)



def sNetVggNetWithSkipConnBNs(input_shape,
                            depthBefore=4,
                            depthAfter=4,
                            activation='relu',
                            n_classes=2,
                            nFilters=32,
                            nConv=3,
                            globAvgPool=False,
                            addDenseLayerNeurons=15,
                            tasksLst=['core','pen']):
    """
    Add Multitask
    """
    base_network = vggNetWithInputLayer(input_shape,
                                 depth=depthBefore,
                                 activation=activation,
                                 nFilters=nFilters,
                                 nConv=nConv,
                                 batchNormSync=True )

    input_left = Input(shape=(input_shape))
    input_right = Input(shape=(input_shape))
    # input_left = InputLayer(input_shape=(input_shape))
    # input_right = InputLayer(shape=(input_shape))



    # Re-use same instance of base network (weights are shared across both branches)
    processed_left = base_network(input_left)
    processed_right = base_network(input_right)


    # Merging features from two hemispheres
    # l1
    l1_distance_layer = Lambda(lambda tensors: K.abs(tensors[0] - tensors[1]))
    merged_layer = l1_distance_layer([processed_left, processed_right])

    #==== Inception modules after merge
    # symmetric branch
    x = vggNet(merged_layer,
            depth=depthAfter,
            activation=activation,
            nFilters=nFilters,
            nConv=nConv,
            batchNormSync=True)
    # non symmetric branches (shared weights between them)
    nonSymNet = vggNetWithInputLayer(processed_left.shape[1:],
                                depth=depthAfter,
                                activation=activation,
                                nFilters=nFilters,
                                nConv=nConv,
                                batchNormSync=True)
    xLeft = nonSymNet(processed_left)
    xRight = nonSymNet(processed_right)

    # BN
    xLeft = SyncBatchNormalization()(xLeft)
    xRight = SyncBatchNormalization()(xRight)
    x = SyncBatchNormalization()(x)

    # # convert dtypes to allow mixed precision
    # cast32Layer = Lambda(lambda tensors: tf.dtypes.cast(tensors, tf.float32) )
    # x = cast32Layer(x)
    # xLeft=cast32Layer(xLeft)
    # xRight=cast32Layer(xRight)

    # # concat
    # x = Concatenate(axis=-1, dtype='float32')([x, xLeft, xRight])

    if globAvgPool:
        x = layers.GlobalAveragePooling3D()(x)


     #====
    # Flatten
    x = Flatten()(x)

    # prediction output lst
    predOutLst = [None] * len(tasksLst)
    for tID in range(len(tasksLst)):

        # additional dense layer
        if addDenseLayerNeurons > 0:
            tmpLayer = Dense(units=addDenseLayerNeurons, activation='tanh',name=tasksLst[tID]+'_dense')(x)
            # Prediction layer
            predOutLst[tID] = Dense(units=n_classes, activation='softmax', name=tasksLst[tID]+'_output')(tmpLayer)

    return Model(inputs=[input_left, input_right], outputs=predOutLst)


def sNetVggNetWithSkipConnBNsOrd(input_shape,
                            depthBefore=4,
                            depthAfter=4,
                            activation='relu',
                            nFilters=32,
                            nConv=3,
                            globAvgPool=False,
                            addDenseLayerNeurons=15,
                            tasksLst=['core','pen'],
                            mergeLeft=False):
    """
    Add Multitask
    """
    base_network = vggNetWithInputLayer(input_shape,
                                 depth=depthBefore,
                                 activation=activation,
                                 nFilters=nFilters,
                                 nConv=nConv,
                                 batchNormSync=True )

    input_left = Input(shape=(input_shape))
    input_right = Input(shape=(input_shape))
    # input_left = InputLayer(input_shape=(input_shape))
    # input_right = InputLayer(shape=(input_shape))



    # Re-use same instance of base network (weights are shared across both branches)
    processed_left = base_network(input_left)
    processed_right = base_network(input_right)


    # Merging features from two hemispheres
    # l1
    l1_distance_layer = Lambda(lambda tensors: K.abs(tensors[0] - tensors[1]))
    merged_layer = l1_distance_layer([processed_left, processed_right])

    #==== Inception modules after merge
    # symmetric branch
    x = vggNet(merged_layer,
            depth=depthAfter,
            activation=activation,
            nFilters=nFilters,
            nConv=nConv,
            batchNormSync=True)
    # non symmetric branches (shared weights between them)
    nonSymNet = vggNetWithInputLayer(processed_left.shape[1:],
                                depth=depthAfter,
                                activation=activation,
                                nFilters=nFilters,
                                nConv=nConv,
                                batchNormSync=True)
    xLeft = nonSymNet(processed_left)
    xRight = nonSymNet(processed_right)

    # BN
    xLeft = SyncBatchNormalization()(xLeft)
    xRight = SyncBatchNormalization()(xRight)
    x = SyncBatchNormalization()(x)

    # # convert dtypes to allow mixed precision
    # cast32Layer = Lambda(lambda tensors: tf.dtypes.cast(tensors, tf.float32) )
    # x = cast32Layer(x)
    # xLeft=cast32Layer(xLeft)
    # xRight=cast32Layer(xRight)

    # concat left
    if mergeLeft:
        x = Concatenate(axis=-1, dtype='float32')([x, xLeft])

    if globAvgPool:
        x = layers.GlobalAveragePooling3D()(x)


     #====
    # Flatten
    x = Flatten()(x)

    # prediction output lst
    predOutLst = [None] * len(tasksLst)
    for tID in range(len(tasksLst)):

        # additional dense layer
        if addDenseLayerNeurons > 0:
            x = Dense(units=addDenseLayerNeurons, activation='relu',name=tasksLst[tID]+'_dense')(x)
        # Prediction layer
        predOutLst[tID] = Dense(units=1, activation='linear', name=tasksLst[tID]+'_output')(x)

    return Model(inputs=[input_left, input_right], outputs=predOutLst)


class PCALayer(keras.layers.Layer):
    def __init__(self,pcaOutNum):
        super(PCALayer, self).__init__()

        self.pcaOutNum = pcaOutNum

    @tf.function
    def normalize(self,data):
        # creates a copy of data
        X = tf.identity(data)
        # calculates the mean
        X -=tf.reduce_mean(data, axis=0)
        return X

    @tf.function
    def pcaNoBatch(self,incVal,x):
        x = self.normalize(x)
        sqTe = tf.transpose(x) @ x

        eigVal, eigVec = tf.linalg.eigh(sqTe) # use eigh for batch
        eigVal = tf.math.real(eigVal)
        eigVec = tf.math.real(eigVec)

        newFeatNum = self.pcaOutNum
        # eigVec[:n_eigVec,:]
        # project data TODO: check transpose
        # out = tf.transpose(eigVec[:,:newFeatNum]) @  tf.transpose(x)
        # out = tf.transpose(out)
        out = x @ eigVec[:,-newFeatNum:]

        return out
        
    @tf.function
    def runPca(self,imgIn):
        # X: batch, width,height, features
        currShape = tf.shape(imgIn)
        # origShape = self.inputShape
        # print('origShape',origShape)
        # origShape = imgIn.shape
        # batchSize = origShape[0]
        # featNum = origShape[-1]
        
        # print('imgIn',imgIn.shape)
        
        x = tf.reshape(imgIn, [-1,currShape[-4]*currShape[-3]*currShape[-2],currShape[-1]])
        # x = tf.reshape(imgIn, [None, 3910, 24])
        
        newFeatNum = self.pcaOutNum
        outVecShape = tf.zeros_like(x)
        outVecShape = outVecShape[0,:,:newFeatNum]

        # apply pca to one element per batch
        out = tf.scan(self.pcaNoBatch, x, initializer=outVecShape)

        # print('out',out.shape)
        outShape = out.shape
        # reshape to original space
        out = tf.reshape(out, [-1,currShape[-4],currShape[-3],currShape[-2], outShape[-1]])

        return out

    def build(self, input_shape):
        self.inputShape  = input_shape
        super(PCALayer, self).build(input_shape)

        
        
    def call(self, inputs):
        # #[None, 10, 23, 17, 24]
        # tf.print('before pooling', inputs[0].shape)
        
        res = self.runPca(inputs[0]  )

        return res



def sNetVggNetWithSkipConnBNsOrdPca(input_shape,
                                    depthBefore=4,
                                    depthAfter=4,
                                    activation='relu',
                                    nFilters=32,
                                    nConv=3,
                                    globAvgPool=False,
                                    addDenseLayerNeurons=15,
                                    pcaComp =5,
                                    tasksLst=['core','pen']):
    """
    Add Multitask
    """
    base_network = vggNetWithInputLayer(input_shape,
                                 depth=depthBefore,
                                 activation=activation,
                                 nFilters=nFilters,
                                 nConv=nConv,
                                 batchNormSync=True,
                                 maxPool=False,
                                 usePca=True,
                                 pcaComp=pcaComp)

    input_left = Input(shape=(input_shape))
    input_right = Input(shape=(input_shape))
    # input_left = InputLayer(input_shape=(input_shape))
    # input_right = InputLayer(shape=(input_shape))



    # Re-use same instance of base network (weights are shared across both branches)
    processed_left = base_network(input_left)
    processed_right = base_network(input_right)


    # Merging features from two hemispheres
    # l1
    l1_distance_layer = Lambda(lambda tensors: K.abs(tensors[0] - tensors[1]))
    merged_layer = l1_distance_layer([processed_left, processed_right])

    #==== Inception modules after merge
    # symmetric branch
    x = vggNet(merged_layer,
            depth=depthAfter,
            activation=activation,
            nFilters=nFilters,
            nConv=nConv,
            batchNormSync=True,
            usePca=True,
            pcaComp=pcaComp)
    # non symmetric branches (shared weights between them)
    nonSymNet = vggNetWithInputLayer(processed_left.shape[1:],
                                depth=depthAfter,
                                activation=activation,
                                nFilters=nFilters,
                                nConv=nConv,
                                batchNormSync=True,
                                usePca=True,
                                 pcaComp=pcaComp)
    xLeft = nonSymNet(processed_left)
    xRight = nonSymNet(processed_right)

    # BN
    xLeft = SyncBatchNormalization()(xLeft)
    xRight = SyncBatchNormalization()(xRight)
    x = SyncBatchNormalization()(x)

    # concat
    x = Concatenate(axis=-1, dtype='float32')([x, xLeft, xRight])

    if globAvgPool:
        x = layers.GlobalAveragePooling3D()(x)


     #====
    # Flatten
    x = Flatten()(x)

    # prediction output lst
    predOutLst = [None] * len(tasksLst)
    for tID in range(len(tasksLst)):

        # additional dense layer
        if addDenseLayerNeurons > 0:
            x = Dense(units=addDenseLayerNeurons, activation='relu',name=tasksLst[tID]+'_dense')(x)
        # Prediction layer
        predOutLst[tID] = Dense(units=1, activation='linear', name=tasksLst[tID]+'_output')(x)

    return Model(inputs=[input_left, input_right], outputs=predOutLst)


class AbsDiffLayer(keras.layers.Layer):
    def __init__(self):
        super(AbsDiffLayer, self).__init__()

    def build(self, input_shape):
        self.inputShape  = input_shape
        
    def call(self, inputs):
        # print(self.inputShape)
        # #[None, 10, 23, 17, 24]

        res = tf.abs(inputs[0] - inputs[1])
        return res



def sNetVggNetWithSkipConnBN(input_shape,
                            depthBefore=4,
                            depthAfter=4,
                            activation='relu',
                            n_classes=2,
                            nFilters=32,
                            nConv=3,
                            globAvgPool=False,
                            addDenseLayerNeurons=15,
                            tasksLst=['core','pen']):
    """
    Add Multitask
    """
    base_network = vggNetWithInputLayer(input_shape,
                                 depth=depthBefore,
                                 activation=activation,
                                 nFilters=nFilters,
                                 nConv=nConv,
                                 batchNorm=True )

    input_left = Input(shape=(input_shape))
    input_right = Input(shape=(input_shape))
    # input_left = InputLayer(input_shape=(input_shape))
    # input_right = InputLayer(shape=(input_shape))



    # Re-use same instance of base network (weights are shared across both branches)
    processed_left = base_network(input_left)
    processed_right = base_network(input_right)


    # Merging features from two hemispheres
    # l1
    l1_distance_layer = Lambda(lambda tensors: K.abs(tensors[0] - tensors[1]))
    merged_layer = l1_distance_layer([processed_left, processed_right])

    #==== Inception modules after merge
    # symmetric branch
    x = vggNet(merged_layer,
            depth=depthAfter,
            activation=activation,
            nFilters=nFilters,
            nConv=nConv,
            batchNorm=True)
    # non symmetric branches (shared weights between them)
    nonSymNet = vggNetWithInputLayer(processed_left.shape[1:],
                                depth=depthAfter,
                                activation=activation,
                                nFilters=nFilters,
                                nConv=nConv,
                                batchNorm=True)
    xLeft = nonSymNet(processed_left)
    xRight = nonSymNet(processed_right)

    # BN
    xLeft = BatchNormalization()(xLeft)
    xRight = BatchNormalization()(xRight)
    x = BatchNormalization()(x)

    # concat
    x = Concatenate(axis=-1)([x, xLeft, xRight])

    if globAvgPool:
        x = layers.GlobalAveragePooling3D()(x)


     #====
    # Flatten
    x = Flatten()(x)

    # prediction output lst
    predOutLst = [None] * len(tasksLst)
    for tID in range(len(tasksLst)):

        # additional dense layer
        if addDenseLayerNeurons > 0:
            tmpLayer = Dense(units=addDenseLayerNeurons, activation='tanh',name=tasksLst[tID]+'_dense')(x)
            # Prediction layer
            predOutLst[tID] = Dense(units=n_classes, activation='softmax', name=tasksLst[tID]+'_output')(tmpLayer)

    return Model(inputs=[input_left, input_right], outputs=predOutLst)


class InstanceNormalization(tf.keras.layers.Layer):
    """Instance Normalization Layer (https://arxiv.org/abs/1607.08022).
    from https://github.com/tensorflow/examples/blob/master/tensorflow_examples/models/pix2pix/pix2pix.py
    """

    def __init__(self, epsilon=1e-5):
        super(InstanceNormalization, self).__init__()
        self.epsilon = epsilon

    def build(self, input_shape):
        self.scale = self.add_weight(
            name='scale',
            shape=input_shape[-1:],
            initializer=tf.random_normal_initializer(1., 0.02),
            trainable=True)

        self.offset = self.add_weight(
            name='offset',
            shape=input_shape[-1:],
            initializer='zeros',
            trainable=True)

    def call(self, x):
        mean, variance = tf.nn.moments(x, axes=[1, 2], keepdims=True)
        inv = tf.math.rsqrt(variance + self.epsilon)
        normalized = (x - mean) * inv
        return self.scale * normalized + self.offset

def upsample(filters, size, norm_type='batchnorm', apply_dropout=False, nameIn='upsample'):
    """Upsamples an input.
    from https://github.com/tensorflow/examples/blob/master/tensorflow_examples/models/pix2pix/pix2pix.py
    Conv2DTranspose => Batchnorm => Dropout => Relu
    Args:
    filters: number of filters
    size: filter size
    norm_type: Normalization type; either 'batchnorm' or 'instancenorm'.
    apply_dropout: If True, adds the dropout layer
    Returns:
    Upsample Sequential Model
    """

    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential(name=nameIn)
    # result.add(
    #     tf.keras.layers.UpSampling3D((2, 2, 2) ) )
    result.add(
        tf.keras.layers.Conv3DTranspose(filters, size, strides=2,
                                        padding='same',
                                        kernel_initializer=initializer,
                                        use_bias=False))


    if norm_type.lower() == 'batchnorm':
        result.add(tf.keras.layers.BatchNormalization())
    elif norm_type.lower() == 'instancenorm':
        result.add(InstanceNormalization())

    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.5))

    result.add(tf.keras.layers.ReLU())

    return result


def sNetAutoPreprocessingVggNetWithSkipConnBNmtSegOld(input_shape,
                                        depthBefore=4,
                                        depthAfter=4,
                                        activation='relu',
                                        n_classes=2,
                                        nFilters=32,
                                        nConv=3,
                                        addDenseLayerNeurons=15,
                                        tasksLst=['core','pen'],
                                        segOutLst=['coreSeg','penSeg']):
    """
    Add Multitask
    """
    base_network = vggNetWithInputLayer(input_shape,
                                 depth=depthBefore,
                                 activation=activation,
                                 nFilters=nFilters,
                                 nConv=nConv)

    input_left = Input(shape=(input_shape))
    input_right = Input(shape=(input_shape))
    # input_left = InputLayer(input_shape=(input_shape))
    # input_right = InputLayer(shape=(input_shape))

    nL = NormLayer(name='normLayer1')

    # Re-use same instance of base network (weights are shared across both branches)
    processed_left = base_network(nL(input_left))
    processed_right = base_network(nL(input_right))


    # Merging features from two hemispheres
    # l1
    l1_distance_layer = Lambda(lambda tensors: K.abs(tensors[0] - tensors[1]))
    merged_layer = l1_distance_layer([processed_left, processed_right])

    #==== Inception modules after merge
    # symmetric branch
    x = vggNet(merged_layer,
            depth=depthAfter,
            activation=activation,
            nFilters=nFilters,
            nConv=nConv,
            maxPool=False)
    # non symmetric branches (shared weights between them)
    nonSymNet = vggNetWithInputLayer(processed_left.shape[1:],
                                depth=depthAfter,
                                activation=activation,
                                nFilters=nFilters,
                                nConv=nConv,
                                maxPool=False)
    xLeft = nonSymNet(processed_left)
    xRight = nonSymNet(processed_right)

    # BN
    xLeft = BatchNormalization()(xLeft)
    xRight = BatchNormalization()(xRight)
    x = BatchNormalization()(x)

    # concat
    concLayer = Concatenate(axis=-1)([x, xLeft, xRight])

    # global average pooling
    x = layers.GlobalAveragePooling3D()(concLayer)


    #====
    # Flatten
    x = Flatten()(x)

    # prediction output lst
    predOutLst = [None] * len(tasksLst)
    for tID in range(len(tasksLst)):

        # additional dense layer
        if addDenseLayerNeurons > 0:
            tmpLayer = Dense(units=addDenseLayerNeurons, activation='tanh',name=tasksLst[tID]+'_dense')(x)
            # Prediction layer
            predOutLst[tID] = Dense(units=n_classes, activation='softmax', name=tasksLst[tID]+'_output')(tmpLayer)


    #==== segmentation branch
    s = upsample(3, 3, nameIn='updsample0')(concLayer)
    s = upsample(3, 3, nameIn='updsample1')(s)
    s = upsample(3, 3, nameIn='updsample2')(s)
    # crop to make tensor compatible with input shape
    s = tf.keras.layers.Cropping3D(cropping=((3,4), (1,1), (1,2)))(s)
    # n segmentation outputs
    predSegOutLst = [None] * len(segOutLst)
    for tSegID in range(len(segOutLst)):
        predSegOutLst[tSegID] = tf.keras.layers.Conv3D(2, kernel_size=(1, 1, 1), name=segOutLst[tSegID]+'_output')(s)
    #====

    return Model(inputs=[input_left, input_right], outputs=(predOutLst+predSegOutLst) )



def sNetAutoPreprocessingVggNetWithSkipConnBNmtSeg(input_shape,
                                        depthBefore=4,
                                        depthAfter=4,
                                        activation='relu',
                                        n_classes=2,
                                        nFilters=32,
                                        nConv=3,
                                        addDenseLayerNeurons=15,
                                        tasksLst=['core','pen'],
                                        segOutLst=['coreSeg','penSeg']):
    """
    Add Multitask
    """
    base_network = vggNetWithInputLayer(input_shape,
                                 depth=depthBefore,
                                 activation=activation,
                                 nFilters=nFilters,
                                 nConv=nConv)

    input_left = Input(shape=(input_shape))
    input_right = Input(shape=(input_shape))
    # input_left = InputLayer(input_shape=(input_shape))
    # input_right = InputLayer(shape=(input_shape))

    nL = NormLayer(name='normLayer1')

    # Re-use same instance of base network (weights are shared across both branches)
    processed_left = base_network(nL(input_left))
    processed_right = base_network(nL(input_right))


    # Merging features from two hemispheres
    # l1
    l1_distance_layer = Lambda(lambda tensors: K.abs(tensors[0] - tensors[1]))
    merged_layer = l1_distance_layer([processed_left, processed_right])

    #==== Inception modules after merge
    # symmetric branch
    x = vggNet(merged_layer,
            depth=depthAfter,
            activation=activation,
            nFilters=nFilters,
            nConv=nConv,
            maxPool=False)
    # non symmetric branches (shared weights between them)
    nonSymNet = vggNetWithInputLayer(processed_left.shape[1:],
                                depth=depthAfter,
                                activation=activation,
                                nFilters=nFilters,
                                nConv=nConv,
                                maxPool=False)

    xLeft = nonSymNet(processed_left)
    xRight = nonSymNet(processed_right)

    # BN
    xLeft = BatchNormalization()(xLeft)
    xRight = BatchNormalization()(xRight)
    x = BatchNormalization()(x)

    # # convert dtypes to allow mixed precision
    # x=tf.dtypes.cast(x, tf.float32)
    # xLeft=tf.dtypes.cast(xLeft, tf.float32)
    # xRight=tf.dtypes.cast(xRight, tf.float32)


    # concat
    concLayer = Concatenate(axis=-1)([x, xLeft, xRight])

    # global average pooling
    x = layers.GlobalAveragePooling3D()(concLayer)


    #====
    # Flatten
    x = Flatten()(x)

    # prediction output lst
    predOutLst = [None] * len(tasksLst)
    for tID in range(len(tasksLst)):

        # additional dense layer
        if addDenseLayerNeurons > 0:
            tmpLayer = Dense(units=addDenseLayerNeurons, activation='tanh',name=tasksLst[tID]+'_dense')(x)
            # Prediction layer
            predOutLst[tID] = Dense(units=n_classes, activation='softmax', name=tasksLst[tID]+'_output')(tmpLayer)


    #==== segmentation branch
    segBranch = tf.keras.Sequential(name='segbranch')
    segBranch.add(  upsample(3, 3, nameIn='upsample0_segbranch') )
    segBranch.add(  upsample(3, 3, nameIn='upsample1_segbranch') )
    segBranch.add(  upsample(3, 3, nameIn='upsample2_segbranch') )
    segBranch.add(  tf.keras.layers.Cropping3D(cropping=((3,4), (1,1), (1,2)), name='crop_segbranch') )
    s = segBranch( concLayer )


 

    # n segmentation outputs
    predSegOutLst = [None] * len(segOutLst)
    for tSegID in range(len(segOutLst)):
        predSegOutLst[tSegID] = tf.keras.layers.Conv3D(2, kernel_size=(1, 1, 1), name=segOutLst[tSegID]+'_output')(s)
 

    # s = upsample(3, 3, nameIn='upsample0_segbranch')(concLayer)
    # s = upsample(3, 3, nameIn='upsample1_segbranch')(s)
    # s = upsample(3, 3, nameIn='upsample2_segbranch')(s)
 
    # # crop to make tensor compatible with input shape
    # s = tf.keras.layers.Cropping3D(cropping=((3,4), (1,1), (1,2)), name='crop_segbranch')(s)
    # # n segmentation outputs
    # predSegOutLst = [None] * len(segOutLst)
    # for tSegID in range(len(segOutLst)):
    #     predSegOutLst[tSegID] = tf.keras.layers.Conv3D(2, kernel_size=(1, 1, 1), name=segOutLst[tSegID]+'_output')(s)
    #====

    return Model(inputs=[input_left, input_right], outputs=(predOutLst+predSegOutLst) )

class SaliencyVisualizer():
    def __init__(self, modelIn, rmSoftMax=False):
        # copy model and weights
        self.model =  tf.keras.models.clone_model( modelIn )
        self.model.set_weights( modelIn.get_weights() )
        if rmSoftMax:
            self.rmSoftMax()

        # set default output layers
        self.setOutputLayers()

        self.useInputForSal = False
    
    def rmSoftMax(self):
        """
        Replace softmax layer with Relu from last activation
        (useful for gradcam)
        """
        self.model.layers[-1].activation = tf.keras.activations.relu

    def setOutputLayers( self, inLayersLst=[], useInput=False ):
        """
        Set output layers for saliency. Do not add output.

        right now fixed to two elements in the array
        if useInput==True
        use inputs out for saliency
        """
        # default
        if (len(inLayersLst) == 0) and (useInput==False):
            lambdaName = [l.name for l in self.model.layers if 'lambda' in l.name][0]

            self.layerSal = [self.model.output,
                            self.model.get_layer(lambdaName).input[0],\
                            self.model.get_layer(lambdaName).input[1]]
        elif useInput:
            self.useInputForSal = True
        else:
            layLst = [self.model.output] + inLayersLst
            self.layerSal = layLst

    def getSaliency(self, brainArr, classIdx=1):
        """
        brainArr: numpy arry with two brain halves

        classIdx: compute 

        return [leftSaliency, right saliency, probability of positive class]
        """
    
        sampleTmpL = tf.convert_to_tensor( brainArr[0], dtype=tf.float32 )
        sampleTmpR = tf.convert_to_tensor( brainArr[1], dtype=tf.float32 )
        # sampleTmpL_var = tf.Variable(sampleTmpL) # needed 
        # check https://github.com/tensorflow/tensorflow/issues/36596
        # and https://www.sicara.ai/blog/2019-08-28-interpretability-deep-learning-tensorflow
        # http://blog.ai.ovgu.de/posts/jens/2019/001_tf20_pitfalls/index.html
        # modelVis = Model(model.input, outputs=[model.output, model.layers[0], model.layers[1]])

        if self.useInputForSal:
            # input/outputs same as original model
            modelVis2 = self.model
        else:
            modelVis2 = Model(self.model.input, outputs=self.layerSal)
        # gPred, in0, in1 = modelVis2([sampleTmpL,sampleTmpR])
        with tf.GradientTape(persistent=True) as g:
            if self.useInputForSal:
                gPred = modelVis2([sampleTmpL,sampleTmpR])
                in0 = modelVis2.inputs[0]
                in1 = modelVis2.inputs[1]
            else:
                gPred, in0, in1 = modelVis2([sampleTmpL,sampleTmpR])

            predOutput = gPred[0,classIdx]

        dy_dxl_lambda = g.gradient(predOutput, in0)
        dy_dxr_lambda = g.gradient(predOutput, in1)

        del g # needed because persistent parameter in GradientTape

        print(np.max(np.array(dy_dxl_lambda).flatten()))

        # average gradients for convolutional layers
        resample_size = sampleTmpL.shape[1:-1]

        redLamL = tf.reduce_mean(dy_dxl_lambda, axis=(0, 4))
        redLamR = tf.reduce_mean(dy_dxr_lambda, axis=(0, 4))
        redLamL = resize(np.array(redLamL), resample_size, order=0, anti_aliasing=False, preserve_range=True)
        redLamR = resize(np.array(redLamR), resample_size, order=0, anti_aliasing=False, preserve_range=True)

        return [redLamL,redLamR, np.array(gPred[:,1][0])]



    def getGradCam(self, brainArr, classIdx=1, outcomeIdx=0, onlyPos=False):
        """
        Compute GradCam
        brainArr: numpy arry with two brain halves

        classIdx: compute grad cam for given class idx, set it to None if only one class (such as regression) 
        outcomeIdx: id of the outcome
        onlyPos: if true return only the positive gradients


        return [leftSaliency, right saliency, probability of positive class]
        """
    
        sampleTmpL = tf.convert_to_tensor( brainArr[0], dtype=tf.float32 )
        sampleTmpR = tf.convert_to_tensor( brainArr[1], dtype=tf.float32 )
        # http://openaccess.thecvf.com/content_ICCV_2017/papers/Selvaraju_Grad-CAM_Visual_Explanations_ICCV_2017_paper.pdf
        # https://towardsdatascience.com/demystifying-convolutional-neural-networks-using-gradcam-554a85dd4e48

        if self.useInputForSal:
            # input/outputs same as original model
            modelVis2 = self.model
        else:
            modelVis2 = Model(self.model.input, outputs=self.layerSal)
        with tf.GradientTape(persistent=True) as g:
            # g.watch(smplIn)
            if self.useInputForSal:
                gPred = modelVis2([sampleTmpL,sampleTmpR])
                in0 = modelVis2.inputs[0].output
                in1 = modelVis2.inputs[1].output
            else:
                gPred, in0, in1 = modelVis2([sampleTmpL,sampleTmpR])

            if classIdx is None:
                predOutput = gPred[outcomeIdx]
            else:
                predOutput = gPred[outcomeIdx,classIdx]
        # calculated puside GradientTape for efficiency
        # all variables needs to be watched or be generated inside GradientTape
        dy_dxl_lambda = g.gradient(predOutput, in0)
        dy_dxr_lambda = g.gradient(predOutput, in1)

        del g # needed because persistent parameter in GradientTape

        print('avg grad 0', tf.reduce_mean(dy_dxl_lambda) )
        print('avg grad 1', tf.reduce_mean(dy_dxr_lambda) )

        # average gradients for convolutional layers
        resample_size = sampleTmpL.shape[1:-1]


        # compute filter weights A
        print(dy_dxr_lambda.shape)
        aWeightsL = tf.reduce_mean(dy_dxl_lambda[0,:,:,:,:], axis=(0,1,2))
        aWeightsR = tf.reduce_mean(dy_dxr_lambda[0,:,:,:,:], axis=(0,1,2))

        # RElu (or clip 0s) # change from standard implementation
        dy_dxl_lambda_0l = tf.clip_by_value( dy_dxl_lambda[0,:,:,:,:], 0, tf.math.reduce_max(dy_dxl_lambda[0,:,:,:,:]))
        dy_dxl_lambda_0r = tf.clip_by_value( dy_dxr_lambda[0,:,:,:,:], 0, tf.math.reduce_max(dy_dxr_lambda[0,:,:,:,:]))

                
        # original
        # redLamL = dy_dxl_lambda[0,:,:,:,:] * aWeightsL
        # redLamR = dy_dxr_lambda[0,:,:,:,:] * aWeightsR

        # weigh filter output by gradient
        redLamL = dy_dxl_lambda_0l * aWeightsL
        redLamR = dy_dxl_lambda_0r * aWeightsR

        # # weigh filter output by gradient
        # redLamL = dy_dxl_lambda_0l 
        # redLamR = dy_dxl_lambda_0r 


        #reduce
        redLamL = tf.reduce_mean( redLamL, axis=(3) )
        redLamR = tf.reduce_mean( redLamR, axis=(3) )

        # RElu (or clip 0s)
        redLamL = tf.clip_by_value( redLamL, 0, tf.math.reduce_max(redLamL))
        redLamR = tf.clip_by_value( redLamR, 0, tf.math.reduce_max(redLamR))

        # resize
        redLamL = resize(np.array(redLamL), resample_size, order=0, anti_aliasing=False, preserve_range=True)
        redLamR = resize(np.array(redLamR), resample_size, order=0, anti_aliasing=False, preserve_range=True)

        if onlyPos:
            # show only i fpositive avera gradient
            pass

        if classIdx is None:
            outPred = np.array(gPred[:][0])
        else:
            outPred = np.array(gPred[:,classIdx][0])

        return [redLamL,redLamR, outPred]

if __name__ == '__main__':
    # test
    print('no default operation')

    # test saliency
