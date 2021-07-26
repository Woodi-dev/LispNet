(in-package :common-lisp-user)

(defpackage #:lispnet
   (:use
   #:common-lisp
   #:petalisp
   #:lispnet.linear-algebra
   #:lispnet.utils)
  (:export


   ;;layer
   #:softmax
   #:relu
   #:sigmoid
   #:conv-2d

   #:layer
   #:call
   #:layer-weights
   #:layer-activation
   #:layer-compile
   #:dense-layer
   #:make-dense-layer
   #:make-flatten-layer
   #:flatten-layer
   #:conv2d-layer
   #:make-conv2d-layer



   ;;model
   #:model
   #:model-compile
   #:forward
   #:model-layers
   #:model-loss
   #:model-optimizer
   #:model-weights
   #:model-weights-total
   #:model-summary
   #:predict

   ;;optimizer
   #:optimizer
   #:sgd
   #:update-weights
   #:make-sgd
   #:adam
   #:make-adam
   #:optimizer-compile
   #:last-gradients

   ;;loss
   #:mse
   #:mae
   #:binary-cross-entropy

   ;;metrics
   #:categorial-accuracy
   #:binary-accuracy

   ;;initializers
   #:init-weights
   #:glorot-uniform
   #:zeros

   ;;network
   #:train-test
   #:predict
   #:fit
   #:network-weights
   #:network-weights-size

   ;;trainable-parameter
   #:trainable-parameter
   #:make-trainable-parameter
   #:trainable-parameter-p

   ))
