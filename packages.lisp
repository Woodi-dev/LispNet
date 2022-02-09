(in-package :common-lisp-user)

(defpackage #:lispnet
   (:use
   #:common-lisp
   #:petalisp)
  (:export


   ;;layer
   #:softmax
   #:relu
   #:sigmoid
   #:conv-2d


   #:layer
   #:call
   #:create-layer
   #:layer-weights
   #:layer-activation
   #:layer-compile
   #:dense-layer
   #:flatten-layer
   #:conv2d-layer
   #:maxpool2d-layer
   #:transposed-conv2d-layer


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
   #:*network-precision*
   #:save-weights
   #:load-weights
   
   ;;model-state
   #:model-state
   #:model-state-layer-pointer
   #:model-state-running
   #:model-state-weights-initialized
   
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
   #:output-loss
   #:binary-cross-entropy
   #:categorial-cross-entropy

   ;;metrics
   #:categorial-accuracy
   #:binary-accuracy

   ;;initializers
   #:init-weights
   #:glorot-uniform
   #:zeros
   #:ones

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
   #:weights
   #:weights-value
   
   ;;linear-algebra
   #:transpose
   #:matmul
   #:l2norm
   
   ;;utils
   #:make-2d-kernel
   #:pad
   #:stride-shape
   #:stride-range
   #:argmax
   #:lazy-batch-argax
   #:lazy-allreduce-batchwise
   #:binary-decision
   ))
