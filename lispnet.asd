(defsystem "lispnet"
  :description "Machine Learning API"
  :author "Michael Holzmann <michael.holzmann@fau.de>"

  :depends-on
  ("petalisp"
   "numpy-file-format"
   "asdf")

  :in-order-to ((test-op (test-op "petalisp.test-suite")))

  :serial t
  :components
  (
  (:file "linear-algebra")
  (:file "utils")
  (:file "packages")
  (:file "model")
  (:file "trainable-parameter")
  (:module "mnist-data"
    :components
    ((:static-file "test-images.npy")
     (:static-file "test-labels.npy")
     (:static-file "train-images.npy")
     (:static-file "train-labels.npy")))
   (:file "initializers")
   (:file "layers/layer")
   (:file "layers/activations")
   (:file "layers/conv2d")
   (:file "layers/dense")
   (:file "layers/flatten")
   (:file "loss")
   (:file "optimizer")
   (:file "examples/mnist")
   


   ))
