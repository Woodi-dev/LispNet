(defsystem "lispnet"
  :description "Machine Learning API"
  :author "Michael Holzmann <michael.holzmann@fau.de>"

  :depends-on
  ("petalisp"
   "petalisp.graphviz"
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
  (:file "initializers")
  (:file "layers/layer")
  (:file "layers/activations")
  (:file "layers/conv2d")
  (:file "layers/transposed-conv2d")
  (:file "layers/dense")
  (:file "layers/flatten")
  (:file "layers/maxpool2d")
  (:file "loss")
  (:file "optimizer")
  (:file "examples/mnist")
  (:file "examples/isbi")))
