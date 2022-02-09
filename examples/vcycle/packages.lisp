(in-package :common-lisp-user)

(defpackage #:lispnet.examples.vcycle
   (:use
   #:common-lisp
   #:petalisp
   #:lispnet)
  (:export

   #:vcycle-model
   #:attention-vcycle-model
   #:create-restriction-matrix
   #:restriction-layer
   #:jacobi-layer
   #:residual-layer
   #:rgbs-layer
   ))

