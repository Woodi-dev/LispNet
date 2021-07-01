
(in-package :common-lisp-user)

(defpackage #:lispnet.layers
  (:use
   #:common-lisp
   #:petalisp
   #:lispnet.initializers
   #:lispnet.trainable-parameter
   )
  (:export
   #:softmax
   #:relu
   #:sigmoid
   #:fully-connected
   ))

(in-package #:lispnet.layers)

(defun sigmoid(input)
(lazy #'/ 1.0 (lazy #'+ 1.0 (lazy #'exp (lazy #'* -1.0 input)))))


(defun softmax (input)
  (let ((totals (lazy #'exp input)))
    (lazy #'/ totals (lazy-allreduce #'+ totals))))

(defun relu (input)
  (lazy #'max (coerce 0 (element-type input)) input))
  
(defun fully-connected (input output-shape)
  (let* ((m (shape-size output-shape))
         (n (total-size input))
         (weights
           (make-trainable-parameter
            (lazy #'/
                  (make-random-array (list n m) :element-type (element-type input))
                  (* m n))))
         (bias
           (make-trainable-parameter
            (lazy #'/
                  (make-random-array m :element-type (element-type input))
                  m))) 

         )
        ;;(lazy #'+ bias (lazy-slices (lazy-flatten weights) (range 0 10)));;this simple test works for forward/backward pass					
		;;comment out following equation and use the line above to test the forward and backward pass	 
         (lazy-reduce
           #'+
           (lazy #'*
                 weights
                 (lazy-reshape (lazy-flatten input) (transform n to n 0)))) 
		  
    
    )
  
  )
