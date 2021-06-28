
(in-package :common-lisp-user)
(defpackage #:lispnet.mnist
  (:use
   #:common-lisp
   #:petalisp
   #:lispnet.layers
   #:lispnet.network
  )
  (:export

   ))

(in-package #:lispnet.mnist)


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;
;;; Classification of MNIST Data

;;; Note: We only ship a very small sample of the MNIST data with Petalisp,
;;; so learning will not be too good.  If you want to run serious tests,
;;; you should replace these paths with paths to the full MNIST data set.

(defparameter *mnist*
  (asdf:find-component
   (asdf:find-system "petalisp.examples")
   "mnist-data"))

(defun load-array (&rest path)
  (numpy-file-format:load-array
   (asdf:component-pathname
    (asdf:find-component *mnist* path))))

(defparameter *train-images* (load-array "train-images.npy"))
(defparameter *train-labels* (load-array "train-labels.npy"))
(defparameter *test-images* (load-array "test-images.npy"))
(defparameter *test-labels* (load-array "test-labels.npy"))

(defun make-test-network ()
  (let* ((input (make-instance 'parameter
                               :shape (~ 0 28 ~ 0 28)
                               :element-type 'single-float)))

    (values
     (make-network
      (fully-connected
       input
       (~ 0 10))
      )
     input))

  )




(defun check-test-data (network index)
  (format t "Check test data~%")
  (format t "Label: ~S~%" (compute (lazy-drop-axes (lazy-slices *test-labels* (range index (+ 1 index))) 0)))

  (let ((input (compute  (lazy-drop-axes (lazy-slices (lazy #'/ *test-images* 255.0) (range index (+ 1 index))) 0 )))
        )
   
    (format t "Prediction: ~S~%"
    (compute (predict network input)))
	
    )
  
  )

(defun main ()
  (multiple-value-bind (network input)
      (make-test-network)
	  ;;pre-processing
	  (let* ((input-data
                 (compute (lazy #'/ *train-images* 255.0)))
		     (label-data
                   (compute 
                    (lazy-collapse
                     (lazy 'coerce
                           (lazy (lambda (n i) (if (= n i) 1.0 0.0))
                                 (lazy-reshape *train-labels* (transform i to i 0))
                                 #(0 1 2 3 4 5 6 7 8 9))
						'single-float))))								 
			)
    (fit network input input-data label-data  :epochs 10 :batch-size 10 :learning-rate 0.01))
    (check-test-data network 0)))

(main)


;;(print (main))

;;(check-test-data (main) 0)



;;multiple-value-bind (net inp) (main) (check-test-data net 0))
