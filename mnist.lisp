
(in-package :common-lisp-user)
(defpackage #:lispnet.mnist
  (:use
   #:common-lisp
   #:petalisp
   #:lispnet.layers
   #:lispnet.network
   #:lispnet.loss
   #:lispnet.optimizer
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

(defun make-mnist-network ()
  (let* ((input (make-instance 'parameter
                  :shape (~ 0 28 ~ 0 28)
                  :element-type 'single-float)))

    (values
     (make-network
      (fcn
       (fcn (flatten
             (conv-2d
              (lazy-reshape input (~ 1 ~ 28 ~ 28))
              :n-filters 32
              :kernel-size 3
              :strides '(2 2)
              ;;:stencil '((0 0) (1 0) (0 1) (-1 0) (0 -1))
              :padding "same"
              :activation #'relu))
            :units 100
            :activation #'relu)
       :units 10
       :activation #'softmax))
     input)))

(defun make-test-network ()
  (let* ((input (make-instance 'parameter
                  :shape (~ 0 28 ~ 0 28)
                  :element-type 'single-float)))
    (values
     (make-network

      (fcn
       (flatten
        (conv-2d
         (lazy-reshape input (~ 1 ~ 28 ~ 28))
         :n-filters 3
         :kernel-size 3
         :strides '(2 2)
         ;;:stencil '((0 0) (1 0) (0 1) (-1 0) (0 -1))
         :padding "same"
         :activation #'relu))
       :units 10
       :activation #'softmax))
     input)))

(defun argmax (li)
  (let ((max-index 0)
        (max-value (aref li 0)))
    (loop for val across li
          for index from 0 do
            (when (> val max-value)  (setf max-value val) (setf max-index index)))
    (values max-index max-value)))

(defun check-test-data (network index)
  (let* ((input (compute  (lazy-drop-axes (lazy-slices (lazy #'/ *test-images* 255.0) (range index (+ 1 index))) 0 )))
         (prediction (compute (predict network input)))
         (predict-val (multiple-value-list (argmax prediction))))
    (format t "Label: ~S~%" (compute (lazy-drop-axes (lazy-slices *test-labels* (range index (+ 1 index))) 0)))
    ;; (format t "Prediction: ~S~%" prediction)
    (format t "Prediction: ~S with certainty: ~S~%" (first predict-val) (second predict-val))))

(defun main ()
  (multiple-value-bind (network input)
      (make-test-network)
    ;;pre-processing
    (let* ((train-input-data
             (compute (lazy-slices (lazy #'/ *train-images* 255.0) (range 0 100))))
           (train-label-data
             (compute (lazy-slices
                       (lazy-collapse
                        (lazy 'coerce
                              (lazy (lambda (n i) (if (= n i) 1.0 0.0))
                                    (lazy-reshape *train-labels* (transform i to i 0))
                                    #(0 1 2 3 4 5 6 7 8 9))
                              'single-float))(range 0 100)) ))

           (val-input-data
             (compute (lazy-slices (lazy #'/ *test-images* 255.0)(range 0 100))))
           (val-label-data
             (compute (lazy-slices
                       (lazy-collapse
                        (lazy 'coerce
                              (lazy (lambda (n i) (if (= n i) 1.0 0.0))
                                    (lazy-reshape *test-labels* (transform i to i 0))
                                    #(0 1 2 3 4 5 6 7 8 9))
                              'single-float))(range 0 100))))
           (optimizer (make-adam :learning-rate 0.001 :network network))
           )
      (format t "~%Trainable parameters: ~S~%" (network-weights-size network))
      (fit network input train-input-data train-label-data val-input-data val-label-data :epochs 20 :batch-size 5 :loss #'binary-cross-entropy :optimizer optimizer))
    (format t "Check test data~%")
    (loop for i from 0 to 10 do
      (check-test-data network i))))

;(main)

