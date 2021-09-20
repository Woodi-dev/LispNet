(in-package #:lispnet)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;
;;; classification of mnist data

;;; note: we only ship a very small sample of the mnist data with petalisp,
;;; so learning will not be too good.  if you want to run serious tests,
;;; you should replace these paths with paths to the full mnist data set.

#|(defparameter *mnist*
  (asdf:find-component
   (asdf:find-system "petalisp.examples")
   "mnist-data"))|#

(defun load-array (path)
  (numpy-file-format:load-array
   (asdf:system-relative-pathname (asdf:find-system "lispnet") path)))

(defparameter *train-images* (load-array "examples/mnist-data/train-images.npy"))
(defparameter *train-labels* (load-array "examples/mnist-data/train-labels.npy"))
(defparameter *test-images* (load-array "examples/mnist-data/test-images.npy"))
(defparameter *test-labels* (load-array "examples/mnist-data/test-labels.npy"))

(defun check-test-data (model index)
  (let* ((input (compute   (lazy-slices (lazy #'/ *test-images* 255.0) (range index (+ 1 index)))))
         (prediction (compute (lazy-drop-axes (predict model input) 0)))
         (predict-val (multiple-value-list (argmax prediction))))
    (format t "label: ~s~%" (argmax(compute (lazy-drop-axes (lazy-slices *test-labels* (range index (+ 1 index))) 0))))
    ;;(format t "prediction: ~s~%" prediction)
    (format t "prediction: ~s with certainty: ~s~%" (first predict-val) (second predict-val)))
	
	
	)

(defclass mnist-model (model)
  ((dense1 :accessor dense1)
   (conv1 :accessor conv1)
   (conv2 :accessor conv2)
   (maxpool1 :accessor maxpool1)
   (maxpool2 :accessor maxpool2)
   (flatten1 :accessor flatten1)))

(defmethod initialize-instance :after ((model mnist-model) &rest initargs)
  (setf (dense1 model) (make-dense-layer model :in-features 3136 :out-features 10 :activation #'softmax))
  (setf (conv1 model) (make-conv2d-layer model :in-channels 1 :out-channels 32 :padding "same" :kernel-size 3  :strides '(1 1) :activation #'relu))
 (setf (conv2 model) (make-conv2d-layer model :in-channels 32 :out-channels 64 :padding "same"  :kernel-size 3 :strides '(1 1) :activation #'relu))
  (setf (maxpool1 model) (make-maxpool2d-layer model :pool-size '(2 2)))
  (setf (maxpool2 model) (make-maxpool2d-layer model :pool-size '(2 2)))
  (setf (flatten1 model) (make-flatten-layer model)))

(defmethod forward ((model mnist-model) input)
   (call (dense1 model)
      (call (flatten1 model)
	  (call (maxpool2 model)
		   (call (conv2 model)  	
				 (call (maxpool1 model)
					 (call (conv1 model)
                          (lazy-reshape input (transform a b c to a b c 0)))))))))
						  
						  

(defun train-mnist()
  (let*  ((optimizer (make-adam :learning-rate 0.01))
          (model (make-instance 'mnist-model)))
    (model-compile model :loss #'categorial-cross-entropy :optimizer optimizer :metrics (list #'categorial-accuracy #'binary-accuracy )) ;;
    (let* ((train-input-data
             (compute (lazy-slices (lazy #'/ *train-images* 255.0) (range 0 1000))))
           (train-label-data
             (compute (lazy-slices *train-labels*(range 0 1000)) ))

           (val-input-data
             (compute (lazy-slices (lazy #'/ *test-images* 255.0)(range 0 200))))
           (val-label-data
             (compute (lazy-slices *test-labels*(range 0 200)) )))
	;;(inspect (model-weights model))
      (format t "--------------------------~%")
      (model-summary model)
      (fit model train-input-data train-label-data val-input-data val-label-data :epochs 15 :batch-size 32)
      (format t "check test data~%")
      (loop for i from 0 below 10 do
        (check-test-data model i)))))

;;(train-mnist)



(defclass test-model (model)
  ((dense1 :accessor dense1)
  (dense2 :accessor dense2)
  ))

(defmethod initialize-instance :after ((model test-model) &rest args)
  (setf (dense1 model) (make-dense-layer model :in-features 10000 :out-features 10))
  (setf (dense2 model) (make-dense-layer model :in-features 10 :out-features 50))

  )

(defmethod forward ((model test-model) input)
 (call (dense2 model)(call (dense1 model) input)))
 ;;(lazy #'expt (lazy #'-  (call (conv1 model) input)(lazy-reshape 1.0 (~ 1 ~ 2 ~ 4 ~ 4))) 2))
;; (mse (lazy-reshape 1.0 (~ 1 ~ 4 ~ 4 ~ 4))


(defmethod testpredict ((model model) input)
    (let* ((args '())
                  (sample-shape (~l (mapcar #'range(array-dimensions input))))
              (input-parameter (make-unknown :shape (~ 1 ~s sample-shape) :element-type 'single-float))
                  (network (make-network(forward model input-parameter))))
        ;; inputs.
                (push input-parameter args)
        (push (lazy-reshape input (~ 1 ~s sample-shape))  args)

        ;; trainable parameters.
                (loop for trainable-parameter in (model-weights model) do
                        (push (weights trainable-parameter) args)
                        (push (weights-value trainable-parameter) args))
                (compute
                                 (lazy-array
                                          (values-list (apply #'call-network network (reverse args)))) )))

(defun testmain()
  (let*  ((optimizer (make-sgd :learning-rate 0.01))
          (model (make-instance 'test-model))
		  (input (compute (glorot-uniform :shape (~ 1000 ~ 10000) :fan-in 10000 :fan-out 50)))
		  (output (compute (glorot-uniform :shape (~ 1000 ~ 50) :fan-in 50 :fan-out 50)))
		  (weights (model-weights model))
		  )
    (model-compile model :loss #'mse :optimizer optimizer)
	;;(print "Weights")
	;;(print (weights-value (first weights)))
	;;(print (weights-value (second weights)))


	(fit model input output input output :epochs 1 :batch-size 1000)
	;;(print "Weights afterwards")
	;;	(print (weights-value (first weights)))
	;;(print (weights-value (second weights)))
	))
	
	;;(setf (weights-value (first (model-weights model))) (lazy-reshape weights2 (~ 1 ~ 2 ~ 2 ~ 1)))
    ;;(print (compute (testpredict model (compute(lazy-reshape input (~ 4 ~ 4 ~ 1))))))))


;;(testmain)

