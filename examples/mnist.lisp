(in-package #:lispnet)


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

(defun load-array (path)
  (numpy-file-format:load-array
   (asdf:system-relative-pathname (asdf:find-system "lispnet") path)))

(defparameter *train-images* (load-array "examples/mnist-data/train-images.npy"))
(defparameter *train-labels* (load-array "examples/mnist-data/train-labels.npy"))
(defparameter *test-images* (load-array "examples/mnist-data/test-images.npy"))
(defparameter *test-labels* (load-array "examples/mnist-data/test-labels.npy"))


(defun check-test-data (model index)
  (let* ((input (compute  (lazy-drop-axes (lazy-slices (lazy #'/ *test-images* 255.0) (range index (+ 1 index))) 0 )))
         (prediction (compute (predict model input)))
         (predict-val (multiple-value-list (argmax prediction))))
    (format t "Label: ~S~%" (argmax(compute (lazy-drop-axes (lazy-slices *test-labels* (range index (+ 1 index))) 0))))
    ;;(format t "Prediction: ~S~%" prediction)
    (format t "Prediction: ~S with certainty: ~S~%" (first predict-val) (second predict-val))))



(defclass mnist-model (model)
   ((dense1 :accessor dense1)
    (dense2 :accessor dense2)
    (conv1 :accessor conv1)
    (conv2 :accessor conv2)	
    (conv3 :accessor conv3)		
    (flatten1 :accessor flatten1)))
   
(defmethod initialize-instance :after ((model mnist-model) &rest initargs)
	(setf (dense2 model) (make-dense-layer model :in-features 100 :out-features 10  :activation #'softmax))
	(setf (dense1 model) (make-dense-layer model :in-features 256 :out-features 100 :activation #'relu))
	(setf (conv1 model) (make-conv2d-layer model :in-channels 1 :out-channels 4 :padding "same" :strides '(2 2) :activation #'relu))
	(setf (conv2 model) (make-conv2d-layer model :in-channels 4 :out-channels 8 :padding "same" :strides '(2 2) :activation #'relu))
	(setf (conv3 model) (make-conv2d-layer model :in-channels 8 :out-channels 16 :padding "same" :strides '(2 2) :activation #'relu))
	(setf (flatten1 model) (make-flatten-layer model))) 
   

(defmethod forward ((model mnist-model) input)
  (call (dense2 model) 
	(call (dense1 model) 
		(call (flatten1 model)
			(call (conv3 model)
				(call (conv2 model)
					(call (conv1 model) 
						(lazy-reshape input (transform a b c to a 0 b c)))))))))
					;;input))))






(defun main()
	(let*  ((optimizer (make-adam :learning-rate 0.001))
			(model (make-instance 'mnist-model)))		
			(model-compile model :loss #'binary-cross-entropy :optimizer optimizer ) ;;:metrics (list #'categorial-accuracy)
	  (let* ((train-input-data
                 (compute (lazy-slices (lazy #'/ *train-images* 255.0) (range 0 200)))) 
		    (train-label-data
                   (compute (lazy-slices *train-labels*(range 0 200)) ))
						   
			(val-input-data
                 (compute (lazy-slices (lazy #'/ *test-images* 255.0)(range 0 200))))  
		    (val-label-data
                   (compute (lazy-slices *test-labels*(range 0 200)) )))

			(format t "--------------------------~%")
		    (model-summary model)
		    (fit model train-input-data train-label-data val-input-data val-label-data :epochs 40 :batch-size 50)
			(format t "Check test data~%")
			(loop for i from 0 below 10 do
			(check-test-data model i)))))

(main)




#|
(defclass test-model (model)
  ((conv1 :accessor conv1)
  ))

(defmethod initialize-instance :after ((model test-model) &rest args)
  (setf (conv1 model) (make-conv2d-layer model :in-channels 2 :out-channels 2 :padding "same"))
  )

(defmethod forward ((model test-model) input)
 (call (conv1 model) input))
 ;;(lazy #'expt (lazy #'-  (call (conv1 model) input)(lazy-reshape 1.0 (~ 1 ~ 2 ~ 4 ~ 4))) 2))
;; (mse (lazy-reshape 1.0 (~ 1 ~ 4 ~ 4 ~ 4))


(defmethod testpredict ((model model) input)
    (let* ((args '())
		  (sample-shape (~l (mapcar #'range(array-dimensions input))))
	      (input-parameter (make-unknown :shape (~ 1 ~s sample-shape) :element-type 'single-float))
		  (network (make-network(forward model input-parameter))))
        ;; Inputs.
		(push input-parameter args)
        (push (lazy-reshape input (~ 1 ~s sample-shape)) args)

        ;; Trainable parameters.
		(loop for trainable-parameter in (model-weights model) do
			(push (weights trainable-parameter) args)
			(push (weights-value trainable-parameter) args))
		(compute 
			         (lazy-array
				          (values-list (apply #'call-network network (reverse args)))) )))
						  
(defun testmain()
  (let*  ((optimizer (make-adam :learning-rate 0.001))
          (model (make-instance 'test-model)))		
    (model-compile model :loss #'mse :optimizer optimizer)
    (print (compute (testpredict model (compute (lazy-reshape 1.0 (~ 2 ~ 4 ~ 4))))))))
|#

;;(testmain)
