(in-package #:lispnet)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;
;;; classification of mnist data


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
         (predict-val (multiple-value-list (argmax prediction)))
	 (label (argmax(compute (lazy-drop-axes (lazy-slices *test-labels* (range index (+ 1 index))) 0)))))
		 
    (when (not (= label (first predict-val)))
      (format t "label: ~S~%" label)
      (format t "prediction: ~S with certainty: ~S~%" (first predict-val) (second predict-val)))
	
    (= label (first predict-val))))

(defclass mnist-model (model)
  ())

(defmethod initialize-instance :after ((model mnist-model) &rest initargs))

(defmethod forward ((model mnist-model) input)
  (let* ((conv1 (create-layer 'conv2d-layer model :in-channels 1 :out-channels 16 :padding "valid" :activation #'relu))
	 (maxpool1 (create-layer 'maxpool2d-layer model ))
	 (conv2 (create-layer 'conv2d-layer model :in-channels 16 :out-channels 32 :padding "valid" :activation #'relu))
	 (maxpool2 (create-layer 'maxpool2d-layer model ))
         (dense (create-layer 'dense-layer model :out-features 10 :activation #'softmax))	 
	 (flatten (create-layer 'flatten-layer model)))	  
    (call dense 
	  (call flatten
		(call maxpool2
		      (call conv2
			    (call maxpool1
			          (call conv1 (lazy-reshape input (transform b y x to b y x 0))))))))))
						
					
 
						  
						  

(defun train-mnist()
  ;;(setf petalisp.core:*backend* (petalisp.xmas-backend:make-xmas-backend))
  (setf petalisp.core:*backend* (petalisp.multicore-backend:make-multicore-backend))
  (setf *network-precision* 'single-float)
  (let*  ((optimizer (make-adam :learning-rate 0.001))
          (model (make-instance 'mnist-model))
	  (train-input-data
            (compute (lazy-slices (lazy #'/ *train-images* 255.0) (range 0 10000))))
          (train-label-data
            (compute (lazy-slices *train-labels*(range 0 10000)) ))

          (val-input-data
            (compute (lazy-slices (lazy #'/ *test-images* 255.0)(range 0 1000))))
          (val-label-data
            (compute (lazy-slices *test-labels*(range 0 1000)) )))
    (format t "--------------------------~%")
	 
    (model-compile model :loss #'categorial-cross-entropy :optimizer optimizer  :metrics (list #'categorial-accuracy)) ;;
    (model-summary model :sample-shape (~ 28 ~ 28))
	 
    (fit model train-input-data train-label-data val-input-data val-label-data :epochs 20 :batch-size 64)
    (save-weights model "mnist-weights/")
	  
    ;;  (predict model (compute (lazy-slices train-input-data (range 0 1))))   

		
    ))


;;(train-mnist)


