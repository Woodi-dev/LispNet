(in-package #:lispnet)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;
;;; classification of mnist data

;;; note: we only ship a very small sample of the mnist data with petalisp,
;;; so learning will not be too good.  if you want to run serious tests,
;;; you should replace these paths with paths to the full mnist data set.

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
    ;;(format t "prediction: ~S~%" prediction)
    (format t "prediction: ~S with certainty: ~S~%" (first predict-val) (second predict-val)))
	
	(= label (first predict-val))))

(defclass mnist-model (model)
  ())

(defmethod initialize-instance :after ((model mnist-model) &rest initargs)
  ;;(setf (dense1 model) (make-dense-layer model :out-features 10 :activation #'sigmoid))
  ;;(setf (flatten1 model) (make-flatten-layer model))
)

(defmethod forward ((model mnist-model) input)
  (let* ((conv1 (create-layer 'conv2d-layer model :in-channels 1 :out-channels 16 :activation #'relu))
	  (maxpool1 (create-layer 'maxpool2d-layer model :padding "same"))
	  (conv2 (create-layer 'conv2d-layer model :in-channels 16 :out-channels 32 :activation #'relu))
	  (maxpool2 (create-layer 'maxpool2d-layer model :padding "same"))
      (dense (create-layer 'dense-layer model :out-features 10 :activation #'softmax))
	 
	  (flatten (create-layer 'flatten-layer model)))
	  

		(call dense 
		(call flatten
			(call maxpool2
				(call conv2
					(call maxpool1
						(call conv1 (lazy-reshape input (transform b y x to b y x 0))))))))))
						
					
 
						  
						  

(defun train-mnist()


  (let*  ((optimizer (make-adam :learning-rate 0.001))
          (model (make-instance 'mnist-model))
		 (train-input-data
             (compute (lazy-slices (lazy #'/ *train-images* 255.0) (range 0 20))))
           (train-label-data
             (compute (lazy-slices *train-labels*(range 0 20)) ))

           (val-input-data
             (compute (lazy-slices (lazy #'/ *test-images* 255.0)(range 0 100))))
           (val-label-data
             (compute (lazy-slices *test-labels*(range 0 100)) )))
      (format t "--------------------------~%")
	 
	  (model-compile model :loss #'categorial-cross-entropy :optimizer optimizer :metrics (list #'categorial-accuracy #'binary-accuracy ));;
      (model-summary model)
      (fit model train-input-data train-label-data val-input-data val-label-data :epochs 30 :batch-size 64)
	  ;;(save-weights model "mnist-weights/")
	  
	  (predict model (compute (lazy-slices train-input-data (range 0 1))))   
	  (load-weights model "mnist-weights/")
	  
	   
      (format t "check test data~%")
    ;; (loop for i from 0 below 2000 do
      ;;  (check-test-data model i))
	 
		(print (compute (binary-accuracy  (lazy-slices *test-labels* (range 2000)) (lazy-array (predict model (compute (lazy-slices (lazy #'/ *test-images* 255.0) (range 2000))))))))
		
		))



;;(train-mnist)

(defclass test-model (model)
  ((dense1 :accessor dense1)
  (dense2 :accessor dense2)
  (dense3 :accessor dense3)
 (dense4 :accessor dense4)
 (dense5 :accessor dense5)
  ))

(defmethod initialize-instance :after ((model test-model) &rest args)
  

  )

(defmethod forward ((model test-model) input)
(let ((dense1 (create-layer 'dense-layer model :out-features 1000 :activation #'relu))
      (dense2 (create-layer 'dense-layer model :out-features 1000 :activation #'relu)))
      (call dense2 (call dense1 input))))



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
		  (input (compute (glorot-uniform :shape (~ 5 ~ 1000) :fan-in 1000 :fan-out 1000)))
		  (output (compute (glorot-uniform :shape (~ 5 ~ 1000) :fan-in 1000 :fan-out 1000)))
		  (weights (model-weights model))
		  )
    (model-compile model :loss #'mse :optimizer optimizer)

	(fit model input output input output :epochs 10 :batch-size 5)
	
	))

;;(testmain)

