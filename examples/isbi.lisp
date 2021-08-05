(in-package #:lispnet)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;
;;; classification of IEEE International Symposium on Biomedical Imaging 2021 data


(defun load-array (path)
  (numpy-file-format:load-array
   (asdf:system-relative-pathname (asdf:find-system "lispnet") path)))

(defparameter *train-images* (load-array "examples/isbi-data/train-images.npy"))
(defparameter *train-labels* (load-array "examples/isbi-data/train-labels.npy"))
(defparameter *test-images* (load-array "examples/isbi-data/test-images.npy"))




(defclass unet-model (model)
  ((conv1 :accessor conv1)
   (maxpool1 :accessor maxpool1)
   (tconv1 :accessor tconv1)))

(defmethod initialize-instance :after ((model unet-model) &rest initargs)
   (setf (conv1 model) (make-conv2d-layer model :in-channels 1 :out-channels 32 :padding "same" :kernel-size 3  :strides '(1 1) :activation #'relu))
  (setf (maxpool1 model) (make-maxpool2d-layer model :pool-size '(2 2)))
  (setf (tconv1 model) (make-transposed-conv2d-layer model :in-channels 32 :out-channels 1 :padding "same" :kernel-size 3  :strides '(2 2) :activation #'sigmoid))
 )

(defmethod forward ((model unet-model) input)
    (lazy-drop-axes
		 (call (tconv1 model)  	
				 (call (maxpool1 model)
					 (call (conv1 model)
                          (lazy-reshape input (transform a b c to a b c 0))))) 3))
						  
						  

(defun train-isbi()
  (let*  ((optimizer (make-adam :learning-rate 0.01))
          (model (make-instance 'unet-model)))
    (model-compile model :loss #'binary-cross-entropy :optimizer optimizer :metrics (list #'binary-accuracy)) ;;
    (let* ((train-input-data
             (compute (lazy-slices (lazy #'/ *train-images* 255.0) (range 0 20))))
           (train-label-data
             (compute (lazy-slices (lazy #'/ *train-labels* 255.0)(range 0 20)) ))

           (val-input-data
             (compute (lazy-slices (lazy #'/ *train-images* 255.0)(range 20 30))))
           (val-label-data
             (compute (lazy-slices (lazy #'/ *train-labels* 255.0)(range 20 30)) )))
      (format t "--------------------------~%")
      (model-summary model)
      (fit model train-input-data train-label-data val-input-data val-label-data :epochs 15 :batch-size 1)
      ;;(format t "check test data~%")
     )))

;;(train-isbi)


