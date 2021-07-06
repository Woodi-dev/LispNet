
(in-package :common-lisp-user)
(defpackage #:lispnet.network
  (:use
   #:common-lisp
   #:petalisp
   #:lispnet.layers
   #:lispnet.initializers
   #:lispnet.trainable-parameter
   #:lispnet.loss
   #:lispnet.optimizer
  )
  (:export
   #:train-test
   #:predict
   #:fit
   ))

(in-package #:lispnet.network)




	  

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;
;;; Training - Validation


(defun train-test (network output-training-data
              &rest training-data-plist
              &key loss optimizer (mode "train") &allow-other-keys
			  )
  (let* ((trainable-parameters
           (remove-if-not #'trainable-parameter-p (network-parameters network)))
         (output-parameters
           (loop for output in (network-outputs network)
                 collect (make-instance 'parameter
                                        :element-type (element-type output)
				        :shape (array-shape output))))

         (lossfunc  (loop for output in (network-outputs network)
                  for output-parameter in output-parameters
                  collect		 
                  (funcall loss output-parameter output )))
		 (loss-network 
		            (apply #'make-network lossfunc))
		 (loss-network-gradient (list(lazy #'/ (first (network-outputs loss-network)) (first (network-outputs loss-network)))))
		 (gradient (differentiator (network-outputs loss-network) loss-network-gradient))

		 
         (training-network
           (apply #'make-network
                 (nconc (loop for trainable-parameter in trainable-parameters
                        collect
                              (funcall gradient trainable-parameter)) lossfunc)))
         (n nil)
	     (losses '())
         (gradients (loop for i below (list-length trainable-parameters) collect (lazy-reshape 0.0 (~))))
         )    
    ;; Determine the training data size.
    (dolist (data output-training-data)
      (if (null n)
          (setf n (range-size (first (shape-ranges (array-shape data)))))
          (assert (= n (range-size (first (shape-ranges (array-shape data))))))))
    (alexandria:doplist (parameter data training-data-plist) ()
      (unless (symbolp parameter)
        (assert (= n (range-size (first (shape-ranges (array-shape data))))))))
    ;; Iterate over the training data.
    (loop for index below n do
          ;; Assemble the arguments.
          (let ((args '()))
            ;; Inputs.
            (alexandria:doplist (parameter data training-data-plist)
                                (unless (symbolp parameter)
                                  (push parameter args)
                                  ;; (push (lazy-slice data index) args)))
                                  (push (lazy-drop-axes (lazy-slices data (range index (+ index 1)) ) 0) args)))
            

            ;; Outputs.
            (loop for data in output-training-data
                  for output-parameter in output-parameters do
                  (push output-parameter args)
                  (push  (lazy-drop-axes (lazy-slices data (range index (+ index 1)) ) 0) args)) 

            ;; Trainable parameters.
            (dolist (trainable-parameter trainable-parameters)
              (push trainable-parameter args)
              (push (trainable-parameter-value trainable-parameter) args))
                    
            ;; Forward + backward pass
			(if (string-equal mode "train")
			;;train
            (let* ((net-out-values
                    (first (multiple-value-list
                            (apply #'call-network training-network (reverse args)))))
                   (loss-out (compute(first(last net-out-values))))
                   (new-gradients (butlast net-out-values 1))
                   )
              (loop for i below (list-length new-gradients) do
                    (setf (nth i gradients) (lazy #'+ (nth i new-gradients)
                        (nth i gradients))))

             (push loss-out losses))
			 ;;test
			(let* ((net-out-values
                    (first (multiple-value-list
                            (apply #'call-network loss-network (reverse args)))))
                   (loss-out (compute(first net-out-values)))
                   )
				   (push loss-out losses))
			 )
			 
			  
            )
          
          )

   
	(when (string-equal mode "train")
    ;;Average gradients
    (loop for i below (length gradients) do
          (setf (nth i gradients) (lazy #'/ (nth i gradients) n)

      ))
	;;Update weights  
    (update-weights optimizer :weights trainable-parameters :gradients gradients)
    )

      
    ;; Return the trained network and batch loss.
  (values network (/ (reduce #'+ losses) (length losses)) )))

	
(defun fit(network input train-input-data train-label-data val-input-data val-label-data &key (epochs 10) (batch-size 100) (loss #'mse) optimizer)
   (let ((train-input-data-length (array-dimension train-input-data 0))
		 (train-label-data-length (array-dimension train-label-data 0))
		 (val-input-data-length (array-dimension val-input-data 0))
		 (val-label-data-length (array-dimension val-label-data 0))	 
	)
    (assert (= train-input-data-length train-label-data-length))
	(assert (= val-input-data-length val-label-data-length))
    (format t "Train on ~d samples~%" train-input-data-length)
    (loop for epoch from 1 to epochs do
    (format t "Epoch ~d/~d~%" epoch epochs)
    (let ((batch-train-losses '())
		  (batch-val-losses '())
		  (time-start (get-internal-real-time))
          )
    ;;Training
    (loop for offset below train-input-data-length by batch-size do
          (let* ((batch-range (range offset (min train-input-data-length (+ offset batch-size))))
                 (batch-data (lazy-slices train-input-data batch-range))
                 (batch-labels (lazy-slices train-label-data batch-range))

                 (batch-input (compute (lazy-collapse batch-data)))
                 (batch-output (compute (lazy-collapse batch-labels)))
                 )
			
            (multiple-value-bind (trained-net batch-loss)
            (train-test network (list batch-output)
                   :loss loss :optimizer optimizer :mode "train"
                   input batch-input)
              (setq batch-train-losses (append batch-train-losses (list batch-loss)))
              ) 
            ))
	;;Validation	
	(loop for offset below val-input-data-length by batch-size do
          (let* ((batch-range (range offset (min val-input-data-length (+ offset batch-size))))
                 (batch-data (lazy-slices val-input-data batch-range))
                 (batch-labels (lazy-slices val-label-data batch-range))

                 (batch-input (compute (lazy-collapse batch-data)))
                 (batch-output (compute (lazy-collapse batch-labels)))
                 )
			
            (multiple-value-bind (net batch-loss)
            (train-test network (list batch-output)
                   :loss loss :optimizer optimizer :mode "test"
                   input batch-input)
                (setq batch-val-losses (append  batch-val-losses (list batch-loss)))
              ) 
            ))	
      (format t "~Ss train_loss: ~S - val_loss: ~S ~%"  (/ (- (get-internal-real-time) time-start) 1000.0)
													(/ (reduce #'+ batch-train-losses) (length batch-train-losses))
													(/ (reduce #'+ batch-val-losses) (length batch-val-losses))
	  )

     )
     )
          )
	network
     )


	

	
;;;;; Prediction
(defun predict (network input)

   (values-list (apply #'call-network
    network
		(loop for parameter in (network-parameters network)
			collect parameter
			collect (if (trainable-parameter-p parameter)
						(trainable-parameter-value parameter)
						input))))
)





