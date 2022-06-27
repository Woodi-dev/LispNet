(in-package #:lispnet)




(defvar *network-precision* 'single-float)

(defclass model-backend()
  ((layers
    :accessor layers
    :initform '())
   (parameters
    :accessor parameters
    :initform '())
   (parameter-pointer
    :accessor parameter-pointer
    :initform -1)
   (running
    :accessor running
    :initform nil)
   (compiled
    :accessor compiled
    :initform nil)
   (network-train
    :accessor network-train
    :initform nil)
   (network-train-rest
    :accessor network-train-rest
    :initform nil)
   (network-val
    :accessor network-val
    :initform nil)
   (network-val-rest
    :accessor network-val-rest
    :initform nil)))

(defmethod reset-pointer ((backend model-backend))
  (setf (parameter-pointer backend) (1- (length (parameters backend)))))
	
(defmethod reset-layers ((backend model-backend))
  (setf (layers backend) '()))

(defun create-network-train (model input-shape)
  (reset-pointer (model-backend model))
  (reset-layers (model-backend model))
  (let* ((input-parameter (make-unknown :shape input-shape :element-type *network-precision*))
	 (forward (funcall (output model) model input-parameter))
	 (model-weights (model-weights model))
         (trainable-parameters (loop for parameter in model-weights
				     when (trainable parameter)
				     collect parameter))
	 (label-parameter (make-unknown :shape (lazy-array-shape forward)
                                        :element-type *network-precision*))
         (lossfunc (list (funcall (model-loss model) label-parameter forward)))
	 (gradient (differentiator lossfunc (list (lazy-reshape 1.0 (~)))))
         (metrics (loop for metric in (metrics model) collect
                        (funcall metric label-parameter forward))))
    (list (apply #'make-network (append (loop for trainable-parameter in trainable-parameters
                                              collect
                                              (funcall gradient (weights trainable-parameter))) lossfunc metrics))
	  input-parameter label-parameter)))
												
(defun create-network-val (model input-shape)
  (reset-pointer (model-backend model))
  (reset-layers (model-backend model))
  (let* ((input-parameter (make-unknown :shape input-shape :element-type *network-precision*))
	 (forward (funcall (output model) model input-parameter))
	 (model-weights (model-weights model))
         (trainable-parameters (loop for parameter in model-weights
				     when (trainable parameter)
				     collect parameter))
	 (label-parameter (make-unknown :shape (lazy-array-shape forward)
                                        :element-type *network-precision*))
         (lossfunc (list (funcall (model-loss model) label-parameter forward)))
         (metrics (loop for metric in (metrics model) collect
                        (funcall metric label-parameter forward))))
    (list (apply #'make-network (append lossfunc metrics))
	  input-parameter label-parameter)))	
										
 

			
(defmethod clear-weights ((backend model-backend))  
  (setf (layers backend) '())
  (setf (parameters backend) '())
  (setf (compiled backend) nil))
			
(defclass model()
  ((loss
    :initarg :loss
    :accessor model-loss
    :initform nil)
   (optimizer
    :initarg :optimizer
    :accessor model-optimizer
    :initform nil)
   (metrics
    :initarg :metrics
    :accessor metrics
    :initform '())
   (output
    :initarg :output
    :accessor output
    :initform #'forward)	
   (backend
   :accessor model-backend
   :initform (make-instance 'model-backend))))



(defmethod model-layers ((model model))
    (layers (model-backend model)))

(defmethod model-weights ((model model))
	(parameters (model-backend model)))



(defmethod compile-networks ((model model) train val batch-size)
	(let ((backend (model-backend model))
		  (sample-shape (~l (mapcar #'range (cdr (array-dimensions train)))))
		  (train-batches-size (multiple-value-list (floor (array-dimension train 0) batch-size)))
		  (val-batches-size (multiple-value-list (floor (array-dimension val 0) batch-size))))
			(when (> (first train-batches-size) 0)
				(setf (network-train backend)  (create-network-train model (~ batch-size ~s sample-shape))))
			(when (> (nth 1 train-batches-size) 0)
				(setf (network-train-rest backend) (create-network-train model (~ (nth 1 train-batches-size) ~s sample-shape))))
			(when (> (first val-batches-size) 0)
				(setf (network-val backend) (create-network-val  model(~ batch-size ~s sample-shape))))
			(when (> (nth 1 val-batches-size) 0)
				(setf (network-val-rest backend) (create-network-val model (~ (nth 1 val-batches-size) ~s sample-shape))))))
			

(defmethod model-compile ((model model) &key optimizer loss (metrics '()) &allow-other-keys)
        (setf (model-loss model) loss)
        (setf (model-optimizer model) optimizer)
        (setf (metrics model) metrics))


(defmethod compile-parameters ((model model) sample-shape)
  (setf (compiled (model-backend model)) nil)
  (setf (parameters (model-backend model)) '())
  (setf (layers (model-backend model)) '())
  (let* ((input-parameter (make-unknown :shape (~ 1 ~s sample-shape) :element-type *network-precision*))
         ;; Generate a computation graph to trigger constructor calls in forward pass
	 (graph (funcall (output model) model input-parameter)))		 
    ;; Initialize layer weights
    (loop for layer in (model-layers model) do
          (layer-compile layer))
    ;; Initialize weights which are not initialized by a layer
    (loop for parameter in (parameters (model-backend model)) do
	  (when (null (weights-value parameter))
            (setf (weights-value parameter) (init-weights :shape (weights-shape parameter) :mode #'uniform))))

    (reset-pointer (model-backend model))		
    (setf (compiled (model-backend model)) t)))
		


(defgeneric forward (model input))

(defmethod fit((model model) train-input-data train-label-data val-input-data val-label-data &key (epochs 10) (batch-size 10) (early-stop nil) (early-stop-delta 0))
  (let* ((train-input-data-length (array-dimension train-input-data 0))
         (train-label-data-length (array-dimension train-label-data 0))
         (val-input-data-length (array-dimension val-input-data 0))
         (val-label-data-length (array-dimension val-label-data 0))
         (sample-shape (~l (mapcar #'range (cdr (array-dimensions train-input-data)))))
	 (epoch-train-losses '())
	 (epoch-val-losses '())
	 (epoch-train-metrices '())
	 (epoch-val-metrices '())
	 (best-epoch 1)
	 (best-weights nil))
    (when (not (and (model-optimizer model) (model-loss model))) 
      (error "Model needs to be compiled"))
    (assert (= train-input-data-length train-label-data-length))
    (assert (= val-input-data-length val-label-data-length))
    ;; Initialize model weights
    (when (not (compiled (model-backend model)))
      (compile-parameters model sample-shape))
    ;; Compile networks
    (compile-networks model train-input-data val-input-data batch-size)
    ;; Initialize optimizers
    (optimizer-compile (model-optimizer model) :model model)	
    (format t "~%Train on ~d samples~%" train-input-data-length)
    (loop for epoch from 1 to epochs when (or (null early-stop) (<= epoch (+ best-epoch early-stop))) do
          (format t "Epoch ~d/~d~%" epoch epochs)
          (let ((batch-train-losses '())
                (batch-val-losses '())
                (metrics-train (loop for i below (list-length (metrics model)) collect '()))
                (metrics-val (loop for i below (list-length (metrics model)) collect '()))
                (time-start (get-internal-real-time)))	  
            ;;Training
            (loop for offset below train-input-data-length by batch-size
                  for batch from 0 do
			
                  ;; (format  t "Batch: ~S~%" batch)
				
                  (let* ((full-batches (floor train-input-data-length batch-size))
			 (batch-range (range offset (min train-input-data-length (+ offset batch-size))))
                         (batch-data (lazy-slices train-input-data batch-range))
                         (batch-labels (lazy-slices train-label-data batch-range))
                         (batch-input (compute (lazy-collapse batch-data)))
                         (batch-output (compute (lazy-collapse batch-labels)))
			 (network (if (< batch full-batches) (network-train (model-backend model)) (network-train-rest (model-backend model)))))
				 	
                    (multiple-value-bind (batch-loss metrics)
			(train-test model :network network :loss (model-loss model) :optimizer (model-optimizer model) :mode "train" :batch-input batch-input :batch-output batch-output)
				   
                      (setq batch-train-losses (append batch-train-losses (list  batch-loss)))
                      (loop for metric in metrics
                            for i from 0 do
                            (setf (nth i metrics-train) (list* metric (nth i metrics-train)))))))
					 
											

            ;; Validation
            (loop for offset below val-input-data-length by batch-size
                  for batch from 0 do
                  ;;    (format  t "Batch: ~S~%" batch)
                  (let* ((full-batches (floor val-input-data-length batch-size))
			 (batch-range (range offset (min val-input-data-length (+ offset batch-size))))
                         (batch-data (lazy-slices val-input-data batch-range))
                         (batch-labels (lazy-slices val-label-data batch-range))
                         (batch-input (compute (lazy-collapse batch-data)))
                         (batch-output (compute (lazy-collapse batch-labels)))
                         (network (if (< batch full-batches) (network-val (model-backend model)) (network-val-rest (model-backend model)))))

                    (multiple-value-bind (batch-loss metrics)
                        (train-test model :network network :loss (model-loss model) :optimizer (model-optimizer model) :mode "test" :batch-input batch-input :batch-output batch-output)
                      (setq batch-val-losses (append  batch-val-losses (list batch-loss)))
                      (loop for metric in metrics
                            for i from 0 do
                            (setf (nth i metrics-val) (list* metric (nth i metrics-val)))))))
	
            ;;Force garbage collection
            (trivial-garbage:gc :full t)
	

            ;;average errors and print to stdout
	    (let ((epoch-train-loss (/ (reduce #'+ batch-train-losses) (length batch-train-losses)))
		  (epoch-val-loss (/ (reduce #'+ batch-val-losses) (length batch-val-losses)))
		  (epoch-train-metrics  (loop for metric in metrics-train collect
					      (/ (reduce #'+ metric) (length metric))))
		  (epoch-val-metrics  (loop for metric in metrics-val collect
					    (/ (reduce #'+ metric) (length metric)))))
	      (format t "~Ss train_loss: ~S - val_loss: ~S" (/ (- (get-internal-real-time) time-start) (float INTERNAL-TIME-UNITS-PER-SECOND)) epoch-train-loss epoch-val-loss)
	      (when (> (length metrics-train) 0)
                (format t " - train_metrics: ")			 
		(print-list-horizontal	epoch-train-metrics)
		(format t " - val_metrics: ")
		(print-list-horizontal	epoch-val-metrics))					 																 
              (format t "~%")
	      (setf epoch-train-losses (append epoch-train-losses (list epoch-train-loss)))
	      (setf epoch-val-losses (append epoch-val-losses (list epoch-val-loss)))
	      (setf epoch-train-metrices (append epoch-train-metrices (list epoch-train-metrics)))
	      (setf epoch-val-metrices (append epoch-val-metrices (list epoch-val-metrics)))
			

	      (when (or (= epoch 1) (< (+ epoch-val-loss early-stop-delta) (nth (1- best-epoch) epoch-val-losses))) 
		(progn 
		  (setf best-epoch epoch)
		  (setf best-weights (loop for weight in (model-weights model) collect
					   (weights-value weight))))))))
    ;;set weights of best-epoch when early-stop
    (when (not (null early-stop))
      (loop for weight in (model-weights model) 
	    for best-val in best-weights do
	    (setf (weights-value weight) best-val)))
	
    ;;return metrices and losses
    (values epoch-train-losses epoch-val-losses epoch-train-metrices epoch-val-metrices)))

(defun train-test (model &rest training-data-plist &key network loss optimizer (mode "train") batch-input batch-output &allow-other-keys)
  (when (not (compiled (model-backend model))) (error "Parameters need to be compiled"))
  (setf (running (model-backend model)) t) 
  (let* ((model-weights (model-weights model))
        (trainable-parameters (loop for parameter in model-weights
									when (trainable parameter)
										collect parameter))
         (batch-loss 0)
         (metrics-values (loop for i below (list-length (metrics model)) collect 0.0))
         (gradients (loop for i below (list-length trainable-parameters) collect (lazy-reshape 0.0 (~))))) 

    ;; Assemble the arguments.
    (let ((args '())
		  (input-parameter (nth 1 network))
		  (output-parameter (nth 2 network)))
      ;; Input.
      (push input-parameter args)
      (push batch-input args)
	  (when (not (eq (model-loss model) #'output-loss))
		;; Output.
		(push output-parameter args)
		(push batch-output args))
      ;; Parameters.
      (loop for parameter in model-weights do
        (push (weights parameter) args)
        (push (weights-value parameter) args))
      ;; Forward + backward pass
      (if (string-equal mode "train")
          ;;train
          (let* ((net-out-values (apply #'call-network (nth 0 network) (reverse args))))
            (loop for i below (list-length trainable-parameters) do
			    (setf (nth i gradients)  (nth i net-out-values)))
            (setf batch-loss (compute (nth (list-length trainable-parameters) net-out-values)))
            (loop for i from 1 to (list-length (metrics model)) do
              (setf (nth (- i 1) metrics-values) (compute (nth (+ (list-length trainable-parameters) i) net-out-values))))
			  
			  )
          ;;test
          (let* ((net-out-values (apply #'call-network (nth 0 network) (reverse args))))
            (setf batch-loss (compute (first net-out-values)))
            (loop for i from 1 to (list-length (metrics model)) do
              (setf (nth (- i 1) metrics-values) (compute (nth i net-out-values)))))))
			  

  (setf (running (model-backend model)) nil) 
    (when (string-equal mode "train")
      ;;Update weights
      (update-weights optimizer :weights trainable-parameters :gradients gradients))
    ;; Return the batch loss and metrics.
    (values batch-loss metrics-values)))


(defmethod predict ((model model) input)
   ;;Initialize weights
   (let ((sample-shape (~l (mapcar #'range (cdr (array-dimensions input))))))
        (when (not (compiled (model-backend model)))
			(compile-parameters model sample-shape))
		(setf (running (model-backend model)) t)
		(reset-pointer (model-backend model))
		(reset-layers (model-backend model))

    (let* ((args '())            
              (input-parameter (make-unknown :shape (~s (array-shape input)) :element-type *network-precision*))
              (network (make-network(funcall (output model) model input-parameter))))
				  
        ;; Inputs.
         (push input-parameter args)
         (push input args)
		 ;;(print (array-shape input))
        ;; Trainable parameters.
                (loop for trainable-parameter in (model-weights model) do
                        (push (weights trainable-parameter) args)
                        (push (weights-value trainable-parameter) args))
		
			(let ((prediction (first (apply #'call-network network (reverse args)))))
					(setf (running (model-backend model)) nil)
					prediction))))

(defmethod model-weights-total((model model))
  (reduce #'+ (mapcar #'shape-size(mapcar #'weights-shape (model-weights model)))))

(defmethod model-summary ((model model) &key sample-shape)
  (when (not (compiled (model-backend model)))
    (compile-parameters model sample-shape))	
  (format t "Model summary: ~%Model ~S~%" model)
  (format t "Layers: ~S ~%"  (length(model-layers model)))
  (format t "Parameters: ~S~%" (model-weights-total model)))
					
					
(defmethod save-weights ((model model) path)
  (ensure-directories-exist path)
  (let ((weights (model-weights model)))
    (loop for weight in weights 
	  for index from 0 do
	  (numpy-file-format:store-array (weights-value weight) (format nil "~a~a.npy" path index)))))
		
	
(defmethod load-weights ((model model) path)
  (let ((weights (model-weights model)))
    (loop for weight in weights 
	  for index from 0 do
	  (setf (weights-value weight) (numpy-file-format:load-array  (format nil "~a~a.npy" path index))))))	
     
					
