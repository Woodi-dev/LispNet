(in-package #:lispnet)


(defclass optimizer ()
  ((learning-rate
    :initarg :learning-rate
    :accessor learning-rate
    :initform 0.001)
	(clip-norm
	:initarg :clip-norm
	:accessor clip-norm
	:initform nil)
	(iterations 
    :accessor iterations
    :initform 1)))

(defgeneric update-weights (optimizer &key weights gradients &allow-other-keys))

(defgeneric optimizer-compile (optimizer &key model &allow-other-keys))

(defclass sgd (optimizer)
  ((momentum
   :initarg :momentum
   :accessor momentum
   :initform 0.0)
  (last-gradients
    :accessor last-gradients
    :initform '())))

(defun make-sgd (&key (learning-rate 0.001)(momentum 0.0)(clip-norm nil))
  (make-instance 'sgd :learning-rate learning-rate :momentum momentum :clip-norm clip-norm))

(defmethod optimizer-compile ((opt sgd) &key model)
  (setf (last-gradients opt) '())
  (let* ((weights (model-weights model))
		(trainable-parameters (loop for parameter in weights
									when (trainable parameter)
										collect parameter)))
  (loop for weight in trainable-parameters do
    (setf (last-gradients opt) (nconc (last-gradients opt) (list (lazy-reshape 0.0 (~))))))))

(defmethod update-weights ((opt sgd) &key weights gradients)
  (loop for weight in weights
        for gradient in gradients
        for last-gradient in (last-gradients opt)
        for i from 0 do
          (when (and (not (null (clip-norm opt))) (> (l2norm gradient) (clip-norm opt))) (setq gradient (lazy #'* (clip-norm opt) (lazy #'/ gradient (l2norm gradient)))))
          (setf (weights-value weight)
                (compute (lazy #'- (lazy #'- (weights-value weight) (lazy #'* (learning-rate opt) gradient))
                               (lazy #'* (momentum opt) last-gradient))))
          (setf (nth i (last-gradients opt)) (compute(lazy #'* (learning-rate opt) gradient))))
  (incf (iterations opt)))

(defclass adam (optimizer)
  ((beta-1
    :initarg :beta-1
    :accessor beta-1
    :initform 0.9)
   (beta-2
    :initarg :beta-2
    :accessor beta-2
    :initform 0.999)
   (epsilon
    :initarg :epsilon
    :accessor epsilon
    :initform 0.0000001)
   (m-list
    :accessor m-list
    :initform '())
   (v-list
    :accessor v-list
    :initform '())))

(defmethod optimizer-compile ((opt adam) &key model)
  (setf (m-list opt) '())
  (setf (v-list opt) '())
  (setf (iterations opt) 1)
  (let* ((weights (model-weights model))
		(trainable-parameters (loop for parameter in weights
									when (trainable parameter)
										collect parameter)))
  (loop for weight in trainable-parameters do
    (setf (m-list opt) (nconc (m-list opt) (list(lazy-reshape 0.0 (~)))))
    (setf (v-list opt) (nconc (v-list opt) (list(lazy-reshape 0.0 (~))))))))

(defun make-adam (&key (learning-rate 0.001)(beta-1 0.9) (beta-2 0.999) (clip-norm nil))
  (make-instance 'adam :learning-rate learning-rate :beta-1 beta-1 :beta-2 beta-2 :clip-norm clip-norm))

(defmethod update-weights ((opt adam) &key weights gradients)
  (loop for weight in weights
        for gradient in gradients
		for i from 0
        for m in (m-list opt)
        for v in (v-list opt) do
		  (when (and (not (null (clip-norm opt))) (> (l2norm gradient) (clip-norm opt))) (setq gradient (lazy #'* (clip-norm opt) (lazy #'/ gradient (l2norm gradient)))))
          (setf m (compute(lazy #'+ (lazy #'* (beta-1 opt) m) (lazy #'* (lazy #'- 1.0 (beta-1 opt)) gradient))))
          (setf v (compute(lazy #'+ (lazy #'* (beta-2 opt) v) (lazy #'* (lazy #'- 1.0 (beta-2 opt)) (lazy #'* gradient gradient)))))
		  (setf (nth i (m-list opt)) m)
		  (setf (nth i (v-list opt)) v)
          (let ((m-bias-corrected (lazy #'/ m (lazy #'- 1.0 (lazy #'expt (beta-1 opt) (iterations opt)))))
                (v-bias-corrected (lazy #'/ v (lazy #'- 1.0 (lazy #'expt (beta-2 opt) (iterations opt))))))
            (setf (weights-value weight)
                  (compute(lazy #'- (weights-value weight)
                                (lazy #'* (learning-rate opt)
                                      (lazy #'/ m-bias-corrected (lazy #'+ (lazy #'sqrt v-bias-corrected) (epsilon opt)))))))))
  (incf (iterations opt)))



