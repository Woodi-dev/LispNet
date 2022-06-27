(in-package #:lispnet)
(in-package #:lispnet.examples.vcycle)


(defun load-array (path)
  (numpy-file-format:load-array
   (asdf:system-relative-pathname (asdf:find-system "lispnet.examples") path)))
   

(defparameter *train-data* (load-array "vcycle/data/train.npy"))
(defparameter *val-data* (load-array "vcycle/data/val.npy"))

;;jump functions c_9 to c_15
(defparameter *test-data* (load-array "vcycle/data/test.npy"))

;;jump functions c_0 to c_8
(defparameter *test2-data* (load-array "vcycle/data/test2.npy"))

(defparameter *jump-data* (load-array "vcycle/data/jump.npy"))
(defparameter *jumpexp-data* (load-array "vcycle/data/jumpexp.npy"))
(defparameter *jumpcount-data* (load-array "vcycle/data/jumpcount.npy"))

(defparameter *weights-attcr* (asdf:system-relative-pathname (asdf:find-system "lispnet.examples") "vcycle/weights/att-cr/"))
(defparameter *weights-cr* (asdf:system-relative-pathname (asdf:find-system "lispnet.examples") "vcycle/weights/cr-learned/"))

(defun genLabels(data)
	(let* ((dimensions (shape-dimensions (array-shape data))))
		   (compute (lazy-reshape 0.0 (~ (nth 0 dimensions) ~ (nth 1 dimensions ) ~ (nth 2 dimensions))))))
(defun genLabels2(data)
	(let* ((dimensions (shape-dimensions (array-shape data))))
		   (compute (lazy-reshape 0.0 (~ (nth 0 dimensions) )))))		   
  
(defun stackInput(v f c)
  (lazy-stack 3 (lazy-reshape v (transform b y x to  b y x 0))
	      (lazy-reshape f (transform b y x to  b y x 0))
	      (lazy-reshape c (transform b y x to  b y x 0))))
   

	
    

(defun convergence (model input err )
  (let* ((residuum-model (make-instance 'vcycle-model :output #'forward-residuum))
	 (f (lazy-drop-axes (lazy-slices input (range 1 2) 3) 3))
	 (c (lazy-drop-axes (lazy-slices input (range 2 3) 3) 3))
	 (r0 (predict residuum-model input))
	 (r-old r0)
	 (r-new r0)
	 (q 0.0)
	 (i 1))
		 
    (loop while (or (= i 1) (and (< r-new 1d60) (< i 250) (>= r-new r-old)) (and (< r-new r-old)  (>= (/ r-new r0) err) (< i 250) )) do ;; 
	  (setf r-old r-new)
	  (setq input (compute (stackInput (predict model input) f c)))
	  (setf r-new  (predict residuum-model input))
	  (incf q (/ r-new r-old))
	  (incf i))
    (if (>= r-new r-old)
	(format t "convergence: >1~%")
	(format t "iterations: ~S convergence: ~S~%" i (/ q i)))))


(defun vcycle-main()
  ;;(setf petalisp.core:*backend* (petalisp.xmas-backend:make-xmas-backend))
  (setf petalisp.core:*backend* (petalisp.multicore-backend:make-multicore-backend))
  (setf *network-precision* 'double-float)
  (setf *random-state* (make-random-state t))
  (let*  ((att-train-model (make-instance 'attention-vcycle-model :output #'forward-train))
	  (att-calc-model (make-instance 'attention-vcycle-model :output #'forward-calc))
	  (gmg (make-instance 'vcycle-model :output #'forward-calc)))

    ;;predict with sample inputs to compile models... will be removed in the future
    (predict att-calc-model (compute (lazy-slices *train-data* (range 0 1))))	
    (model-compile att-train-model :loss #'output-loss :optimizer (make-adam :learning-rate 0.001))

    ;;fit attention dmg
    ;; (fit att-train-model *train-data* (genLabels2 *train-data*) *val-data* (genLabels2 *val-data*) :epochs 250 :batch-size 20)
    ;;(save-weights att-train-model "weights-att/")


    ;;test gmg on jump function c_0
    (format t "------ compute convergence of gmg ------~%")
    (convergence gmg (compute (lazy-slices *test2-data* (range 0 225 9))) 1e-10)
	
	
    ;;test att-cr on jump function c_0
    (format t "------ compute convergence of deep attention multi grid ------~%")
    (load-weights att-calc-model  *weights-attcr*)	
    (convergence att-calc-model  (compute (lazy-slices *test2-data* (range 0 225 9))) 1e-10) ;;


    (format t "--------- jump test ----------~%")
    (format t "--------------  attention cr --------~%")
    ;;	(load-weights att-calc-model  *weights-attcr*)	
    ;;	(loop for i from 0 below 150 do
    ;;		(format t "~S " i)
    ;;	(convergence att-calc-model (compute (lazy-slices *jump-data* (range i (1+ i)))) 1e-10))
    (format t "--------------  geometric multi grid --------~%")
    ;;	(loop for i from 0 below 150 do
    ;;		(format t "~S " i)
    ;;			(convergence gmg (compute (lazy-slices *jump-data* (range i (1+ i)))) 1e-10))
	
    (format t "-------------- testing jump count for attention dmg ----~%")
    ;;(load-weights att-calc-model  *weights-attcr*)
    ;;(loop for i from 0 below 33 do
    ;;		(format t "~S " i)
    ;;		(convergence att-calc-model (compute (lazy-slices *jumpcount-data* (range i (1+ i)))) 1e-10))
				
    (format t "-------------- testing jump exp height for attention dmg ----~%")
    ;;(load-weights att-calc-model *weights-attcr* )		
    ;;(loop for i from 0 below 257 do
    ;;(format t "~S " i)
    ;;(convergence vcycle-calc-model (compute (lazy-slices *jumpexp-data* (range i (1+ i)))) 1e-10))			
    ))



;;(vcycle-main)
