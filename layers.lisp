
(in-package :common-lisp-user)

(defpackage #:lispnet.layers
  (:use
   #:common-lisp
   #:petalisp
   #:lispnet.initializers
   #:lispnet.trainable-parameter
   )
  (:export
   #:softmax
   #:relu
   #:sigmoid
   #:fcn
   #:conv-2d
   ))

(in-package #:lispnet.layers)

(define-modify-macro minf (&rest numbers) min)
(define-modify-macro maxf (&rest numbers) max)


(defun lazy-multi-stack (axis arrays)
  (if (> 2 (length arrays)) (lazy-reshape (first arrays) (~s (~ 1) ~s (array-shape (first arrays))))
  (let ((combined-array (lazy-stack axis (first arrays) (second arrays))))
      (loop for index from 2 to (- (length arrays) 1) do
          (setq combined-array (lazy-stack axis combined-array (nth index arrays)))
  )
    combined-array
  )
))

(defun sigmoid(input)
	(lazy #'/ (lazy #'1+ (lazy #'exp (lazy #'- input)))))
  
(defun softmax (input)
  (let ((totals (lazy #'exp input)))
    (lazy #'/ totals (lazy-allreduce #'+ totals))))

(defun relu (input)
  (lazy #'max (coerce 0 (element-type input)) input))
  
(defun fcn (input output-shape)
  (let* ((m (shape-size output-shape))
         (n (total-size input))
         (weights
           (make-trainable-parameter
            (lazy #'/
                  (make-random-array (list n m) :element-type (element-type input))
                  (* m n))))
         (bias
           (make-trainable-parameter
            (lazy #'/
                  (make-random-array m :element-type (element-type input))
                  m))) 

         )
        ;;(lazy #'+ bias (lazy-slices (lazy-flatten input) (range 0 10)));;this simple test works for forward/backward pass					
		;;comment out following equation and use the line above to test the forward and backward pass	 
         (lazy-reduce
           #'+
           (lazy #'*
                 weights
                 (lazy-reshape (lazy-reshape (lazy-flatten input ) (transform n to n 0) )))) 
		  
    
    )
  
  )
  
 (defun conv-2d (array &key (stencil '()) (n-filters 1))
  (let* ((rank (rank array))
         (n-weights (length stencil))
         (lower-bounds (make-array rank :initial-element 0))
         (upper-bounds (make-array rank :initial-element 0))
		 (n-filters-input (first(shape-dimensions(array-shape array))))
         (d nil))
		 
    ;; Determine the dimension of the stencil.
    (loop for offsets in stencil do
          (if (null d)
              (setf d (length offsets))
              (assert (= (length offsets) d))))
    ;; Determine the bounding box of the stencil.
    (loop for offsets in stencil do
          (loop for offset in offsets
                for index from (- rank d) do
                (minf (aref lower-bounds index) offset)
                (maxf (aref upper-bounds index) offset)))
    ;; Use the bounding box to compute the shape of the result.
    (let* ((result-shape
            (~l 
               (loop for lb across lower-bounds
                     for ub across upper-bounds
                     for range in (shape-ranges (array-shape array))
					 for index from 0
                     collect
					 (if (= index 0) (range 0 n-filters)
                     (if (and (integerp lb)
                              (integerp ub))
                         (let ((lo (- (range-start range) lb))
                               (hi (- (range-end range) ub)))
                           (assert (< lo hi))
                           (range lo hi))
                         range)))))
		  (interior-shape (~l (cdr(shape-ranges result-shape))))
          (filters
            (make-trainable-parameter
             (make-random-array
              (list n-filters
			        n-weights      
                    n-filters-input)
              :element-type (element-type array)))))

									
      ;; Compute the result.
	  (lazy-collapse
	  (lazy-multi-stack 0
	  (loop for filter-index below n-filters
				collect 				
				(lazy-reshape 				
				(lazy-reduce #'+
				(apply #'lazy #'+
					(loop for offsets in stencil
						  for offset-index from 0
						  collect
						  (lazy #'*
								(lazy-reshape (lazy-slice (lazy-slice filters filter-index) offset-index) (transform A to A 0 0))
								(lazy-reshape array
									(make-transformation
									:offsets
									(cons 0 (mapcar #'- offsets)))
									(~r (range n-filters-input) ~s interior-shape)
									))))) (transform A B to 0 A B )))))
																	
		)))
								

						   
