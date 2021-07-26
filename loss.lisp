(in-package #:lispnet)

;; automatic differentiator cant resolve min function. -> max(-1 * f(x))
(defun clip (x minx maxx)
  (lazy #'* -1.0 (lazy #'max (* -1 maxx) (lazy #'* -1.0 (lazy #'max minx x)))))


(defun mse (y-true y-pred)  (lazy #'/ (lazy-allreduce #'+ (lazy #'expt (lazy #'- y-pred y-true) 2))
                                  (coerce (shape-size (lazy-array-shape y-pred)) 'single-float )))

(defun mae (y-true y-pred)  (lazy #'/ (lazy-allreduce #'+ (lazy #'abs (lazy #'- y-pred y-true) ))
                                  (coerce (shape-size (lazy-array-shape y-pred)) 'single-float )))

(defun binary-cross-entropy (y-true y-pred)
  (let ((y-pred-stable (clip y-pred 0.0000001 0.9999999)))
    (lazy #'* -1.0 (lazy #'/  (lazy-allreduce #'+ (lazy #'+ (lazy #'* y-true (lazy #'log y-pred-stable))
                                                        (lazy #'* (lazy #'- 1.0 y-true) (lazy #'log (lazy #'- 1.0 y-pred-stable)))))
                         (coerce (shape-size (lazy-array-shape y-pred)) 'single-float )))))


(defun binary-accuracy (y-true y-pred)
  (let ((binary-pred (binary-decision y-pred 0.5)))
    (lazy #'/ (lazy-allreduce #'+ (lazy #'(lambda (x y) (if (= x y) 1.0 0.0)) binary-pred y-true))
          (coerce (shape-size (lazy-array-shape y-pred)) 'single-float ))))

(defun categorial-accuracy (y-true y-pred)
  (let* ((batch-size (first(shape-dimensions(lazy-array-shape y-true))))
         (sample-shape (~l (mapcar #'range (cdr(shape-dimensions(lazy-array-shape y-true))))))
         (argmax-pred (apply #'lazy-fuse (loop for i below batch-size collect
                                                                      (lazy-argmax (lazy-slices y-pred (range i (+ i 1)))) ))))
    (lazy #'/ (lazy-allreduce #'+ (lazy #'* y-true argmax-pred)) (coerce batch-size 'single-float))))

(defun categorial-accuracy-old (y-true y-pred)
  (let* ((batch-size (first(shape-dimensions(lazy-array-shape y-true))))
         (sample-shape (~l (mapcar #'range (cdr(shape-dimensions(lazy-array-shape y-true))))))
         (argmax-pred (apply #'lazy-fuse (loop for i below batch-size collect
                                                                      (lazy-reshape(lazy-argmax (lazy-slice y-pred i)) (~ i (+ i 1) ~s sample-shape))))))
    (lazy #'/ (lazy-allreduce #'+ (lazy #'* y-true argmax-pred)) (coerce batch-size 'single-float))))


