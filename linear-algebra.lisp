(in-package :common-lisp-user)
(in-package #:lispnet)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;
;;; Matrix Utilities

(defun coerce-to-matrix (x)
  (setf x (lazy-array x))
  (trivia:ematch (lazy-array-shape x)
    ((~)
     (lazy-reshape x (~ 1 ~ 1)))
    ((~l (list range))
     (lazy-reshape x (~ (range-size range) ~ 1)))
    ((~l (list range-1 range-2))
     (lazy-reshape x (~ (range-size range-1) ~ (range-size range-2))))))

(defun coerce-to-scalar (x)
  (setf x (lazy-array x))
  (trivia:ematch (lazy-array-shape x)
    ((~) x)
    ((~ i 1+i)
     (unless (= (1+ i) 1+i)
       (trivia.fail:fail))
     (lazy-reshape x (make-transformation :input-mask (vector i) :output-rank 0)))
    ((~ i 1+i ~ j 1+j)
     (unless (and (= (1+ i) 1+i)
                  (= (1+ j) 1+j))
       (trivia.fail:fail))
     (lazy-reshape x (make-transformation :input-mask (vector i j) :output-rank 0)))))

(trivia:defpattern matrix (m n)
  (alexandria:with-gensyms (it)
    `(trivia:guard1 ,it (lazy-array-p ,it)
                    (lazy-array-shape ,it) (~ ,m ~ ,n))))

(trivia:defpattern square-matrix (m)
  (alexandria:with-gensyms (g)
    `(matrix (and ,m ,g) (= ,g))))

(trivia:defun-match matrix-p (object)
  ((matrix _ _) t)
  (_ nil))

(trivia:defun-match square-matrix-p (object)
  ((square-matrix _) t)
  (_ nil))

(deftype matrix ()
  '(satisfies matrix-p))

(deftype square-matrix ()
  '(satisfies square-matrix-p))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;
;;; Linear Algebra Subroutines



(declaim (inline δ))
(defun δ (i j)
  (declare (type integer i j))
  (if (= i j) 1 0))

(defun eye (m &optional (n m))
  (let ((shape (~ m ~ n)))
    (lazy #'δ
          (lazy-shape-indices shape 0)
          (lazy-shape-indices shape 1))))

(defun transpose (x)
  (lazy-reshape
   (coerce-to-matrix x)
   (transform m n to n m)))

(defun dot (x y)
  (coerce-to-scalar
   (matmul
    (transpose x)
    (coerce-to-matrix y))))

(defun l2norm (x)	
	(lazy #'sqrt (lazy-allreduce #'+ (lazy #'* x x))))

(defun asum (x)
  (coerce-to-scalar
   (lazy-reduce #'+ (lazy #'abs (coerce-to-matrix x)))))

(defun max* (x)
  (lazy-reduce
   (lambda (lv li rv ri)
     (if (> lv rv)
         (values lv li)
         (values rv ri)))
     x (lazy-array-indices x)))

(defun matmul (A B)
  (lazy-reduce #'+
               (lazy #'*
                     (lazy-reshape A (transform m n to n m 0))
                     (lazy-reshape B (transform n k to n 0 k)))))


