(define (problem ten-blocks)
  (:domain blocks-world)
  
  (:objects r g b c m y o p w k s - block)
  
  (:init
    (ontable r) (ontable g) (ontable b) (ontable c)
    (ontable m) (ontable y) (ontable o) (ontable p)
    (ontable k) (ontable s)
    (clear r) (clear g) (clear b) (clear c)
    (clear m) (clear y) (clear o) (clear p)
    (clear k) (clear s)
    (handempty)
  )
  
  (:goal (and
    (on g r)
    (on b g)
    (on y b)
    (on m y)
    (on c m)
    (on o c)
    (on p o)
    (on k p)
    (on s k)
  ))
)
