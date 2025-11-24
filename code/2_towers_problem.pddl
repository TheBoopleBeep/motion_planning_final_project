(define (problem two-towers)
  (:domain blocks-world)
  
  (:objects r g b c m y - block)
  
  (:init
    (ontable r) (ontable g) (ontable b)
    (ontable c) (ontable m) (ontable y)
    (clear r) (clear g) (clear b)
    (clear c) (clear m) (clear y)
    (handempty)
  )
  
  (:goal (and
    (on g b)
    (on r g)
    (on m c)
    (on y m)
  ))
)