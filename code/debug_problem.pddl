(define (problem special-tower-blocks)
  (:domain special-tower-blocks-world)

  (:objects r1 g1 b1 o1 y1 r2 g2 b2 o2 y2 - block)

  (:init
    (oriented r1)
    (oriented g1)
    (oriented b1)
    (oriented o1)
    (oriented y1)
    (oriented r2)
    (oriented g2)
    (oriented b2)
    (oriented o2)
    (oriented y2)
    (handempty)
  )

  (:goal (and
    (oriented r1)
    (relative b1 r1)
  ))
)
