(define (domain blocks-world)
  (:requirements :strips :typing)
  
  (:types block)
  
  (:predicates
    (Nneighbor ?x - block ?y - block)  ; block x is the north neighbor of block y
    (Eneighbor ?x - block ?y - block)  ; block x is the East neighbor of block y
    (Sneighbor ?x - block ?y - block)  ; block x is the South neighbor of block y
    (Wneighbor ?x - block ?y - block)  ; block x is the West neighbor of block y
    (Nclear ?x - block); north of block x is clear
    (Eclear ?x - block); east of block x is clear
    (Sclear ?x - block); south of block x is clear
    (Wclear ?x - block); west of block x is clear
    (longDiag ?x - block ?y - block)   ; block x is along the long NE diagonal of block y
    (longDiagClear ?x - block)         ; block x has no other block placed along  its long NE diagonal 
    (handempty) ; robot hand is empty

  )
  
  (:action pick
    :parameters (?x - block)
    :precondition (and (clear ?x) (ontable ?x) (handempty)) ; block x is clear and on table and hand is empty
    :effect (and (holding ?x) (not (clear ?x)) (not (ontable ?x)) (not (handempty))) ; x is held and not on table and hand is not empty
  )

  (:action placeNorth
    :parameters (?x - block ?y - block)
    :precondition (and (holding ?x) (Nclear ?y)) ; block y is clear on the north side and hand is full
    :effect (and (Nneighbor ?x ?y) (not (Nclear ?y)) (handempty) (Sneighbor ?y ?x) (not (Sclear ?x))) ; x is north of y, hand is clear, opposing block sides are not clear
  )
    
  (:action placeEast
    :parameters (?x - block ?y - block)
    :precondition (and (holding ?x) (Eclear ?y)) ; block y is clear on the east side and hand is full
    :effect (and (Eneighbor ?x ?y) (not (Eclear ?y)) (handempty) (Wneighbor ?y ?x) (not (Wclear ?x))) ; x is east of y, hand is clear, opposing block sides are not clear
  )
  
  (:action placeSouth
    :parameters (?x - block ?y - block)
    :precondition (and (holding ?x) (Sclear ?y)) ; block y is clear on the south side and hand is full
    :effect (and (Sneighbor ?x ?y) (not (Sclear ?y)) (handempty) (Nneighbor ?y ?x) (not (Nclear ?x))) ; x is south of y, hand is clear, opposing block sides are not clear
  )
  
  (:action placeWest
    :parameters (?x - block ?y - block)
    :precondition (and (holding ?x) (Wclear ?y)) ; block y is clear on the west side and hand is full
    :effect (and (Wneighbor ?x ?y) (not (Wclear ?y)) (handempty) (Eneighbor ?y ?x) (not (Eclear ?x))) ; x is west of y, hand is clear, opposing block sides are not clear
  )

  (:action placeLongDiag
    :parameters (?x - block ?y - block)
    :precondition (and (holding ?x) (not (longDiagClear ?y))) ; hand is full and no long diagonal of y
    :effect (and (handempty) (longDiag ?x ?y) (longDiagClear ?y)) ;  hand is clear, y has a full long diagonal, and x is now at long diagonal of y
  )


)