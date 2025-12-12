(define (domain adjacent-blocks-world-noclear)
  (:requirements :strips :typing)
  
  (:types block)
  
  (:predicates
    (Nneighbor ?x - block ?y - block)  ; block x is the north neighbor of block y
    (Eneighbor ?x - block ?y - block)  ; block x is the East neighbor of block y
    (Sneighbor ?x - block ?y - block)  ; block x is the South neighbor of block y
    (Wneighbor ?x - block ?y - block)  ; block x is the West neighbor of block y
    (longDiag ?x - block ?y - block)   ; block x is along the long NW diagonal of block y
    (handempty) ; robot hand is empty
    (holding ?x - block)  ; robot is holding block x
    (placed ?x - block)   ; block has been placed in final position

  )
  
  (:action pick
    :parameters (?x - block)
    :precondition (and (handempty)) ; hand is empty
    :effect (and (holding ?x) (not (handempty))) ; x is held, hand is not empty
  )
  (:action place
    :parameters (?x - block)
    :precondition (holding ?x)  ; holding x
    :effect (and (handempty) (not (holding ?x)) (placed ?x))  ; x not held, hand is empty
  )


  (:action placeNorth
    :parameters (?x - block ?y - block)
    :precondition (and (holding ?x) (placed ?y)) ; hand is full
    :effect (and (Nneighbor ?x ?y) (handempty) (not (holding ?x)) (Sneighbor ?y ?x) (placed ?x)) ; x is north of y, hand is clear
  )
    
  (:action placeEast
    :parameters (?x - block ?y - block)
    :precondition (and (holding ?x) (placed ?y)) ; hand is full
    :effect (and (Eneighbor ?x ?y) (handempty) (not (holding ?x)) (Wneighbor ?y ?x) (placed ?x)) ; x is east of y, hand is clear
  )
  
  (:action placeSouth
    :parameters (?x - block ?y - block)
    :precondition (and (holding ?x) (placed ?y)) ; hand is full
    :effect (and (Sneighbor ?x ?y) (handempty) (not (holding ?x)) (Nneighbor ?y ?x) (placed ?x)) ; x is south of y, hand is clear
  )
  
  (:action placeWest
    :parameters (?x - block ?y - block)
    :precondition (and (holding ?x) (placed ?y)) ; hand is full
    :effect (and (Wneighbor ?x ?y) (handempty) (not (holding ?x)) (Eneighbor ?y ?x) (placed ?x)) ; x is west of y, hand is clear
  )

  (:action placeLongDiag
    :parameters (?x - block ?y - block)
    :precondition (and (holding ?x) (placed ?y)) ; hand is full
    :effect (and (handempty) (not (holding ?x)) (longDiag ?x ?y) (placed ?x)) ; hand is clear, x is now at long diagonal of y
  )


)
