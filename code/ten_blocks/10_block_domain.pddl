(define (domain blocks-world)
  (:requirements :strips :typing)
  
  (:types block)
  
  (:predicates
    (on ?x - block ?y - block)  ; block x is on block y
    (clear ?x - block)  ; block x has nothing on top
    (ontable ?x - block)  ; block x is on the table
    (holding ?x - block)  ; robot is holding block x
    (handempty) ; robot hand is empty
  )
  
  (:action pick
    :parameters (?x - block)
    :precondition (and (clear ?x) (ontable ?x) (handempty)) ; block x is clear and on table and hand is empty
    :effect (and (holding ?x) (not (clear ?x)) (not (ontable ?x)) (not (handempty))) ; x is held and not on table and hand is not empty
  )
  
  (:action stack
    :parameters (?x - block ?y - block)
    :precondition (and (holding ?x) (clear ?y)) ; x is held and y is clear
    :effect (and (on ?x ?y) (clear ?x) (handempty) (not (holding ?x)) (not (clear ?y))) ; x is on y and y is not clear and x is no longer held and x is clear
  )

  (:action unstack
    :parameters (?x - block ?y - block)
    :precondition (and (handempty) (clear ?x) (on ?x ?y)) ; hand is empty and x is on y
    :effect (and (not (on ?x ?y)) (clear ?y) (not (handempty)) (holding ?x)) ; x is in hand and y is clear
  )

  (:action place
    :parameters (?x - block)
    :precondition (holding ?x) ; robot is holding block x
    :effect (and (ontable ?x) (clear ?x) (handempty) (not (holding ?x))) ; x is on table, x is clear, hand is empty, not holding x
  )
)
