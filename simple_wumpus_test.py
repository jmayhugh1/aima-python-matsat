import pytest
from simple_wumpus import *
from agents4e import Bump

def test_init():
    kb : SimpleWumpusKB = SimpleWumpusKB(4)
    assert kb.ask_if_true(location(1, 1)) == True
    assert kb.ask_if_true(pit(1, 1)) == False
    assert kb.ask_if_true(wumpus(1, 1)) == False
    assert(0 == 0)

def test_safe():
    kb : SimpleWumpusKB = SimpleWumpusKB(4)
    print("The amount of clauses in the kb before", len(kb.clauses))
    kb.tell(pit(1, 2))
    assert kb.ask_if_true(ok_to_move(1, 2)) == False
    kb.tell(bump(1, 3))
    assert kb.ask_if_true(ok_to_move(1, 3)) == False
    print("The amount of clauses in the kb after", len(kb.clauses))
    
def test_no_wall_collision():
    kb = SimpleWumpusKB(4)
    directional_percepts = [[], [], [], [], []]
    
    
  
