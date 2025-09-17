from logic4e import WumpusKB, HybridWumpusAgent
from typing import *
from utils4e import Expr
from mat_sat import mat_sat


class WumpusKBMatSat(WumpusKB):
    """same as the original wumpusKB but checks for entailment using MatSat"""

    def ask_if_true(self, query):
        kb_expr: Expr = Expr("&", *self.clauses)
        return mat_sat(kb_expr & ~query) is False


# hybrid_agent_mat_sat = HybridWumpusAgent(dimentions=5, kb_class=WumpusKBMatSat)
