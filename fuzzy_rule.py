from fuzzy_input import FuzzyInput
from fuzzy_output import FuzzyOutput
from fuzzy_logic_operator import FuzzyLogicOperator

class FuzzyRule:
    def __init__(
        self,
        operand1: FuzzyInput,
        operand2: FuzzyInput,
        output: FuzzyOutput,
        operator: FuzzyLogicOperator):
        
        if operator == FuzzyLogicOperator.AND:
            output.mu = max(output.mu, min(operand1.mu, operand2.mu))
        elif operator == FuzzyLogicOperator.OR:
            output.mu = max(output.mu, max(operand1.mu, operand2.mu))
