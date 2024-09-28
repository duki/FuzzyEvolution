banner = """                                                                                     
8 8888888888   8 8888         ,o888888o.    8 8888888888   b.             8  8   
8 8888         8 8888        8888     `88.  8 8888         888o.          8  8         
8 8888         8 8888     ,8 8888       `8. 8 8888         Y88888o.       8  8         
8 8888         8 8888     88 8888           8 8888         .`Y888888o.    8  8         
8 888888888888 8 8888     88 8888           8 888888888888 8o. `Y888888o. 8  8 
8 8888         8 8888     88 8888           8 8888         8`Y8o. `Y88888o8  8         
8 8888         8 8888     88 8888   8888888 8 8888         8   `Y8o. `Y8888  8         
8 8888         8 8888     `8 8888       .8' 8 8888         8      `Y8o. `Y8  8       
8 8888         8 8888        8888     ,88'  8 8888         8         `Y8o.`  8       
8 8888         8 888888888888 `8888888P'    8 888888888888 8            `Yo  8 
"""

from fuzzy_input import FuzzyInput
from fuzzy_output import FuzzyOutput
from fuzzy_rule import FuzzyRule
from fuzzy_logic_operator import FuzzyLogicOperator

import numpy as np
import matplotlib.pyplot as plt
import random

def defuzzify():
    cs = np.array([v.c for v in vrednost])
    mus = np.array([v.mu for v in vrednost])
    return cs.dot(mus) / mus.sum()

if __name__ == "__main__":
    with open('new.csv', 'w') as f:
        f.write("potrosnja,pouzdanost,vrednost\n")
        
        for i in range(50000):
            random_float_1 = random.random()
            random_float_2 = random.random()

            tmp_vrednost = (3 + random_float_1 * 12, random_float_2 * 15)

            potrosnja = []
            potrosnja.append(FuzzyInput('mala', [3,10], [1,0], tmp_vrednost[0]))
            potrosnja.append(FuzzyInput('srednja', [7,10,12,15], [0,1,1,0], tmp_vrednost[0]))
            potrosnja.append(FuzzyInput('velika', [12,15], [0,1], tmp_vrednost[0]))

            pouzdanost = []
            pouzdanost.append(FuzzyInput('visoka', [5,10], [1,0], tmp_vrednost[1]))
            pouzdanost.append(FuzzyInput('niska', [8,15], [0,1], tmp_vrednost[1]))

            vrednost = []
            vrednost.append(FuzzyOutput('mala', [7,15], [1,0]))
            vrednost.append(FuzzyOutput('srednja', [7,15,25,40], [0,1,1,0]))
            vrednost.append(FuzzyOutput('velika', [25,40], [0,1]))

            rules = []
            rules.append(FuzzyRule(potrosnja[0], pouzdanost[0], vrednost[2], FuzzyLogicOperator.AND))
            rules.append(FuzzyRule(potrosnja[0], pouzdanost[1], vrednost[1], FuzzyLogicOperator.AND))
            rules.append(FuzzyRule(potrosnja[1], pouzdanost[0], vrednost[1], FuzzyLogicOperator.AND))
            rules.append(FuzzyRule(potrosnja[1], pouzdanost[1], vrednost[1], FuzzyLogicOperator.AND))
            rules.append(FuzzyRule(potrosnja[2], pouzdanost[0], vrednost[1], FuzzyLogicOperator.AND))
            rules.append(FuzzyRule(potrosnja[2], pouzdanost[1], vrednost[0], FuzzyLogicOperator.AND))

            f.write("{},{},{}\n".format(tmp_vrednost[0], tmp_vrednost[1], defuzzify()))
