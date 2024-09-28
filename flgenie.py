banner = """                                                                                     
8 8888888888   8 8888         ,o888888o.    8 8888888888   b.             8  8 8888 8 8888888888   
8 8888         8 8888        8888     `88.  8 8888         888o.          8  8 8888 8 8888         
8 8888         8 8888     ,8 8888       `8. 8 8888         Y88888o.       8  8 8888 8 8888         
8 8888         8 8888     88 8888           8 8888         .`Y888888o.    8  8 8888 8 8888         
8 888888888888 8 8888     88 8888           8 888888888888 8o. `Y888888o. 8  8 8888 8 888888888888 
8 8888         8 8888     88 8888           8 8888         8`Y8o. `Y88888o8  8 8888 8 8888         
8 8888         8 8888     88 8888   8888888 8 8888         8   `Y8o. `Y8888  8 8888 8 8888         
8 8888         8 8888     `8 8888       .8' 8 8888         8      `Y8o. `Y8  8 8888 8 8888         
8 8888         8 8888        8888     ,88'  8 8888         8         `Y8o.`  8 8888 8 8888         
8 8888         8 888888888888 `8888888P'    8 888888888888 8            `Yo  8 8888 8 888888888888
"""

from fuzzy_input import FuzzyInput
from fuzzy_output import FuzzyOutput
from fuzzy_rule import FuzzyRule
from fuzzy_logic_operator import FuzzyLogicOperator

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def defuzzify():
    cs = np.array([v.c for v in vrednost])
    mus = np.array([v.mu for v in vrednost])
    return cs.dot(mus) / mus.sum()

if __name__ == "__main__":

    potrosnja = []
    potrosnja.append(FuzzyInput('mala potrosnja', [3,10], [1,0], 9))
    potrosnja.append(FuzzyInput('srednja potrosnja', [7,10,12,15], [0,1,1,0], 9))
    potrosnja.append(FuzzyInput('velika potrosnja', [12,15], [0,1], 9))

    pouzdanost = []
    pouzdanost.append(FuzzyInput('visoka pouzdanost', [5,10], [1,0], 8))
    pouzdanost.append(FuzzyInput('niska pouzdanost', [8,15], [0,1], 8))

    vrednost = []
    vrednost.append(FuzzyOutput('mala vrednost', [7,15], [1,0]))
    vrednost.append(FuzzyOutput('srednja vrednost', [7,15,25,40], [0,1,1,0]))
    vrednost.append(FuzzyOutput('velika vrednost', [25,40], [0,1]))

    rules = []
    FuzzyRule(potrosnja[0], pouzdanost[0], vrednost[2], FuzzyLogicOperator.AND)
    FuzzyRule(potrosnja[0], pouzdanost[1], vrednost[1], FuzzyLogicOperator.AND)
    FuzzyRule(potrosnja[1], pouzdanost[0], vrednost[1], FuzzyLogicOperator.AND)
    FuzzyRule(potrosnja[1], pouzdanost[1], vrednost[1], FuzzyLogicOperator.AND)
    FuzzyRule(potrosnja[2], pouzdanost[0], vrednost[1], FuzzyLogicOperator.AND)
    FuzzyRule(potrosnja[2], pouzdanost[1], vrednost[0], FuzzyLogicOperator.AND)


    for i in range(len(vrednost)):
        print(vrednost[i].mu)

    print("defuzzification result:", defuzzify())

    figure, axis = plt.subplots(2, 2)

    for i, input_data in enumerate(potrosnja):
        xs, ys = potrosnja[i].xs, potrosnja[i].ys
        axis[0, 0].plot(xs, ys, label=f"Fuzzy {['mala', 'srednja', 'velika'][i]}")



    for i, input_data in enumerate(pouzdanost):
        xs, ys = pouzdanost[i].xs, pouzdanost[i].ys
        axis[0, 1].plot(xs, ys, label=f"Fuzzy {['visoka', 'niska'][i]}")



    for i, input_data in enumerate(vrednost):
        xs, ys = vrednost[i].xs, vrednost[i].ys
        axis[1, 0].plot(xs, ys, label=f"Fuzzy {['mala', 'srednja', 'velika'][i]}")


    axis[0, 0].grid(True)
    axis[0, 1].grid(True)
    axis[1, 0].grid(True)
    axis[1, 1].grid(False)

    axis[0, 0].set_title('Potrosnja')
    axis[0, 1].set_title('Pouzdanost')
    axis[1, 0].set_title('Vrednost')
    axis[1, 1].set_title('Blank')
    
    plt.xlabel('Input')
    plt.ylabel('Membership Function')
    plt.title('Fuzzy Graph')
    plt.legend()
    plt.grid(True)        
    plt.tight_layout()
    plt.show()
