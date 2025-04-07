from pysolve.model import Model
from pysolve.utils import is_close,round_solution

import matplotlib.pyplot as plt

def create_sim_model():
    model = Model()

    model.set_var_default(0)
    model.var('Cd', desc='Consumption goods demand by households')
    model.var('Cs', desc='Consumption goods supply')
    model.var('Gs', desc='Government goods, supply')
    model.var('Hh', desc='Cash money held by households')
    model.var('Hs', desc='Cash money supplied by the government')
    model.var('Nd', desc='Demand for labor')
    model.var('Ns', desc='Supply of labor')
    model.var('Td', desc='Taxes, demand')
    model.var('Ts', desc='Taxes, supply')
    model.var('Y', desc='Income = GDP')
    model.var('YD', desc='Disposable income of households')

    model.param('Gd', desc='Government goods, demand')
    model.param('W', desc='Wage rate')
    model.param('alpha1', desc='Propensity to consume out of income')
    model.param('alpha2', desc='Propensity to consume out of wealth')
    model.param('theta', desc='Tax rate')

    model.add('Cs = Cd')  # 3.1
    model.add('Gs = Gd')  # 3.2
    model.add('Ts = Td')  # 3.3
    model.add('Ns = Nd')  # 3.4
    model.add('YD = (W*Ns) - Ts') # 3.5
    model.add('Td = theta * W * Ns')  # 3.6, theta < 1.0
    model.add('Cd = alpha1*YD + alpha2*Hh(-1)') # 3.7, 0 < alpha2 < alpha1 < 1
    model.add('Hs - Hs(-1) =  Gd - Td')  # 3.8
    model.add('Hh - Hh(-1) = YD - Cd') # 3.9
    model.add('Y = Cs + Gs') # 3.10
    model.add('Nd = Y/W') # 3.11
    
    return model

steady_state = create_sim_model()
steady_state.set_values({'alpha1': 0.6,
                  'alpha2': 0.4,
                  'theta': 0.2,
                  'Gd': 20,
                  'W': 1})

for _ in range(100):
    steady_state.solve(iterations=100, threshold=1e-5)

    prev_soln = steady_state.solutions[-2]
    soln = steady_state.solutions[-1]
    if is_close(prev_soln, soln, atol=1e-4):
        break

prev = round_solution(steady_state.solutions[-2], decimals=1)
solution = round_solution(steady_state.solutions[-1], decimals=1)
economy_gdp = list()
for x in steady_state.solutions:
    economy_gdp.append(x['Y'])

plt.plot(list(range(len(economy_gdp))), economy_gdp)
plt.show()