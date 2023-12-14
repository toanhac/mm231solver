import numpy as np
import pandas as pd
import gurobipy as gp
from gurobipy import GRB, quicksum

# Reset the seed for the random number generator
np.random.seed(np.random.randint(0, 2**31 - 1))

dataframe = pd.read_excel('data.xlsx')
# Sets
n = range(1, int(dataframe.iloc[0, 0])+1)
m = range(1, int(dataframe.iloc[0, 1])+1)
nb_of_scenarios = int(dataframe.iloc[0, 2])
scenData = dataframe['Density of each Scenarios']
scen = []
for i in range(nb_of_scenarios):
    scen.append(f's{i + 1}')

# Parameters
a = {(i, j): np.random.randint(0, 3) for i in n for j in m}
d = {(k, i): np.random.binomial(10, 0.5) for k in scen for i in n}
b = {j: np.random.randint(5, 30) for j in m}
s = {j: np.random.randint(1, b[j] + 1) for j in m}
q = {i: np.random.randint(150, 300) for i in n}
l = {i: np.random.randint(5, q[i] - 10) for i in n}
p = {f's{s + 1}' : scenData[s] for s in range(nb_of_scenarios)}

print("----- Input Data -----")
print("n =", dataframe.iloc[0, 0])
print("m =", dataframe.iloc[0, 1])
print("Number of Scenarios = ", nb_of_scenarios)
print("Density of each Scenarios:")
for i in range(len(scenData)):
    print(scenData[i])
print("----------------------")
# Create a linear programming model
model = gp.Model("StochasticProduction")
model.setParam('OutputFlag', 1)  # Telling gurobi not to be verbose
model.params.logtoconsole = 0
# Decision Variables with Bounds
x = model.addVars(m, vtype=GRB.INTEGER, name="x", lb=0)
y = model.addVars(scen, m, vtype=GRB.INTEGER, name="y", lb=0)
z = model.addVars(scen, n, vtype=GRB.INTEGER, name="z", lb=0)

# Objective Function
obj = quicksum(b[j] * x[j] for j in m) + quicksum(
    p[k] * (l[i] * z[k, i] - q[i] * z[k, i]) for k in scen for i in n
) - quicksum(p[k] * s[j] * y[k, j] for k in scen for j in m)

model.setObjective(obj, GRB.MINIMIZE)

# Constraints
balance_constraints = {}
for k in scen:
    for j in m:
        balance_constraints[k, j] = model.addConstr(
            y[k, j] == x[j] - quicksum(a[i, j] * z[k, i] for i in n),
            name=f"Balance_{k}_{j}",
        )

demand_constraints = {}
for k in scen:
    for i in n:
        demand_constraints[k, i] = model.addConstr(
            z[k, i] <= d[k, i], name=f"Demand_{k}_{i}"
        )

# Solve the problem
model.optimize()

# Print the results
if model.status == GRB.Status.OPTIMAL:
    print("Optimal solution found.")
    print("Total Cost =", model.objVal)

elif model.status == GRB.Status.INFEASIBLE:
    print("The model is infeasible.")
elif model.status == GRB.Status.UNBOUNDED:
    print("The model is unbounded.")
else:
    print("Optimization ended with status", model.status)

file_path = "result.txt"
# Open the file in write mode
with open(file_path, 'w') as file:
    # Write the status of the optimization
    file.write("Optimization Status: ")
    if model.status == GRB.Status.OPTIMAL:
        file.write("Optimal solution found.\n")
        file.write(f"Total Cost = {model.objVal}\n")
    elif model.status == GRB.Status.INFEASIBLE:
        file.write("The model is infeasible.\n")
    elif model.status == GRB.Status.UNBOUNDED:
        file.write("The model is unbounded.\n")
    else:
        file.write(f"Optimization ended with status {model.status}\n")

    # Write decision variable values
    file.write("\nDecision Variable Values:\n")
    file.write("x:\n")
    for j in m:
        file.write(f"x[{j}] = {int(x[j].x)}\n")  # Convert to int to remove decimal part

    file.write("\ny:\n")
    for k in scen:
        for j in m:
            file.write(f"y[{k}, {j}] = {int(y[k, j].x)}\n")

    file.write("\nz:\n")
    for k in scen:
        for i in n:
            file.write(f"z[{k}, {i}] = {int(z[k, i].x)}\n")
# Print a message indicating where the results are saved
print(f"Results written to {file_path}")