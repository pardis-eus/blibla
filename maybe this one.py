import gurobipy as gp
from gurobipy import GRB
import numpy as np

# 1. input data (from paper)

num_groups = 2
num_machines = 2
bmax = 3

debug = True

bp = {0: 2, 1: 3}  # Number of jobs in each group (0-based indexing)

# Processing times: t[group][job][machine]
t = {
    0: {1: [3, 2], 2: [4, 1], 3: [0, 0]},
    1: {1: [2, 3], 2: [3, 2], 3: [1, 4]}
}

# Setup times between groups: s[from_group, to_group] = [M1_setup, M2_setup]
s = {
    (0, 1): [1, 2],
    (1, 0): [2, 1]
}

# 2. model

model = gp.Model("FSDGS_Makespan_Min")

# 3. variables

# W[i,p] = 1 if group p is assigned to slot i
W = model.addVars(num_groups, num_groups, vtype=GRB.BINARY, name="W")

# C[i,k] = completion time of slot i on machine k
C = model.addVars(num_groups, num_machines, vtype=GRB.CONTINUOUS, name="C")

# Cmax = makespan
Cmax = model.addVar(vtype=GRB.CONTINUOUS, name="Cmax")

# 4. constraints

# Each group assigned to one slot
for p in range(num_groups):
    model.addConstr(gp.quicksum(W[i, p] for i in range(num_groups)) == 1)

# Each slot gets one group
for i in range(num_groups):
    model.addConstr(gp.quicksum(W[i, p] for p in range(num_groups)) == 1)

# Add dummy processing time per group (sum of jobs)
total_proc_time = {}
for p in range(num_groups):
    total_proc_time[p] = [sum(t[p][j][k] for j in range(1, bp[p]+1)) for k in range(num_machines)]

# Completion time constraints
for i in range(num_groups):
    for k in range(num_machines):
        # Completion time at slot i, machine k is at least processing time of assigned group
        expr = gp.quicksum(W[i, p] * total_proc_time[p][k] for p in range(num_groups))

        if k == 0:
            model.addConstr(C[i, k] >= expr)
        else:
            model.addConstr(C[i, k] >= C[i, k-1] + expr)

# Setup time constraints between slots (when group changes)
for i in range(1, num_groups):
    for p in range(num_groups):
        for l in range(num_groups):
            if p != l and (p, l) in s:
                for k in range(num_machines):
                    setup = s[p, l][k]
                    model.addConstr(C[i, k] >= C[i-1, k] + setup * W[i-1, p] * W[i, l])

# Makespan constraint
for i in range(num_groups):
    for k in range(num_machines):
        model.addConstr(Cmax >= C[i, k])

# 5. objective

model.setObjective(Cmax, GRB.MINIMIZE)

# 6. solve

model.optimize()

# 7. Output

if model.status == GRB.OPTIMAL:
    for v in model.getVars():
        if v.X > 0.5 or v.VarName.startswith("C") or v.VarName.startswith("Cmax"):
            print(f"{v.VarName} = {v.X:.2f}")
    print("\nOptimal makespan:", model.ObjVal)
else:
    print("No optimal solution found.")
