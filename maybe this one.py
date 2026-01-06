import gurobipy as gp
from gurobipy import GRB

# made up data
g = 3                 # number of groups
m = 3                 # number of machines
bp = {1: 3, 2: 2, 3: 2}  # jobs per group (sums to 7) which are randomly assigned atm bc length of all jobs is 2
bmax = max(bp.values())

P = list(range(1, g + 1))     # real groups: 1..g
I = list(range(1, g + 1))     # slots: 1..g
K = list(range(1, m + 1))     # machines: 1..m
J = list(range(1, bmax + 1))  # job index within a group (pad with dummy jobs)

# processing times t[p,j,k] = 2 for real jobs, 0 for padded dummy jobs
t = {(p, j, k): (2 if j <= bp[p] else 0) for p in P for j in J for k in K}

# sequence-dependent setup times between groups S[p,l,k] = 1 for p!=l
S = {(p, l, k): (1 if p != l else 0) for p in P for l in P for k in K}

# setup from "dummy start" group 0 to first group
S0 = {(l, k): 1 for l in P for k in K}

# model ^-^
model = gp.Model("flwgr")

# variables based on salmasi paper
# W[i,p] = 1 if group p is assigned to slot i
W = model.addVars(I, P, vtype=GRB.BINARY, name="W")

# A[i,p,l] = 1 if slot i has group p and slot i+1 has group l (transition)
A = model.addVars(range(1, g), P, P, vtype=GRB.BINARY, name="A")  # i=1..g-1

# O[i,k] = setup time before processing slot i on machine k
O = model.addVars(I, K, vtype=GRB.CONTINUOUS, lb=0.0, name="O")

# X[i,j,k] = completion time of jth (padded) job of slot i on machine k
X = model.addVars(I, J, K, vtype=GRB.CONTINUOUS, lb=0.0, name="X")

# makespan
Cmax = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name="Cmax")

# constraints (before degree of freedom)

# 1. each group assigned to exactly one slot
for p in P:
    model.addConstr(gp.quicksum(W[i, p] for i in I) == 1, name=f"assign_group_{p}")

# 2. each slot gets exactly one group
for i in I:
    model.addConstr(gp.quicksum(W[i, p] for p in P) == 1, name=f"fill_slot_{i}")

# 3 4 5 6. linearize transitions A[i,p,l] = W[i,p] and W[i+1,l]
for i in range(1, g):
    for p in P:
        for l in P:
            model.addConstr(A[i, p, l] <= W[i, p], name=f"A_ub1_{i}_{p}_{l}")
            model.addConstr(A[i, p, l] <= W[i+1, l], name=f"A_ub2_{i}_{p}_{l}")
            model.addConstr(A[i, p, l] >= W[i, p] + W[i+1, l] - 1, name=f"A_lb_{i}_{p}_{l}")

# setup time before slot 1 (from dummy start)
for k in K:
    model.addConstr(
        O[1, k] == gp.quicksum(S0[l, k] * W[1, l] for l in P),
        name=f"setup_slot1_m{k}"
    )

# setup time before slot i>1 (from previous slot group to current slot group)
for i in range(2, g + 1):
    for k in K:
        model.addConstr(
            O[i, k] == gp.quicksum(S[p, l, k] * A[i-1, p, l] for p in P for l in P),
            name=f"setup_slot{i}_m{k}"
        )

# helper: processing time of the group assigned to slot i, job j, machine k
def proc_expr(i, j, k):
    return gp.quicksum(W[i, p] * t[p, j, k] for p in P)

# 7 8 9 10 11. flowshop style completion time constraints with group setup before first job in each slot
for i in I:
    for k in K:
        prev_slot_last = 0 if i == 1 else X[i-1, bmax, k]

        model.addConstr(
            X[i, 1, k] >= prev_slot_last + O[i, k] + proc_expr(i, 1, k),
            name=f"firstjob_sameM_slot{i}_m{k}"
        )

        if k > 1:
            model.addConstr(
                X[i, 1, k] >= X[i, 1, k-1] + proc_expr(i, 1, k),
                name=f"firstjob_prevM_slot{i}_m{k}"
            )

        for j in range(2, bmax + 1):
            model.addConstr(
                X[i, j, k] >= X[i, j-1, k] + proc_expr(i, j, k),
                name=f"jobchain_sameM_slot{i}_j{j}_m{k}"
            )
            if k > 1:
                model.addConstr(
                    X[i, j, k] >= X[i, j, k-1] + proc_expr(i, j, k),
                    name=f"jobchain_prevM_slot{i}_j{j}_m{k}"
                )

# makespan definition
for k in K:
    model.addConstr(Cmax >= X[g, bmax, k], name=f"cmax_m{k}")

# objective
model.setObjective(Cmax, GRB.MINIMIZE)

# solve ):
model.optimize()

# output ):
if model.status == GRB.OPTIMAL:
    print(f"\nOptimal makespan (Cmax) = {Cmax.X:.2f}\n")

    seq = {}
    for i in I:
        for p in P:
            if W[i, p].X > 0.5:
                seq[i] = p
    print("Group sequence (slot -> group):", seq)
else:
    print("No optimal solution found.")
    