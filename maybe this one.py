import gurobipy as gp
from gurobipy import GRB

# made up data
g = 3
m = 3
P = [1, 2, 3]
I = [1, 2, 3]
K = [1, 2, 3]
P0 = [0] + P

pt = {
    1: {1: 1, 2: 1, 3: 3, 4: 1, 5: 3, 6: 1, 7: 2},
    2: {1: 1, 2: 1, 3: 1, 4: 1, 5: 4, 6: 5, 7: 3},
    3: {1: 1, 2: 1, 3: 4, 4: 1, 5: 2, 6: 1, 7: 1},
}

# Group 1: jobs {1,2}
# Group 2: jobs {3,4,5}
# Group 3: jobs {6,7}
group_jobs = {
    1: [1, 2],
    2: [3, 4, 5],
    3: [6, 7],
}

# setup time: 1 before each group (converted to sequence form)
setup_simple = {(p, k): 1 for p in P for k in K}
setup = {(p_prev, p_next, k): setup_simple[p_next, k] for p_prev in P0 for p_next in P for k in K}

bmax = max(len(group_jobs[p]) for p in P)
J = list(range(1, bmax + 1))

# padding with dummy jobs
t = {}
tprime = {}
job_label = {}
padded_jobs = {}

sum_all_proc = sum(pt[k][a] for k in K for a in pt[k])
sum_all_setup = sum(setup[p_prev, p_next, k] for p_prev in P0 for p_next in P for k in K)
M = sum_all_proc + sum_all_setup + 100

for p in P:
    real = list(group_jobs[p])
    padded_jobs[p] = real + [None] * (bmax - len(real))

for p in P:
    for j in J:
        a = padded_jobs[p][j - 1]
        if a is None:
            job_label[p, j] = "DUMMY"
            for k in K:
                t[p, j, k] = 0
                tprime[p, j, k] = -M
        else:
            job_label[p, j] = a
            for k in K:
                t[p, j, k] = pt[k][a]
                tprime[p, j, k] = pt[k][a]

T = {(p, k): sum(t[p, j, k] for j in J) for p in P for k in K}

# model
model = gp.Model("flwgr")

W = model.addVars(I, P, vtype=GRB.BINARY, name="W")
A = model.addVars(range(0, g), P0, P, vtype=GRB.BINARY, name="A")
Y = model.addVars([(i, j, q) for i in I for j in J for q in J if j < q],
                  vtype=GRB.BINARY, name="Y")
X = model.addVars(I, J, K, lb=0.0, vtype=GRB.CONTINUOUS, name="X")
C = model.addVars(I, K, lb=0.0, vtype=GRB.CONTINUOUS, name="C")
O = model.addVars(I, K, lb=0.0, vtype=GRB.CONTINUOUS, name="O")

model.setObjective(C[g, m], GRB.MINIMIZE)

# group assignment
for p in P:
    model.addConstr(gp.quicksum(W[i, p] for i in I) == 1)

for i in I:
    model.addConstr(gp.quicksum(W[i, p] for p in P) == 1)

# adjacency
for i in range(0, g):
    model.addConstr(gp.quicksum(A[i, p, l] for p in P0 for l in P if l != p) == 1)

for l in P:
    model.addConstr(A[0, 0, l] <= W[1, l])
    model.addConstr(A[0, 0, l] >= W[1, l])

for i in range(1, g):
    for p in P:
        for l in P:
            if l != p:
                model.addConstr(A[i, p, l] <= W[i, p])
                model.addConstr(A[i, p, l] <= W[i + 1, l])
                model.addConstr(A[i, p, l] >= W[i, p] + W[i + 1, l] - 1)

for i in range(0, g):
    for p in P0:
        for l in P:
            if p == l or (i == 0 and p != 0):
                model.addConstr(A[i, p, l] == 0)

# setup
for i in I:
    for k in K:
        model.addConstr(
            O[i, k] == gp.quicksum(A[i - 1, p, l] * setup[p, l, k]
                                  for p in P0 for l in P if l != p)
        )

# completion machine 1
for i in I:
    prev = 0 if i == 1 else C[i - 1, 1]
    model.addConstr(
        C[i, 1] == prev + O[i, 1] + gp.quicksum(W[i, p] * T[p, 1] for p in P)
    )

# job timing
for i in I:
    for j in J:
        for k in K:
            prevC = 0 if i == 1 else C[i - 1, k]
            model.addConstr(
                X[i, j, k] >= prevC + O[i, k] +
                gp.quicksum(W[i, p] * tprime[p, j, k] for p in P)
            )

# sequencing inside group
for i in I:
    for j in J:
        for q in J:
            if j < q:
                for k in K:
                    rhs_j = gp.quicksum(W[i, p] * tprime[p, j, k] for p in P)
                    rhs_q = gp.quicksum(W[i, p] * tprime[p, q, k] for p in P)

                    model.addConstr(X[i, j, k] - X[i, q, k] + M * Y[i, j, q] >= rhs_j)
                    model.addConstr(X[i, q, k] - X[i, j, k] + M * (1 - Y[i, j, q]) >= rhs_q)

# flow shop
for i in I:
    for j in J:
        for k in K:
            if k >= 2:
                model.addConstr(
                    X[i, j, k] - X[i, j, k - 1] >=
                    gp.quicksum(W[i, p] * t[p, j, k] for p in P)
                )

# slot completion
for i in I:
    for k in K:
        for j in J:
            model.addConstr(C[i, k] >= X[i, j, k])

model.optimize()

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

if model.status != GRB.OPTIMAL:
    print("No optimal solution")
    exit()

print(f"\nOptimal makespan = {C[g,m].X:.2f}")

# find group order
seq = {}
for i in I:
    for p in P:
        if W[i,p].X > 0.5:
            seq[i] = p
            break

print("Group order:", seq)

# colors
group_colors = {
    1: "tab:purple",
    2: "tab:red",
    3: "tab:pink",
}

bars = []

# build schedule
for k in K:
    for i in I:
        p = seq[i]

        # sort jobs by completion time on machine k
        jobs_sorted = []
        for j in J:
            label = job_label[p, j]
            if label == "DUMMY":
                continue
            jobs_sorted.append((j, label, X[i,j,k].X))

        jobs_sorted.sort(key=lambda x: x[2])  # sort by completion time

        for j, a, end_time in jobs_sorted:
            dur = pt[k][a]
            start = end_time - dur
            label = f"j{a}"

            bars.append((k-1, start, dur, label, p))

# plot
fig, ax = plt.subplots(figsize=(12, 2 + 0.8*m))
bar_height = 0.6

for machine, start, dur, label, group in bars:
    ax.barh(machine, dur, left=start,
            height=bar_height,
            color=group_colors[group],
            edgecolor="black")

    ax.text(start + dur/2, machine, label,
            ha="center", va="center", fontsize=9)

ax.set_yticks(range(m))
ax.set_yticklabels([f"Machine {k}" for k in K])
ax.set_xlabel("Time")
ax.set_title("Flowshop Schedule")
ax.grid(True, axis="x")

legend = [Patch(color=group_colors[p], label=f"Group {p}") for p in P]
ax.legend(handles=legend)

plt.tight_layout()
plt.show()