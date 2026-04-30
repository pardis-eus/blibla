import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# =========================
# READ DATA FROM EXCEL
# =========================
excel_file = "your_excel_file.xlsx"   # change this to your Excel file name

pt_df = pd.read_excel(excel_file, sheet_name="ProcessingTimes")
groups_df = pd.read_excel(excel_file, sheet_name="Groups")
setup_df = pd.read_excel(excel_file, sheet_name="Setup")

# Sets
P = sorted(groups_df["group"].unique().tolist())
K = sorted(pt_df["machine"].unique().tolist())

g = len(P)
m = len(K)

I = list(range(1, g + 1))
P0 = [0] + P

# Processing times: pt[machine][job]
pt = {k: {} for k in K}

for _, row in pt_df.iterrows():
    k = int(row["machine"])
    a = int(row["job"])
    time = float(row["time"])
    pt[k][a] = time

# Group jobs: group_jobs[group] = [jobs]
group_jobs = {}

for p in P:
    jobs = groups_df[groups_df["group"] == p]["job"].tolist()
    group_jobs[p] = [int(j) for j in jobs]

# Simple setup time: setup_simple[group, machine]
setup_simple = {}

for _, row in setup_df.iterrows():
    p = int(row["group"])
    k = int(row["machine"])
    setup_simple[p, k] = float(row["setup"])

# Convert setup to sequence form
setup = {
    (p_prev, p_next, k): setup_simple[p_next, k]
    for p_prev in P0
    for p_next in P
    for k in K
}

# =========================
# PREPARE MODEL DATA
# =========================
bmax = max(len(group_jobs[p]) for p in P)
J = list(range(1, bmax + 1))

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

# =========================
# MODEL
# =========================
model = gp.Model("flwgr")

W = model.addVars(I, P, vtype=GRB.BINARY, name="W")
A = model.addVars(range(0, g), P0, P, vtype=GRB.BINARY, name="A")

Y = model.addVars(
    [(i, j, q) for i in I for j in J for q in J if j < q],
    vtype=GRB.BINARY,
    name="Y"
)

X = model.addVars(I, J, K, lb=0.0, vtype=GRB.CONTINUOUS, name="X")
C = model.addVars(I, K, lb=0.0, vtype=GRB.CONTINUOUS, name="C")
O = model.addVars(I, K, lb=0.0, vtype=GRB.CONTINUOUS, name="O")

model.setObjective(C[g, m], GRB.MINIMIZE)

# =========================
# CONSTRAINTS
# =========================

# Each group assigned once
for p in P:
    model.addConstr(gp.quicksum(W[i, p] for i in I) == 1)

# Each slot has one group
for i in I:
    model.addConstr(gp.quicksum(W[i, p] for p in P) == 1)

# Adjacency
for i in range(0, g):
    model.addConstr(
        gp.quicksum(A[i, p, l] for p in P0 for l in P if l != p) == 1
    )

# First slot comes from dummy group 0
for l in P:
    model.addConstr(A[0, 0, l] <= W[1, l])
    model.addConstr(A[0, 0, l] >= W[1, l])

# Transitions between real groups
for i in range(1, g):
    for p in P:
        for l in P:
            if l != p:
                model.addConstr(A[i, p, l] <= W[i, p])
                model.addConstr(A[i, p, l] <= W[i + 1, l])
                model.addConstr(A[i, p, l] >= W[i, p] + W[i + 1, l] - 1)

# Invalid transitions
for i in range(0, g):
    for p in P0:
        for l in P:
            if p == l or (i == 0 and p != 0):
                model.addConstr(A[i, p, l] == 0)

# Setup time
for i in I:
    for k in K:
        model.addConstr(
            O[i, k] == gp.quicksum(
                A[i - 1, p, l] * setup[p, l, k]
                for p in P0
                for l in P
                if l != p
            )
        )

# Completion on machine 1
first_machine = K[0]

for i in I:
    prev = 0 if i == 1 else C[i - 1, first_machine]
    model.addConstr(
        C[i, first_machine] ==
        prev + O[i, first_machine] +
        gp.quicksum(W[i, p] * T[p, first_machine] for p in P)
    )

# Job timing
for i in I:
    for j in J:
        for k in K:
            prevC = 0 if i == 1 else C[i - 1, k]
            model.addConstr(
                X[i, j, k] >= prevC + O[i, k] +
                gp.quicksum(W[i, p] * tprime[p, j, k] for p in P)
            )

# Sequencing inside group
for i in I:
    for j in J:
        for q in J:
            if j < q:
                for k in K:
                    rhs_j = gp.quicksum(W[i, p] * tprime[p, j, k] for p in P)
                    rhs_q = gp.quicksum(W[i, p] * tprime[p, q, k] for p in P)

                    model.addConstr(
                        X[i, j, k] - X[i, q, k] + M * Y[i, j, q] >= rhs_j
                    )
                    model.addConstr(
                        X[i, q, k] - X[i, j, k] + M * (1 - Y[i, j, q]) >= rhs_q
                    )

# Flowshop constraints
for i in I:
    for j in J:
        for k_index in range(1, len(K)):
            k_prev = K[k_index - 1]
            k = K[k_index]

            model.addConstr(
                X[i, j, k] - X[i, j, k_prev] >=
                gp.quicksum(W[i, p] * t[p, j, k] for p in P)
            )

# Slot completion
for i in I:
    for k in K:
        for j in J:
            model.addConstr(C[i, k] >= X[i, j, k])

# =========================
# SOLVE
# =========================
model.optimize()

if model.status != GRB.OPTIMAL:
    print("No optimal solution")
    exit()

print(f"\nOptimal makespan = {C[g, m].X:.2f}")

# =========================
# OUTPUT
# =========================
seq = {}

for i in I:
    for p in P:
        if W[i, p].X > 0.5:
            seq[i] = p
            break

print("Group order:", seq)

# =========================
# PLOT
# =========================
default_colors = [
    "tab:purple",
    "tab:red",
    "tab:pink",
    "tab:blue",
    "tab:green",
    "tab:orange",
    "tab:brown",
]

group_colors = {
    p: default_colors[index % len(default_colors)]
    for index, p in enumerate(P)
}

bars = []

for k in K:
    for i in I:
        p = seq[i]

        jobs_sorted = []

        for j in J:
            label = job_label[p, j]

            if label == "DUMMY":
                continue

            jobs_sorted.append((j, label, X[i, j, k].X))

        jobs_sorted.sort(key=lambda x: x[2])

        for j, a, end_time in jobs_sorted:
            dur = pt[k][a]
            start = end_time - dur
            label = f"j{a}"

            bars.append((K.index(k), start, dur, label, p))

fig, ax = plt.subplots(figsize=(12, 2 + 0.8 * m))
bar_height = 0.6

for machine, start, dur, label, group in bars:
    ax.barh(
        machine,
        dur,
        left=start,
        height=bar_height,
        color=group_colors[group],
        edgecolor="black"
    )

    ax.text(
        start + dur / 2,
        machine,
        label,
        ha="center",
        va="center",
        fontsize=9
    )

ax.set_yticks(range(m))
ax.set_yticklabels([f"Machine {k}" for k in K])
ax.set_xlabel("Time")
ax.set_title("Flowshop Schedule")
ax.grid(True, axis="x")

legend = [Patch(color=group_colors[p], label=f"Group {p}") for p in P]
ax.legend(handles=legend)

plt.tight_layout()
plt.show()