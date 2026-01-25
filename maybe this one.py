import gurobipy as gp
from gurobipy import GRB
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# made up data
g = 3
m = 3
P = [1, 2, 3]          # groups
I = [1, 2, 3]          # slots
K = [1, 2, 3]          # machines

# Processing times per machine for job IDs 1..7
pt = {
    1: {1: 1, 2: 1, 3: 3, 4: 1, 5: 3, 6: 1, 7: 2},  # machine 1
    2: {1: 1, 2: 1, 3: 1, 4: 1, 5: 4, 6: 5, 7: 3},  # machine 2
    3: {1: 1, 2: 1, 3: 4, 4: 1, 5: 2, 6: 1, 7: 1},  # machine 3
}

# Group 1: jobs {1,2} on every machine
# Group 2: jobs {3,4,5} on every machine
# Group 3: jobs {6,7} on every machine
jobs_in_group = {
    1: {1: [1, 2],     2: [1, 2],     3: [1, 2]},
    2: {1: [3, 4, 5],  2: [3, 4, 5],  3: [3, 4, 5]},
    3: {1: [6, 7],     2: [6, 7],     3: [6, 7]},
}

# setup time: 1 before each group on each machine (no setup inside group)
setup = {(p, k): 1 for p in P for k in K}

# max number of jobs in any group on any machine (here = 3)
bmax = max(len(jobs_in_group[p][k]) for p in P for k in K)
Jpos = list(range(1, bmax + 1))  # positions inside group on a machine: 1..bmax

# Colors per group for plotting
group_colors = {
    1: "tab:blue",
    2: "tab:red",
    3: "tab:green",
}


# model
model = gp.Model("flwgr")

# W[i,p] = 1 if group p is assigned to slot i
W = model.addVars(I, P, vtype=GRB.BINARY, name="W")

# Z[i,p,k,a,j] = 1 if (when group p is in slot i) job a (on machine k) is assigned to position j within that group on that machine.
Z = {}
for i in I:
    for p in P:
        for k in K:
            for a in jobs_in_group[p][k]:
                for j in Jpos:
                    Z[i, p, k, a, j] = model.addVar(vtype=GRB.BINARY, name=f"Z[{i},{p},{k},{a},{j}]")

# X[i,j,k] = completion time of position j in slot i on machine k
X = model.addVars(I, Jpos, K, vtype=GRB.CONTINUOUS, lb=0.0, name="X")

# makespan
Cmax = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name="Cmax")


# group assignment constraints
for p in P:
    model.addConstr(gp.quicksum(W[i, p] for i in I) == 1, name=f"assign_group_{p}")

for i in I:
    model.addConstr(gp.quicksum(W[i, p] for p in P) == 1, name=f"fill_slot_{i}")


# internal sequencing constraints
for i in I:
    for p in P:
        for k in K:
            jobs_pk = jobs_in_group[p][k]
            n_jobs = len(jobs_pk)

            # each real job appears in exactly one position if W[i,p]=1 (else 0)
            for a in jobs_pk:
                model.addConstr(
                    gp.quicksum(Z[i, p, k, a, j] for j in Jpos) == W[i, p],
                    name=f"job_once_i{i}_p{p}_k{k}_a{a}"
                )

            # each position has at most one job if W[i,p]=1 (can be empty if fewer than bmax jobs)
            for j in Jpos:
                model.addConstr(
                    gp.quicksum(Z[i, p, k, a, j] for a in jobs_pk) <= W[i, p],
                    name=f"pos_cap_i{i}_p{p}_k{k}_j{j}"
                )

            # exactly n_jobs assigned if group is selected in that slot
            model.addConstr(
                gp.quicksum(Z[i, p, k, a, j] for a in jobs_pk for j in Jpos) == n_jobs * W[i, p],
                name=f"count_i{i}_p{p}_k{k}"
            )

# helper: processing time expression at (slot i, machine k, position j)
def proc_pos_expr(i, k, j):
    expr = gp.LinExpr()
    for p in P:
        for a in jobs_in_group[p][k]:
            expr += pt[k][a] * Z[i, p, k, a, j]
    return expr

# helper: setup time before slot i on machine k (depends on which group is in slot i)
def setup_expr(i, k):
    return gp.quicksum(setup[p, k] * W[i, p] for p in P)

# completion time constraints (per machine timeline)
for k in K:
    for i in I:
        prev_slot_last = 0 if i == 1 else X[i - 1, bmax, k]

        # first position in slot i
        model.addConstr(
            X[i, 1, k] >= prev_slot_last + setup_expr(i, k) + proc_pos_expr(i, k, 1),
            name=f"firstpos_i{i}_k{k}"
        )

        # chain positions within slot i
        for j in range(2, bmax + 1):
            model.addConstr(
                X[i, j, k] >= X[i, j - 1, k] + proc_pos_expr(i, k, j),
                name=f"chain_i{i}_k{k}_j{j}"
            )

# makespan definition
for k in K:
    model.addConstr(Cmax >= X[g, bmax, k], name=f"Cmax_k{k}")

model.setObjective(Cmax, GRB.MINIMIZE)

# 3) solve
model.optimize()

if model.status != GRB.OPTIMAL:
    print("No optimal solution found. Status:", model.status)
    raise SystemExit

print(f"\nOptimal makespan (Cmax) = {Cmax.X:.2f}")

# extract group order
seq = {}
for i in I:
    for p in P:
        if W[i, p].X > 0.5:
            seq[i] = p
            break
print("Group order (slot -> group):", seq)

# 4) BUILD SCHEDULE FOR PLOTTING
# bars entries: (machine_index0, start, dur, label, group)
bars = []

for k in K:
    for i in I:
        p = seq[i]

        for j in Jpos:
            chosen_a = None
            for a in jobs_in_group[p][k]:
                if Z[i, p, k, a, j].X > 0.5:
                    chosen_a = a
                    break

            if chosen_a is None:
                continue  # empty position

            dur = pt[k][chosen_a]
            end_time = X[i, j, k].X
            start_time = end_time - dur
            label = f"j{chosen_a}{k}"  # job a on machine k

            bars.append((k - 1, start_time, dur, label, p))


# 5) stupid gantt chart
fig, ax = plt.subplots(figsize=(12, 2 + 0.8 * m))
bar_height = 0.6

for machine_idx0, start, dur, label, group in bars:
    ax.barh(
        y=machine_idx0,
        width=dur,
        left=start,
        height=bar_height,
        align="center",
        edgecolor="black",
        color=group_colors.get(group, "gray"),
    )
    ax.text(
        x=start + dur / 2,
        y=machine_idx0,
        s=label,
        ha="center",
        va="center",
        fontsize=9,
        color="black",
    )

ax.set_yticks(list(range(m)))
ax.set_yticklabels([f"Machine {k}" for k in K])
ax.set_xlabel("Time")
ax.set_ylabel("Machines")
ax.set_title("Schedule (Group order + internal sequencing decided by model)")
ax.grid(True, axis="x")

# Legend for groups
legend_patches = [Patch(color=group_colors[p], label=f"Group {p}") for p in P]
ax.legend(handles=legend_patches, loc="upper right")

plt.tight_layout()
plt.show()
