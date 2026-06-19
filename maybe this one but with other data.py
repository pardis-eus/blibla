import zipfile
import gurobipy as gp
from gurobipy import GRB

# =========================
# SETTINGS
# =========================
zip_path = "GS Instances Schaller.zip"
folder = "ssu33"
instance = "D01"


# =========================
# READ SCHALLER DATA
# =========================
def find_file_in_zip(zip_path, folder, filename):
    with zipfile.ZipFile(zip_path, "r") as z:
        for name in z.namelist():
            clean = name.replace("\\", "/")
            if folder in clean and clean.endswith(filename):
                return clean
    raise FileNotFoundError(f"Could not find {filename} in {folder}")


def read_numbers_from_zip(zip_path, file_inside_zip):
    with zipfile.ZipFile(zip_path, "r") as z:
        text = z.read(file_inside_zip).decode("latin1")
    return [int(x) for x in text.split()]


def read_schaller_instance(zip_path, folder, instance):
    setup_file = find_file_in_zip(zip_path, folder, f"FAMSETUP.{instance}")
    proc_file = find_file_in_zip(zip_path, folder, f"PROCTIME.{instance}")

    setup_nums = read_numbers_from_zip(zip_path, setup_file)

    idx = 0
    g = setup_nums[idx]
    idx += 1
    m = setup_nums[idx]
    idx += 1

    P = list(range(1, g + 1))
    K = list(range(1, m + 1))
    P0 = [0] + P

    setup = {}

    for k in K:
        # initial setup row: 0 -> family
        for l in P:
            setup[0, l, k] = setup_nums[idx]
            idx += 1

        # family-to-family setup matrix
        for p in P:
            for l in P:
                val = setup_nums[idx]
                idx += 1

                if p == l:
                    setup[p, l, k] = 0
                else:
                    setup[p, l, k] = val

    proc_nums = read_numbers_from_zip(zip_path, proc_file)

    idx = 0
    pt = {k: {} for k in K}
    group_jobs = {}

    global_job = 1

    for p in P:
        num_jobs = proc_nums[idx]
        idx += 1

        group_jobs[p] = []
        family_jobs = []

        for _ in range(num_jobs):
            group_jobs[p].append(global_job)
            family_jobs.append(global_job)
            global_job += 1

        # rows = machines, columns = jobs
        for k in K:
            for job in family_jobs:
                pt[k][job] = proc_nums[idx]
                idx += 1

    return g, m, P, K, P0, pt, group_jobs, setup


g, m, P, K, P0, pt, group_jobs, setup = read_schaller_instance(
    zip_path, folder, instance
)

I = list(range(1, g + 1))

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
sum_all_setup = sum(setup[p, l, k] for p in P0 for l in P for k in K)
M = sum_all_proc + sum_all_setup + 100

for p in P:
    real_jobs = list(group_jobs[p])
    padded_jobs[p] = real_jobs + [None] * (bmax - len(real_jobs))

for p in P:
    for j in J:
        real_job = padded_jobs[p][j - 1]

        if real_job is None:
            job_label[p, j] = "DUMMY"
            for k in K:
                t[p, j, k] = 0
                tprime[p, j, k] = -M
        else:
            job_label[p, j] = real_job
            for k in K:
                t[p, j, k] = pt[k][real_job]
                tprime[p, j, k] = pt[k][real_job]

T = {
    (p, k): sum(t[p, j, k] for j in J)
    for p in P
    for k in K
}

# =========================
# SALMASI-STYLE MILP MODEL
# =========================
model = gp.Model("Salmasi_with_Schaller_data")

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

# each group assigned once
for p in P:
    model.addConstr(gp.quicksum(W[i, p] for i in I) == 1)

# each slot has one group
for i in I:
    model.addConstr(gp.quicksum(W[i, p] for p in P) == 1)

# one active transition
for i in range(0, g):
    model.addConstr(
        gp.quicksum(A[i, p, l] for p in P0 for l in P if l != p) == 1
    )

# first transition: dummy group 0 -> first family
for l in P:
    model.addConstr(A[0, 0, l] == W[1, l])

# real transitions
for i in range(1, g):
    for p in P:
        for l in P:
            if p != l:
                model.addConstr(A[i, p, l] <= W[i, p])
                model.addConstr(A[i, p, l] <= W[i + 1, l])
                model.addConstr(A[i, p, l] >= W[i, p] + W[i + 1, l] - 1)

# invalid transitions
for i in range(0, g):
    for p in P0:
        for l in P:
            if p == l or (i == 0 and p != 0):
                model.addConstr(A[i, p, l] == 0)

# setup time
for i in I:
    for k in K:
        model.addConstr(
            O[i, k] == gp.quicksum(
                A[i - 1, p, l] * setup[p, l, k]
                for p in P0
                for l in P
                if p != l
            )
        )

# completion on first machine
for i in I:
    prev = 0 if i == 1 else C[i - 1, 1]
    model.addConstr(
        C[i, 1] ==
        prev + O[i, 1] + gp.quicksum(W[i, p] * T[p, 1] for p in P)
    )

# job timing
for i in I:
    for j in J:
        for k in K:
            prevC = 0 if i == 1 else C[i - 1, k]
            model.addConstr(
                X[i, j, k] >=
                prevC + O[i, k] +
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

                    model.addConstr(
                        X[i, j, k] - X[i, q, k] + M * Y[i, j, q] >= rhs_j
                    )

                    model.addConstr(
                        X[i, q, k] - X[i, j, k] + M * (1 - Y[i, j, q]) >= rhs_q
                    )

# flowshop constraints
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

# =========================
# SOLVE
# =========================
model.optimize()

if model.status != GRB.OPTIMAL:
    print("No optimal solution found")
    print("Status:", model.status)
    exit()

print(f"\nInstance: {folder}/{instance}")
print(f"Optimal makespan = {C[g, m].X:.2f}")

seq = {}

for i in I:
    for p in P:
        if W[i, p].X > 0.5:
            seq[i] = p
            break

print("Group order:", seq)

print("\nJob order inside each selected group:")
for i in I:
    p = seq[i]
    jobs_sorted = []

    for j in J:
        if job_label[p, j] == "DUMMY":
            continue
        jobs_sorted.append((job_label[p, j], X[i, j, m].X))

    jobs_sorted.sort(key=lambda x: x[1])
    order = [job for job, _ in jobs_sorted]

    print(f"Slot {i}, Group {p}: {order}")