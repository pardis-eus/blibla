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