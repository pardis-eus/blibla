import csv
import json
import math
import os
import re
import sys
import traceback
import zipfile
from datetime import datetime

import gurobipy as gp
from gurobipy import GRB

# Matplotlib is only needed when SAVE_GANTT_CHARTS = True.

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


# ============================================================
# USER SETTINGS
# ============================================================

ZIP_PATH = "GS Instances Schaller.zip"

# Solve all D-files inside this folder.
FOLDER = "ssu33"

# Maximum solver time for EACH instance, in seconds.
TIME_LIMIT = 1800

# Stop when the relative MIP gap is 1% or smaller.
TARGET_MIP_GAP = 0.01

# Save a PNG Gantt chart for every instance that has a solution.
# Set to False for faster experiments and less disk usage.
SAVE_GANTT_CHARTS = True

# Stop the folder after this many consecutive instances have no
# feasible solution. Set to None to disable automatic stopping.
STOP_AFTER_CONSECUTIVE_NO_SOLUTION = 3

# Output directories.
RESULTS_DIRECTORY = "results"
LOG_DIRECTORY = "logs"
GANTT_DIRECTORY = "gantt_charts"

# Results file for the selected folder.
RESULTS_CSV = os.path.join(
    RESULTS_DIRECTORY,
    f"results_{FOLDER}.csv",
)


# ============================================================
# STATUS NAMES
# ============================================================

GUROBI_STATUS_NAMES = {
    GRB.LOADED: "LOADED",
    GRB.OPTIMAL: "OPTIMAL",
    GRB.INFEASIBLE: "INFEASIBLE",
    GRB.INF_OR_UNBD: "INF_OR_UNBD",
    GRB.UNBOUNDED: "UNBOUNDED",
    GRB.CUTOFF: "CUTOFF",
    GRB.ITERATION_LIMIT: "ITERATION_LIMIT",
    GRB.NODE_LIMIT: "NODE_LIMIT",
    GRB.TIME_LIMIT: "TIME_LIMIT",
    GRB.SOLUTION_LIMIT: "SOLUTION_LIMIT",
    GRB.INTERRUPTED: "INTERRUPTED",
    GRB.NUMERIC: "NUMERIC",
    GRB.SUBOPTIMAL: "SUBOPTIMAL",
    GRB.USER_OBJ_LIMIT: "USER_OBJ_LIMIT",
    GRB.WORK_LIMIT: "WORK_LIMIT",
    GRB.MEM_LIMIT: "MEM_LIMIT",
}


def get_status_name(status_code):
    """Return a readable name for a Gurobi status code."""
    return GUROBI_STATUS_NAMES.get(
        status_code,
        f"UNKNOWN_STATUS_{status_code}",
    )


# ============================================================
# FILE AND DIRECTORY HELPERS
# ============================================================

def create_output_directories():
    """Create output directories if they do not already exist."""
    os.makedirs(RESULTS_DIRECTORY, exist_ok=True)
    os.makedirs(LOG_DIRECTORY, exist_ok=True)

    if SAVE_GANTT_CHARTS:
        os.makedirs(GANTT_DIRECTORY, exist_ok=True)


def normalize_zip_path(path):
    """Convert backslashes in ZIP paths to forward slashes."""
    return path.replace("\\", "/")


def path_belongs_to_folder(path, folder):
    """
    Check whether a ZIP member belongs to the requested folder.

    This uses folder components rather than a simple substring search.
    """
    clean_path = normalize_zip_path(path)
    components = clean_path.split("/")
    return folder in components


def find_file_in_zip(zip_path, folder, filename):
    """Find one named file inside the requested ZIP folder."""
    with zipfile.ZipFile(zip_path, "r") as archive:
        for name in archive.namelist():
            clean_name = normalize_zip_path(name)

            if (
                path_belongs_to_folder(clean_name, folder)
                and clean_name.endswith(filename)
            ):
                return name

    raise FileNotFoundError(
        f"Could not find {filename} in folder {folder}"
    )


def find_instances_in_zip(zip_path, folder):
    """
    Find every instance for which a PROCTIME.Dxx file exists.

    Returns:
        ["D01", "D02", "D03", ...]
    """
    instances = set()

    pattern = re.compile(r"PROCTIME\.(D\d+)$", re.IGNORECASE)

    with zipfile.ZipFile(zip_path, "r") as archive:
        for name in archive.namelist():
            clean_name = normalize_zip_path(name)

            if not path_belongs_to_folder(clean_name, folder):
                continue

            filename = clean_name.split("/")[-1]
            match = pattern.fullmatch(filename)

            if match:
                instances.add(match.group(1).upper())

    def instance_sort_key(instance_name):
        match = re.search(r"\d+", instance_name)

        if match:
            return int(match.group())

        return instance_name

    return sorted(instances, key=instance_sort_key)


def read_numbers_from_zip(zip_path, file_inside_zip):
    """Read a Schaller text file and return all numbers as integers."""
    with zipfile.ZipFile(zip_path, "r") as archive:
        text = archive.read(file_inside_zip).decode("latin1")

    return [int(value) for value in text.split()]


# ============================================================
# READ ONE SCHALLER INSTANCE
# ============================================================

def read_schaller_instance(zip_path, folder, instance):
    """Read the processing and setup data for one Schaller instance."""
    setup_file = find_file_in_zip(
        zip_path,
        folder,
        f"FAMSETUP.{instance}",
    )

    processing_file = find_file_in_zip(
        zip_path,
        folder,
        f"PROCTIME.{instance}",
    )

    # --------------------------------------------------------
    # Read setup data
    # --------------------------------------------------------

    setup_numbers = read_numbers_from_zip(
        zip_path,
        setup_file,
    )

    index = 0

    number_of_groups = setup_numbers[index]
    index += 1

    number_of_machines = setup_numbers[index]
    index += 1

    groups = list(range(1, number_of_groups + 1))
    machines = list(range(1, number_of_machines + 1))
    groups_with_dummy = [0] + groups

    setup = {}

    for machine in machines:
        # Initial setup:
        # dummy group 0 -> real family
        for next_group in groups:
            setup[0, next_group, machine] = setup_numbers[index]
            index += 1

        # Sequence-dependent family-to-family setup matrix
        for previous_group in groups:
            for next_group in groups:
                value = setup_numbers[index]
                index += 1

                if previous_group == next_group:
                    setup[
                        previous_group,
                        next_group,
                        machine,
                    ] = 0
                else:
                    setup[
                        previous_group,
                        next_group,
                        machine,
                    ] = value

    # --------------------------------------------------------
    # Read processing-time data
    # --------------------------------------------------------

    processing_numbers = read_numbers_from_zip(
        zip_path,
        processing_file,
    )

    index = 0

    processing_times = {
        machine: {}
        for machine in machines
    }

    group_jobs = {}

    global_job_number = 1

    for group in groups:
        number_of_jobs = processing_numbers[index]
        index += 1

        group_jobs[group] = []
        jobs_in_current_group = []

        for _ in range(number_of_jobs):
            group_jobs[group].append(global_job_number)
            jobs_in_current_group.append(global_job_number)
            global_job_number += 1

        # Rows are machines.
        # Columns are jobs.
        for machine in machines:
            for job in jobs_in_current_group:
                processing_times[machine][job] = (
                    processing_numbers[index]
                )
                index += 1

    return {
        "g": number_of_groups,
        "m": number_of_machines,
        "P": groups,
        "K": machines,
        "P0": groups_with_dummy,
        "pt": processing_times,
        "group_jobs": group_jobs,
        "setup": setup,
    }


# ============================================================
# CSV SAVING
# ============================================================

CSV_FIELDNAMES = [
    "timestamp",
    "folder",
    "instance",
    "number_of_groups",
    "number_of_machines",
    "number_of_jobs",
    "status_code",
    "status",
    "solution_count",
    "feasible_solution_found",
    "objective",
    "best_bound",
    "gap_decimal",
    "gap_percent",
    "runtime_seconds",
    "node_count",
    "iteration_count",
    "time_limit_seconds",
    "target_gap_percent",
    "group_order",
    "job_orders",
    "log_file",
    "gantt_file",
    "error",
]


def initialize_results_csv(csv_path):
    """
    Create the CSV file and header if the file does not exist.

    Existing results are preserved.
    """
    file_exists = os.path.exists(csv_path)

    if not file_exists:
        with open(
            csv_path,
            "w",
            newline="",
            encoding="utf-8",
        ) as csv_file:
            writer = csv.DictWriter(
                csv_file,
                fieldnames=CSV_FIELDNAMES,
            )
            writer.writeheader()


def append_result_to_csv(csv_path, result):
    """
    Append one result immediately.

    Saving after every instance protects the experiment if the server
    job is interrupted later.
    """
    with open(
        csv_path,
        "a",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=CSV_FIELDNAMES,
        )

        row = {
            field: result.get(field, "")
            for field in CSV_FIELDNAMES
        }

        writer.writerow(row)
        csv_file.flush()
        os.fsync(csv_file.fileno())


# ============================================================
# EXTRACT SOLUTION
# ============================================================

def extract_group_order(I, P, W):
    """
    Extract the selected family in every slot.

    Returns:
        {1: 2, 2: 3, 3: 1}
    """
    group_order = {}

    for slot in I:
        for group in P:
            if W[slot, group].X > 0.5:
                group_order[slot] = group
                break

    return group_order


def extract_job_orders(
    I,
    J,
    last_machine,
    group_order,
    job_label,
    X,
):
    """
    Extract job orders by sorting completion times on the final machine.

    Returns:
        {
            1: {"group": 2, "jobs": [7, 9, 6, 8]},
            ...
        }
    """
    job_orders = {}

    for slot in I:
        group = group_order[slot]
        jobs_with_completion_times = []

        for job_position in J:
            real_job = job_label[group, job_position]

            if real_job == "DUMMY":
                continue

            completion_time = X[
                slot,
                job_position,
                last_machine,
            ].X

            jobs_with_completion_times.append(
                (real_job, completion_time)
            )

        jobs_with_completion_times.sort(
            key=lambda item: item[1]
        )

        job_orders[slot] = {
            "group": group,
            "jobs": [
                job
                for job, _ in jobs_with_completion_times
            ],
        }

    return job_orders


# ============================================================
# GANTT CHART
# ============================================================

def save_gantt_chart(
    folder,
    instance,
    I,
    J,
    K,
    group_order,
    job_label,
    processing_time_by_position,
    setup_time_variables,
    X,
    output_path,
):
    """
    Save a machine-based Gantt chart.

    Job operations are drawn from:
        start time = completion time - processing time

    Setup operations are also drawn when their duration is positive.
    """
    figure_height = max(4, 1.1 * len(K) + 2)

    fig, ax = plt.subplots(
        figsize=(16, figure_height)
    )

    color_map = plt.get_cmap("tab20")
    group_colors = {
        group: color_map((group - 1) % 20)
        for group in set(group_order.values())
    }

    bar_height = 0.62

    for machine in K:
        y_position = machine - 1

        for slot in I:
            group = group_order[slot]

            # Determine when processing of this family begins on
            # this machine. This is useful for displaying setup.
            if slot == 1:
                previous_slot_completion = 0.0
            else:
                previous_job_completion_times = []

                previous_group = group_order[slot - 1]

                for previous_job_position in J:
                    if (
                        job_label[
                            previous_group,
                            previous_job_position,
                        ]
                        == "DUMMY"
                    ):
                        continue

                    previous_job_completion_times.append(
                        X[
                            slot - 1,
                            previous_job_position,
                            machine,
                        ].X
                    )

                previous_slot_completion = max(
                    previous_job_completion_times,
                    default=0.0,
                )

            setup_duration = setup_time_variables[
                slot,
                machine,
            ].X

            if setup_duration > 1e-6:
                ax.barh(
                    y_position,
                    setup_duration,
                    left=previous_slot_completion,
                    height=bar_height,
                    color="lightgray",
                    edgecolor="black",
                    hatch="//",
                    linewidth=0.7,
                )

                ax.text(
                    previous_slot_completion
                    + setup_duration / 2,
                    y_position,
                    f"S{group}",
                    ha="center",
                    va="center",
                    fontsize=7,
                )

            operations = []

            for job_position in J:
                real_job = job_label[group, job_position]

                if real_job == "DUMMY":
                    continue

                duration = processing_time_by_position[
                    group,
                    job_position,
                    machine,
                ]

                completion = X[
                    slot,
                    job_position,
                    machine,
                ].X

                start = completion - duration

                operations.append(
                    (
                        start,
                        duration,
                        real_job,
                        group,
                    )
                )

            operations.sort(key=lambda item: item[0])

            for start, duration, real_job, operation_group in operations:
                ax.barh(
                    y_position,
                    duration,
                    left=start,
                    height=bar_height,
                    color=group_colors[operation_group],
                    edgecolor="black",
                    linewidth=0.7,
                )

                ax.text(
                    start + duration / 2,
                    y_position,
                    f"J{real_job}",
                    ha="center",
                    va="center",
                    fontsize=7,
                )

    ax.set_yticks(range(len(K)))
    ax.set_yticklabels(
        [f"Machine {machine}" for machine in K]
    )

    ax.set_xlabel("Time")
    ax.set_ylabel("Machine")
    ax.set_title(
        f"Gantt Chart — {folder}/{instance}"
    )

    ax.grid(
        True,
        axis="x",
        linestyle=":",
        alpha=0.6,
    )

    legend_items = [
        Patch(
            facecolor=group_colors[group],
            edgecolor="black",
            label=f"Group {group}",
        )
        for group in sorted(group_colors)
    ]

    legend_items.append(
        Patch(
            facecolor="lightgray",
            edgecolor="black",
            hatch="//",
            label="Setup",
        )
    )

    ax.legend(
        handles=legend_items,
        loc="upper right",
    )

    fig.tight_layout()
    fig.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)


# ============================================================
# SOLVE ONE INSTANCE
# ============================================================

def solve_instance(
    zip_path,
    folder,
    instance,
    time_limit,
    target_mip_gap,
):
    """
    Build and solve the Salmasi-style MILP for one instance.

    Returns one dictionary that can be saved directly to CSV.
    """
    timestamp = datetime.now().isoformat(timespec="seconds")

    safe_instance_name = f"{folder}_{instance}"

    log_path = os.path.join(
        LOG_DIRECTORY,
        f"{safe_instance_name}.log",
    )

    gantt_path = os.path.join(
        GANTT_DIRECTORY,
        f"{safe_instance_name}.png",
    )

    result = {
        "timestamp": timestamp,
        "folder": folder,
        "instance": instance,
        "number_of_groups": "",
        "number_of_machines": "",
        "number_of_jobs": "",
        "status_code": "",
        "status": "",
        "solution_count": 0,
        "feasible_solution_found": False,
        "objective": "",
        "best_bound": "",
        "gap_decimal": "",
        "gap_percent": "",
        "runtime_seconds": "",
        "node_count": "",
        "iteration_count": "",
        "time_limit_seconds": time_limit,
        "target_gap_percent": target_mip_gap * 100,
        "group_order": "",
        "job_orders": "",
        "log_file": log_path,
        "gantt_file": "",
        "error": "",
    }

    model = None

    try:
        print("\n" + "=" * 70)
        print(f"Reading instance {folder}/{instance}")
        print("=" * 70)

        data = read_schaller_instance(
            zip_path,
            folder,
            instance,
        )

        g = data["g"]
        m = data["m"]
        P = data["P"]
        K = data["K"]
        P0 = data["P0"]
        pt = data["pt"]
        group_jobs = data["group_jobs"]
        setup = data["setup"]

        I = list(range(1, g + 1))

        number_of_jobs = sum(
            len(jobs)
            for jobs in group_jobs.values()
        )

        result["number_of_groups"] = g
        result["number_of_machines"] = m
        result["number_of_jobs"] = number_of_jobs

        # ----------------------------------------------------
        # Prepare model data
        # ----------------------------------------------------

        maximum_group_size = max(
            len(group_jobs[group])
            for group in P
        )

        J = list(range(1, maximum_group_size + 1))

        processing_time_by_position = {}
        modified_processing_time = {}
        job_label = {}
        padded_jobs = {}

        total_processing_time = sum(
            pt[machine][job]
            for machine in K
            for job in pt[machine]
        )

        total_setup_time = sum(
            setup[previous_group, next_group, machine]
            for previous_group in P0
            for next_group in P
            for machine in K
        )

        big_m = (
            total_processing_time
            + total_setup_time
            + 100
        )

        for group in P:
            real_jobs = list(group_jobs[group])

            padded_jobs[group] = (
                real_jobs
                + [None]
                * (
                    maximum_group_size
                    - len(real_jobs)
                )
            )

        for group in P:
            for job_position in J:
                real_job = padded_jobs[group][
                    job_position - 1
                ]

                if real_job is None:
                    job_label[
                        group,
                        job_position,
                    ] = "DUMMY"

                    for machine in K:
                        processing_time_by_position[
                            group,
                            job_position,
                            machine,
                        ] = 0

                        modified_processing_time[
                            group,
                            job_position,
                            machine,
                        ] = -big_m
                else:
                    job_label[
                        group,
                        job_position,
                    ] = real_job

                    for machine in K:
                        value = pt[machine][real_job]

                        processing_time_by_position[
                            group,
                            job_position,
                            machine,
                        ] = value

                        modified_processing_time[
                            group,
                            job_position,
                            machine,
                        ] = value

        total_group_processing_time = {
            (group, machine): gp.quicksum(
                processing_time_by_position[
                    group,
                    job_position,
                    machine,
                ]
                for job_position in J
            )
            for group in P
            for machine in K
        }

        # ----------------------------------------------------
        # Build model
        # ----------------------------------------------------

        model = gp.Model(
            f"Salmasi_{folder}_{instance}"
        )

        # Save the complete Gurobi log for this instance.
        model.setParam("LogFile", log_path)

        # Maximum time for this instance.
        model.setParam("TimeLimit", time_limit)

        # Requested relative MIP gap.
        model.setParam("MIPGap", target_mip_gap)

        # W[i,p] = 1 if family p is assigned to slot i.
        W = model.addVars(
            I,
            P,
            vtype=GRB.BINARY,
            name="W",
        )

        # A[i,p,l] = 1 if slot transition i is p -> l.
        A = model.addVars(
            range(0, g),
            P0,
            P,
            vtype=GRB.BINARY,
            name="A",
        )

        # Y[i,j,q] determines the relative order of job
        # positions j and q inside slot i.
        Y = model.addVars(
            [
                (slot, job_j, job_q)
                for slot in I
                for job_j in J
                for job_q in J
                if job_j < job_q
            ],
            vtype=GRB.BINARY,
            name="Y",
        )

        # X[i,j,k] is the completion time of job position j
        # in slot i on machine k.
        X = model.addVars(
            I,
            J,
            K,
            lb=0.0,
            vtype=GRB.CONTINUOUS,
            name="X",
        )

        # C[i,k] is the completion time of slot i on machine k.
        C = model.addVars(
            I,
            K,
            lb=0.0,
            vtype=GRB.CONTINUOUS,
            name="C",
        )

        # O[i,k] is the setup time before slot i on machine k.
        O = model.addVars(
            I,
            K,
            lb=0.0,
            vtype=GRB.CONTINUOUS,
            name="O",
        )

        # ----------------------------------------------------
        # Objective:
        #
        # Minimize the completion time of the final slot
        # on the final machine, which is the makespan.
        # ----------------------------------------------------

        model.setObjective(
            C[g, m],
            GRB.MINIMIZE,
        )

        # ----------------------------------------------------
        # Assignment constraints
        # ----------------------------------------------------

        # Every family is assigned exactly once.
        for group in P:
            model.addConstr(
                gp.quicksum(
                    W[slot, group]
                    for slot in I
                )
                == 1,
                name=f"assign_group_{group}",
            )

        # Every slot receives exactly one family.
        for slot in I:
            model.addConstr(
                gp.quicksum(
                    W[slot, group]
                    for group in P
                )
                == 1,
                name=f"fill_slot_{slot}",
            )

        # ----------------------------------------------------
        # Transition constraints
        # ----------------------------------------------------

        # Exactly one family transition is active before
        # every real slot.
        for transition_position in range(0, g):
            model.addConstr(
                gp.quicksum(
                    A[
                        transition_position,
                        previous_group,
                        next_group,
                    ]
                    for previous_group in P0
                    for next_group in P
                    if next_group != previous_group
                )
                == 1,
                name=(
                    f"one_transition_"
                    f"{transition_position}"
                ),
            )

        # Initial transition:
        # dummy group 0 -> family in slot 1.
        for next_group in P:
            model.addConstr(
                A[0, 0, next_group]
                == W[1, next_group],
                name=f"initial_transition_{next_group}",
            )

        # Real family-to-family transitions.
        for transition_position in range(1, g):
            for previous_group in P:
                for next_group in P:
                    if previous_group == next_group:
                        continue

                    model.addConstr(
                        A[
                            transition_position,
                            previous_group,
                            next_group,
                        ]
                        <= W[
                            transition_position,
                            previous_group,
                        ]
                    )

                    model.addConstr(
                        A[
                            transition_position,
                            previous_group,
                            next_group,
                        ]
                        <= W[
                            transition_position + 1,
                            next_group,
                        ]
                    )

                    model.addConstr(
                        A[
                            transition_position,
                            previous_group,
                            next_group,
                        ]
                        >= (
                            W[
                                transition_position,
                                previous_group,
                            ]
                            + W[
                                transition_position + 1,
                                next_group,
                            ]
                            - 1
                        )
                    )

        # Invalid transitions are fixed to zero.
        for transition_position in range(0, g):
            for previous_group in P0:
                for next_group in P:
                    invalid_same_group = (
                        previous_group == next_group
                    )

                    invalid_initial_transition = (
                        transition_position == 0
                        and previous_group != 0
                    )

                    if (
                        invalid_same_group
                        or invalid_initial_transition
                    ):
                        model.addConstr(
                            A[
                                transition_position,
                                previous_group,
                                next_group,
                            ]
                            == 0
                        )

        # ----------------------------------------------------
        # Setup-time constraints
        # ----------------------------------------------------

        for slot in I:
            for machine in K:
                model.addConstr(
                    O[slot, machine]
                    == gp.quicksum(
                        A[
                            slot - 1,
                            previous_group,
                            next_group,
                        ]
                        * setup[
                            previous_group,
                            next_group,
                            machine,
                        ]
                        for previous_group in P0
                        for next_group in P
                        if previous_group != next_group
                    ),
                    name=f"setup_{slot}_{machine}",
                )

        # ----------------------------------------------------
        # Completion on the first machine
        # ----------------------------------------------------

        for slot in I:
            previous_completion = (
                0
                if slot == 1
                else C[slot - 1, 1]
            )

            model.addConstr(
                C[slot, 1]
                == (
                    previous_completion
                    + O[slot, 1]
                    + gp.quicksum(
                        W[slot, group]
                        * total_group_processing_time[
                            group,
                            1,
                        ]
                        for group in P
                    )
                ),
                name=f"first_machine_{slot}",
            )

        # ----------------------------------------------------
        # Job release/timing constraints
        # ----------------------------------------------------

        for slot in I:
            for job_position in J:
                for machine in K:
                    previous_slot_completion = (
                        0
                        if slot == 1
                        else C[slot - 1, machine]
                    )

                    model.addConstr(
                        X[
                            slot,
                            job_position,
                            machine,
                        ]
                        >= (
                            previous_slot_completion
                            + O[slot, machine]
                            + gp.quicksum(
                                W[slot, group]
                                * modified_processing_time[
                                    group,
                                    job_position,
                                    machine,
                                ]
                                for group in P
                            )
                        )
                    )

        # ----------------------------------------------------
        # Pairwise sequencing inside each selected family
        # ----------------------------------------------------

        for slot in I:
            for job_j in J:
                for job_q in J:
                    if job_j >= job_q:
                        continue

                    for machine in K:
                        processing_j = gp.quicksum(
                            W[slot, group]
                            * modified_processing_time[
                                group,
                                job_j,
                                machine,
                            ]
                            for group in P
                        )

                        processing_q = gp.quicksum(
                            W[slot, group]
                            * modified_processing_time[
                                group,
                                job_q,
                                machine,
                            ]
                            for group in P
                        )

                        model.addConstr(
                            X[slot, job_j, machine]
                            - X[slot, job_q, machine]
                            + big_m
                            * Y[slot, job_j, job_q]
                            >= processing_j
                        )

                        model.addConstr(
                            X[slot, job_q, machine]
                            - X[slot, job_j, machine]
                            + big_m
                            * (
                                1
                                - Y[
                                    slot,
                                    job_j,
                                    job_q,
                                ]
                            )
                            >= processing_q
                        )

        # ----------------------------------------------------
        # Flowshop constraints
        # ----------------------------------------------------

        for slot in I:
            for job_position in J:
                for machine in K:
                    if machine < 2:
                        continue

                    model.addConstr(
                        X[
                            slot,
                            job_position,
                            machine,
                        ]
                        - X[
                            slot,
                            job_position,
                            machine - 1,
                        ]
                        >= gp.quicksum(
                            W[slot, group]
                            * processing_time_by_position[
                                group,
                                job_position,
                                machine,
                            ]
                            for group in P
                        )
                    )

        # ----------------------------------------------------
        # Slot completion constraints
        # ----------------------------------------------------

        for slot in I:
            for machine in K:
                for job_position in J:
                    model.addConstr(
                        C[slot, machine]
                        >= X[
                            slot,
                            job_position,
                            machine,
                        ]
                    )

        # ----------------------------------------------------
        # Optimize
        # ----------------------------------------------------

        print(
            f"Solving {folder}/{instance} "
            f"with a {time_limit}-second limit..."
        )

        model.optimize()

        status_code = model.Status
        status_name = get_status_name(status_code)
        solution_count = model.SolCount

        result["status_code"] = status_code
        result["status"] = status_name
        result["solution_count"] = solution_count
        result["runtime_seconds"] = round(
            model.Runtime,
            4,
        )
        result["node_count"] = round(
            model.NodeCount,
            4,
        )
        result["iteration_count"] = round(
            model.IterCount,
            4,
        )

        # The best bound can be useful even when no feasible
        # solution has been found.
        try:
            best_bound = model.ObjBound

            if math.isfinite(best_bound):
                result["best_bound"] = round(
                    best_bound,
                    6,
                )
        except gp.GurobiError:
            pass

        if solution_count == 0:
            result["feasible_solution_found"] = False

            print(
                f"No feasible solution found for "
                f"{folder}/{instance}."
            )
            print(f"Status: {status_name}")

            return result

        # ----------------------------------------------------
        # A feasible solution exists
        # ----------------------------------------------------

        result["feasible_solution_found"] = True
        result["objective"] = round(
            model.ObjVal,
            6,
        )

        try:
            mip_gap = model.MIPGap

            if math.isfinite(mip_gap):
                result["gap_decimal"] = round(
                    mip_gap,
                    8,
                )

                result["gap_percent"] = round(
                    mip_gap * 100,
                    6,
                )
        except gp.GurobiError:
            pass

        group_order = extract_group_order(
            I,
            P,
            W,
        )

        job_orders = extract_job_orders(
            I,
            J,
            m,
            group_order,
            job_label,
            X,
        )

        result["group_order"] = json.dumps(
            group_order,
            sort_keys=True,
        )

        result["job_orders"] = json.dumps(
            job_orders,
            sort_keys=True,
        )

        print(f"\nInstance: {folder}/{instance}")
        print(
            f"Status: {status_name}"
        )
        print(
            f"Best makespan found: "
            f"{model.ObjVal:.2f}"
        )

        if result["best_bound"] != "":
            print(
                f"Best bound: "
                f"{float(result['best_bound']):.2f}"
            )

        if result["gap_percent"] != "":
            print(
                f"Optimality gap: "
                f"{float(result['gap_percent']):.2f}%"
            )

        print(
            f"Runtime: {model.Runtime:.2f} seconds"
        )
        print(f"Group order: {group_order}")

        print("\nJob order inside each selected group:")

        for slot in sorted(job_orders):
            group = job_orders[slot]["group"]
            jobs = job_orders[slot]["jobs"]

            print(
                f"Slot {slot}, Group {group}: {jobs}"
            )

        # ----------------------------------------------------
        # Save Gantt chart
        # ----------------------------------------------------

        if SAVE_GANTT_CHARTS:
            try:
                save_gantt_chart(
                    folder=folder,
                    instance=instance,
                    I=I,
                    J=J,
                    K=K,
                    group_order=group_order,
                    job_label=job_label,
                    processing_time_by_position=(
                        processing_time_by_position
                    ),
                    setup_time_variables=O,
                    X=X,
                    output_path=gantt_path,
                )

                result["gantt_file"] = gantt_path

                print(
                    f"Gantt chart saved to: {gantt_path}"
                )

            except Exception as chart_error:
                print(
                    "Warning: the solution was saved, "
                    "but the Gantt chart could not be created."
                )
                print(f"Chart error: {chart_error}")

        return result

    except Exception as error:
        result["status"] = "ERROR"
        result["error"] = (
            f"{type(error).__name__}: {error}"
        )

        print(
            f"\nError while solving "
            f"{folder}/{instance}:"
        )
        print(result["error"])
        traceback.print_exc()

        return result

    finally:
        if model is not None:
            model.dispose()


# ============================================================
# MAIN BATCH EXPERIMENT
# ============================================================

def main():
    """Run every available D-instance in the selected folder."""
    create_output_directories()
    initialize_results_csv(RESULTS_CSV)

    if not os.path.exists(ZIP_PATH):
        print(f"ZIP file not found: {ZIP_PATH}")
        sys.exit(1)

    instances = find_instances_in_zip(
        ZIP_PATH,
        FOLDER,
    )

    if not instances:
        print(
            f"No PROCTIME.Dxx files were found "
            f"inside folder {FOLDER}."
        )
        sys.exit(1)

    print("=" * 70)
    print("BATCH EXPERIMENT")
    print("=" * 70)
    print(f"ZIP file: {ZIP_PATH}")
    print(f"Folder: {FOLDER}")
    print(f"Instances found: {instances}")
    print(
        f"Time limit per instance: "
        f"{TIME_LIMIT} seconds"
    )
    print(
        f"Target gap: "
        f"{TARGET_MIP_GAP * 100:.2f}%"
    )
    print(f"Results CSV: {RESULTS_CSV}")
    print(f"Log directory: {LOG_DIRECTORY}")

    consecutive_no_solution = 0
    completed_instances = 0

    for instance in instances:
        result = solve_instance(
            zip_path=ZIP_PATH,
            folder=FOLDER,
            instance=instance,
            time_limit=TIME_LIMIT,
            target_mip_gap=TARGET_MIP_GAP,
        )

        # Save immediately after every run.
        append_result_to_csv(
            RESULTS_CSV,
            result,
        )

        completed_instances += 1

        print(
            f"\nResult saved to {RESULTS_CSV}"
        )

        if result["feasible_solution_found"]:
            consecutive_no_solution = 0
        else:
            consecutive_no_solution += 1

        if (
            STOP_AFTER_CONSECUTIVE_NO_SOLUTION
            is not None
            and consecutive_no_solution
            >= STOP_AFTER_CONSECUTIVE_NO_SOLUTION
        ):
            print("\n" + "=" * 70)
            print(
                "Stopping this folder because "
                f"{consecutive_no_solution} consecutive "
                "instances had no feasible solution."
            )
            print("=" * 70)
            break

    print("\n" + "=" * 70)
    print("BATCH FINISHED")
    print("=" * 70)
    print(
        f"Instances attempted: {completed_instances}"
    )
    print(f"Results saved to: {RESULTS_CSV}")
    print(f"Logs saved in: {LOG_DIRECTORY}")

    if SAVE_GANTT_CHARTS:
        print(
            f"Gantt charts saved in: "
            f"{GANTT_DIRECTORY}"
        )


if __name__ == "__main__":
    main()