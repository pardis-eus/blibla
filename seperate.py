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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


# ============================================================
# SETTINGS
# ============================================================

# Everything will be saved relative to this Python file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

ZIP_PATH = os.path.join(
    BASE_DIR,
    "GS Instances Schaller.zip"
)

# Change this when you want another Schaller folder
FOLDER = "ssu33"

# 30 minutes for EACH D-instance
TIME_LIMIT = 1800

# Stop earlier if Gurobi reaches 1% gap
TARGET_MIP_GAP = 0.01

# Save Gantt charts
SAVE_GANTT_CHARTS = True

# Stop after 3 consecutive instances without any feasible solution
STOP_AFTER_CONSECUTIVE_NO_SOLUTION = 3


# ============================================================
# OUTPUT DIRECTORIES
# ============================================================

# Different folders from the original grouped-family experiment
RESULTS_DIRECTORY = os.path.join(
    BASE_DIR,
    "results_each_job_family"
)

LOG_DIRECTORY = os.path.join(
    BASE_DIR,
    "logs_each_job_family"
)

GANTT_DIRECTORY = os.path.join(
    BASE_DIR,
    "gantt_each_job_family"
)

RESULTS_CSV = os.path.join(
    RESULTS_DIRECTORY,
    f"results_each_job_family_{FOLDER}.csv"
)


# ============================================================
# GUROBI STATUS NAMES
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
    return GUROBI_STATUS_NAMES.get(
        status_code,
        f"UNKNOWN_STATUS_{status_code}"
    )


# ============================================================
# CREATE OUTPUT FOLDERS
# ============================================================

def create_output_directories():

    os.makedirs(
        RESULTS_DIRECTORY,
        exist_ok=True
    )

    os.makedirs(
        LOG_DIRECTORY,
        exist_ok=True
    )

    if SAVE_GANTT_CHARTS:
        os.makedirs(
            GANTT_DIRECTORY,
            exist_ok=True
        )


# ============================================================
# ZIP HELPERS
# ============================================================

def normalize_zip_path(path):
    return path.replace("\\", "/")


def path_belongs_to_folder(path, folder):

    clean_path = normalize_zip_path(path)

    components = clean_path.split("/")

    return folder in components


def find_file_in_zip(
    zip_path,
    folder,
    filename
):

    with zipfile.ZipFile(
        zip_path,
        "r"
    ) as archive:

        for name in archive.namelist():

            clean_name = normalize_zip_path(name)

            if (
                path_belongs_to_folder(
                    clean_name,
                    folder
                )
                and clean_name.endswith(filename)
            ):
                return name

    raise FileNotFoundError(
        f"Could not find {filename} in {folder}"
    )


def find_instances_in_zip(
    zip_path,
    folder
):

    instances = set()

    pattern = re.compile(
        r"PROCTIME\.(D\d+)$",
        re.IGNORECASE
    )

    with zipfile.ZipFile(
        zip_path,
        "r"
    ) as archive:

        for name in archive.namelist():

            clean_name = normalize_zip_path(name)

            if not path_belongs_to_folder(
                clean_name,
                folder
            ):
                continue

            filename = clean_name.split("/")[-1]

            match = pattern.fullmatch(filename)

            if match:
                instances.add(
                    match.group(1).upper()
                )

    def instance_sort_key(instance_name):

        match = re.search(
            r"\d+",
            instance_name
        )

        if match:
            return int(match.group())

        return instance_name

    return sorted(
        instances,
        key=instance_sort_key
    )


def read_numbers_from_zip(
    zip_path,
    file_inside_zip
):

    with zipfile.ZipFile(
        zip_path,
        "r"
    ) as archive:

        text = archive.read(
            file_inside_zip
        ).decode("latin1")

    return [
        int(value)
        for value in text.split()
    ]


# ============================================================
# READ ORIGINAL SCHALLER INSTANCE
# ============================================================

def read_original_schaller_instance(
    zip_path,
    folder,
    instance
):

    setup_file = find_file_in_zip(
        zip_path,
        folder,
        f"FAMSETUP.{instance}"
    )

    processing_file = find_file_in_zip(
        zip_path,
        folder,
        f"PROCTIME.{instance}"
    )


    # --------------------------------------------------------
    # ORIGINAL SETUP DATA
    # --------------------------------------------------------

    setup_numbers = read_numbers_from_zip(
        zip_path,
        setup_file
    )

    index = 0

    original_g = setup_numbers[index]
    index += 1

    m = setup_numbers[index]
    index += 1

    original_families = list(
        range(1, original_g + 1)
    )

    machines = list(
        range(1, m + 1)
    )

    original_families_with_dummy = (
        [0] + original_families
    )

    original_setup = {}


    # Each machine contains:
    # initial setup row
    # + family-to-family setup matrix

    for machine in machines:

        # Initial setup:
        # 0 -> family
        for next_family in original_families:

            original_setup[
                0,
                next_family,
                machine
            ] = setup_numbers[index]

            index += 1


        # Family-to-family setups
        for previous_family in original_families:

            for next_family in original_families:

                value = setup_numbers[index]

                index += 1

                if previous_family == next_family:

                    original_setup[
                        previous_family,
                        next_family,
                        machine
                    ] = 0

                else:

                    original_setup[
                        previous_family,
                        next_family,
                        machine
                    ] = value


    # --------------------------------------------------------
    # ORIGINAL PROCESSING DATA
    # --------------------------------------------------------

    processing_numbers = read_numbers_from_zip(
        zip_path,
        processing_file
    )

    index = 0

    pt = {
        machine: {}
        for machine in machines
    }

    original_group_jobs = {}

    # VERY IMPORTANT:
    # remember the original family of every job
    job_original_family = {}

    global_job_number = 1


    for family in original_families:

        number_of_jobs = (
            processing_numbers[index]
        )

        index += 1

        original_group_jobs[family] = []

        jobs_in_family = []


        for _ in range(number_of_jobs):

            job = global_job_number

            original_group_jobs[
                family
            ].append(job)

            jobs_in_family.append(job)

            job_original_family[
                job
            ] = family

            global_job_number += 1


        # Rows = machines
        # Columns = jobs
        for machine in machines:

            for job in jobs_in_family:

                pt[machine][job] = (
                    processing_numbers[index]
                )

                index += 1


    return {
        "original_g": original_g,
        "m": m,
        "original_families": original_families,
        "machines": machines,
        "original_group_jobs": original_group_jobs,
        "job_original_family": job_original_family,
        "pt": pt,
        "original_setup": original_setup,
    }


# ============================================================
# TRANSFORM:
# EVERY JOB BECOMES ITS OWN FAMILY
# ============================================================

def transform_jobs_into_families(
    original_data
):

    machines = original_data["machines"]

    pt = original_data["pt"]

    original_setup = (
        original_data["original_setup"]
    )

    job_original_family = (
        original_data["job_original_family"]
    )


    # All real jobs
    all_jobs = sorted(
        job_original_family.keys()
    )


    # --------------------------------------------------------
    # NEW FAMILY SET
    # --------------------------------------------------------
    #
    # New family ID = job ID
    #
    # Example:
    #
    # Job 1 -> new Family 1
    # Job 2 -> new Family 2
    # ...
    #
    # Each family contains exactly one job.
    # --------------------------------------------------------

    P = list(all_jobs)

    g = len(P)

    P0 = [0] + P

    K = list(machines)

    m = len(K)


    # --------------------------------------------------------
    # NEW GROUP MEMBERSHIP
    # --------------------------------------------------------

    group_jobs = {
        job: [job]
        for job in all_jobs
    }


    # --------------------------------------------------------
    # CREATE NEW SETUP MATRIX
    # --------------------------------------------------------

    setup = {}


    for machine in K:

        # ----------------------------------------------------
        # Initial setup
        # ----------------------------------------------------
        #
        # Job-family inherits initial setup
        # of its ORIGINAL family.
        #
        # Example:
        #
        # Job 5 originally belonged to F1
        #
        # setup[0,5,k]
        # =
        # original_setup[0,1,k]
        # ----------------------------------------------------

        for next_job_family in P:

            original_next_family = (
                job_original_family[
                    next_job_family
                ]
            )

            setup[
                0,
                next_job_family,
                machine
            ] = original_setup[
                0,
                original_next_family,
                machine
            ]


        # ----------------------------------------------------
        # Job-family to job-family setup
        # ----------------------------------------------------

        for previous_job_family in P:

            original_previous_family = (
                job_original_family[
                    previous_job_family
                ]
            )

            for next_job_family in P:

                original_next_family = (
                    job_original_family[
                        next_job_family
                    ]
                )


                # Same job-family
                if (
                    previous_job_family
                    == next_job_family
                ):

                    setup[
                        previous_job_family,
                        next_job_family,
                        machine
                    ] = 0


                # IMPORTANT RULE:
                #
                # Both jobs came from the same
                # original Schaller family
                #
                # -> no setup
                elif (
                    original_previous_family
                    == original_next_family
                ):

                    setup[
                        previous_job_family,
                        next_job_family,
                        machine
                    ] = 0


                # Different original families
                #
                # -> inherit Schaller setup
                else:

                    setup[
                        previous_job_family,
                        next_job_family,
                        machine
                    ] = original_setup[
                        original_previous_family,
                        original_next_family,
                        machine
                    ]


    return {
        "g": g,
        "m": m,
        "P": P,
        "P0": P0,
        "K": K,
        "pt": pt,
        "group_jobs": group_jobs,
        "setup": setup,
        "job_original_family": (
            job_original_family
        ),
    }


# ============================================================
# READ + TRANSFORM ONE INSTANCE
# ============================================================

def read_modified_instance(
    zip_path,
    folder,
    instance
):

    original_data = (
        read_original_schaller_instance(
            zip_path,
            folder,
            instance
        )
    )

    modified_data = (
        transform_jobs_into_families(
            original_data
        )
    )

    return modified_data


# ============================================================
# CSV
# ============================================================

CSV_FIELDNAMES = [
    "timestamp",
    "folder",
    "instance",
    "number_of_job_families",
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
    "job_family_order",
    "original_family_order",
    "log_file",
    "gantt_file",
    "error",
]


def initialize_results_csv(
    csv_path
):

    if not os.path.exists(csv_path):

        with open(
            csv_path,
            "w",
            newline="",
            encoding="utf-8"
        ) as csv_file:

            writer = csv.DictWriter(
                csv_file,
                fieldnames=CSV_FIELDNAMES
            )

            writer.writeheader()


def append_result_to_csv(
    csv_path,
    result
):

    with open(
        csv_path,
        "a",
        newline="",
        encoding="utf-8"
    ) as csv_file:

        writer = csv.DictWriter(
            csv_file,
            fieldnames=CSV_FIELDNAMES
        )

        row = {
            field: result.get(
                field,
                ""
            )
            for field in CSV_FIELDNAMES
        }

        writer.writerow(row)

        csv_file.flush()

        os.fsync(
            csv_file.fileno()
        )


# ============================================================
# EXTRACT JOB-FAMILY ORDER
# ============================================================

def extract_family_order(
    I,
    P,
    W
):

    sequence = {}

    for slot in I:

        for family in P:

            if W[
                slot,
                family
            ].X > 0.5:

                sequence[slot] = family

                break

    return sequence


# ============================================================
# GANTT CHART
# ============================================================

def save_gantt_chart(
    folder,
    instance,
    I,
    K,
    sequence,
    pt,
    X,
    O,
    job_original_family,
    output_path
):

    figure_height = max(
        4,
        1.1 * len(K) + 2
    )

    fig, ax = plt.subplots(
        figsize=(18, figure_height)
    )


    # Color represents ORIGINAL family
    original_families = sorted(
        set(job_original_family.values())
    )

    color_map = plt.get_cmap("tab20")

    original_family_colors = {
        family: color_map(
            (family - 1) % 20
        )
        for family in original_families
    }


    bar_height = 0.62


    for machine in K:

        y_position = machine - 1

        previous_completion = 0.0


        for slot in I:

            job = sequence[slot]

            original_family = (
                job_original_family[job]
            )


            # ------------------------------------------------
            # Setup
            # ------------------------------------------------

            setup_duration = O[
                slot,
                machine
            ].X

            if setup_duration > 1e-6:

                ax.barh(
                    y_position,
                    setup_duration,
                    left=previous_completion,
                    height=bar_height,
                    color="lightgray",
                    edgecolor="black",
                    hatch="//",
                    linewidth=0.7
                )

                ax.text(
                    previous_completion
                    + setup_duration / 2,
                    y_position,
                    f"S",
                    ha="center",
                    va="center",
                    fontsize=7
                )


            # ------------------------------------------------
            # Job
            # ------------------------------------------------

            duration = pt[
                machine
            ][job]

            completion = X[
                slot,
                1,
                machine
            ].X

            start = (
                completion
                - duration
            )


            ax.barh(
                y_position,
                duration,
                left=start,
                height=bar_height,
                color=original_family_colors[
                    original_family
                ],
                edgecolor="black",
                linewidth=0.7
            )

            ax.text(
                start + duration / 2,
                y_position,
                f"J{job}",
                ha="center",
                va="center",
                fontsize=7
            )


            previous_completion = completion


    ax.set_yticks(
        range(len(K))
    )

    ax.set_yticklabels(
        [
            f"Machine {machine}"
            for machine in K
        ]
    )

    ax.set_xlabel("Time")

    ax.set_ylabel("Machine")

    ax.set_title(
        f"Each Job as Family — "
        f"{folder}/{instance}"
    )

    ax.grid(
        True,
        axis="x",
        linestyle=":",
        alpha=0.6
    )


    legend_items = [
        Patch(
            facecolor=(
                original_family_colors[
                    family
                ]
            ),
            edgecolor="black",
            label=(
                f"Original Family {family}"
            )
        )
        for family in original_families
    ]

    legend_items.append(
        Patch(
            facecolor="lightgray",
            edgecolor="black",
            hatch="//",
            label="Setup"
        )
    )

    ax.legend(
        handles=legend_items,
        loc="upper right"
    )

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close(fig)


# ============================================================
# SOLVE ONE MODIFIED INSTANCE
# ============================================================

def solve_instance(
    zip_path,
    folder,
    instance,
    time_limit,
    target_mip_gap
):

    timestamp = datetime.now().isoformat(
        timespec="seconds"
    )

    name = (
        f"{folder}_{instance}_each_job_family"
    )

    log_path = os.path.join(
        LOG_DIRECTORY,
        f"{name}.log"
    )

    gantt_path = os.path.join(
        GANTT_DIRECTORY,
        f"{name}.png"
    )


    result = {
        "timestamp": timestamp,
        "folder": folder,
        "instance": instance,
        "number_of_job_families": "",
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
        "target_gap_percent": (
            target_mip_gap * 100
        ),
        "job_family_order": "",
        "original_family_order": "",
        "log_file": log_path,
        "gantt_file": "",
        "error": "",
    }


    model = None


    try:

        print("\n" + "=" * 70)

        print(
            f"Reading modified instance "
            f"{folder}/{instance}"
        )

        print("=" * 70)


        data = read_modified_instance(
            zip_path,
            folder,
            instance
        )


        g = data["g"]
        m = data["m"]

        P = data["P"]
        P0 = data["P0"]
        K = data["K"]

        pt = data["pt"]

        group_jobs = data[
            "group_jobs"
        ]

        setup = data["setup"]

        job_original_family = data[
            "job_original_family"
        ]


        I = list(
            range(1, g + 1)
        )


        result[
            "number_of_job_families"
        ] = g

        result[
            "number_of_machines"
        ] = m

        result[
            "number_of_jobs"
        ] = g


        # ====================================================
        # MODEL DATA
        # ====================================================

        # Every family has exactly ONE job
        J = [1]


        t = {}

        job_label = {}


        for family in P:

            job = group_jobs[
                family
            ][0]

            job_label[
                family,
                1
            ] = job


            for machine in K:

                t[
                    family,
                    1,
                    machine
                ] = pt[
                    machine
                ][job]


        # Total processing time of each
        # one-job family
        T = {
            (
                family,
                machine
            ): t[
                family,
                1,
                machine
            ]
            for family in P
            for machine in K
        }


        # ====================================================
        # GUROBI MODEL
        # ====================================================

        model = gp.Model(
            f"EachJobFamily_{folder}_{instance}"
        )


        model.setParam(
            "LogFile",
            log_path
        )

        model.setParam(
            "TimeLimit",
            time_limit
        )

        model.setParam(
            "MIPGap",
            target_mip_gap
        )


        # ----------------------------------------------------
        # VARIABLES
        # ----------------------------------------------------

        # W[i,p] = 1
        # if job-family p is in slot i
        W = model.addVars(
            I,
            P,
            vtype=GRB.BINARY,
            name="W"
        )


        # Transition variable
        A = model.addVars(
            range(0, g),
            P0,
            P,
            vtype=GRB.BINARY,
            name="A"
        )


        # Completion time of the single job
        X = model.addVars(
            I,
            J,
            K,
            lb=0.0,
            vtype=GRB.CONTINUOUS,
            name="X"
        )


        # Slot completion
        C = model.addVars(
            I,
            K,
            lb=0.0,
            vtype=GRB.CONTINUOUS,
            name="C"
        )


        # Setup time
        O = model.addVars(
            I,
            K,
            lb=0.0,
            vtype=GRB.CONTINUOUS,
            name="O"
        )


        # ====================================================
        # OBJECTIVE
        # ====================================================

        model.setObjective(
            C[g, m],
            GRB.MINIMIZE
        )


        # ====================================================
        # EACH JOB-FAMILY ASSIGNED ONCE
        # ====================================================

        for family in P:

            model.addConstr(
                gp.quicksum(
                    W[
                        slot,
                        family
                    ]
                    for slot in I
                )
                == 1
            )


        # ====================================================
        # EACH SLOT GETS ONE JOB-FAMILY
        # ====================================================

        for slot in I:

            model.addConstr(
                gp.quicksum(
                    W[
                        slot,
                        family
                    ]
                    for family in P
                )
                == 1
            )


        # ====================================================
        # ONE TRANSITION PER SLOT
        # ====================================================

        for transition_position in range(
            0,
            g
        ):

            model.addConstr(
                gp.quicksum(
                    A[
                        transition_position,
                        previous_family,
                        next_family
                    ]
                    for previous_family in P0
                    for next_family in P
                    if (
                        previous_family
                        != next_family
                    )
                )
                == 1
            )


        # ====================================================
        # FIRST FAMILY
        # ====================================================

        for next_family in P:

            model.addConstr(
                A[
                    0,
                    0,
                    next_family
                ]
                ==
                W[
                    1,
                    next_family
                ]
            )


        # ====================================================
        # REAL TRANSITIONS
        # ====================================================

        for transition_position in range(
            1,
            g
        ):

            for previous_family in P:

                for next_family in P:

                    if (
                        previous_family
                        == next_family
                    ):
                        continue


                    model.addConstr(
                        A[
                            transition_position,
                            previous_family,
                            next_family
                        ]
                        <=
                        W[
                            transition_position,
                            previous_family
                        ]
                    )


                    model.addConstr(
                        A[
                            transition_position,
                            previous_family,
                            next_family
                        ]
                        <=
                        W[
                            transition_position + 1,
                            next_family
                        ]
                    )


                    model.addConstr(
                        A[
                            transition_position,
                            previous_family,
                            next_family
                        ]
                        >=
                        (
                            W[
                                transition_position,
                                previous_family
                            ]
                            +
                            W[
                                transition_position + 1,
                                next_family
                            ]
                            - 1
                        )
                    )


        # ====================================================
        # INVALID TRANSITIONS
        # ====================================================

        for transition_position in range(
            0,
            g
        ):

            for previous_family in P0:

                for next_family in P:

                    if (
                        previous_family
                        == next_family
                        or
                        (
                            transition_position == 0
                            and previous_family != 0
                        )
                    ):

                        model.addConstr(
                            A[
                                transition_position,
                                previous_family,
                                next_family
                            ]
                            == 0
                        )


        # ====================================================
        # SETUP TIME
        # ====================================================

        for slot in I:

            for machine in K:

                model.addConstr(
                    O[
                        slot,
                        machine
                    ]
                    ==
                    gp.quicksum(
                        A[
                            slot - 1,
                            previous_family,
                            next_family
                        ]
                        *
                        setup[
                            previous_family,
                            next_family,
                            machine
                        ]
                        for previous_family in P0
                        for next_family in P
                        if (
                            previous_family
                            != next_family
                        )
                    )
                )


        # ====================================================
        # FIRST MACHINE
        # ====================================================

        for slot in I:

            previous_completion = (
                0
                if slot == 1
                else C[
                    slot - 1,
                    1
                ]
            )


            model.addConstr(
                C[
                    slot,
                    1
                ]
                ==
                previous_completion
                +
                O[
                    slot,
                    1
                ]
                +
                gp.quicksum(
                    W[
                        slot,
                        family
                    ]
                    *
                    T[
                        family,
                        1
                    ]
                    for family in P
                )
            )


        # ====================================================
        # JOB TIMING
        # ====================================================

        for slot in I:

            for machine in K:

                previous_slot_completion = (
                    0
                    if slot == 1
                    else C[
                        slot - 1,
                        machine
                    ]
                )


                model.addConstr(
                    X[
                        slot,
                        1,
                        machine
                    ]
                    >=
                    previous_slot_completion
                    +
                    O[
                        slot,
                        machine
                    ]
                    +
                    gp.quicksum(
                        W[
                            slot,
                            family
                        ]
                        *
                        t[
                            family,
                            1,
                            machine
                        ]
                        for family in P
                    )
                )


        # ====================================================
        # FLOWSHOP CONSTRAINT
        # ====================================================

        for slot in I:

            for machine in K:

                if machine < 2:
                    continue


                model.addConstr(
                    X[
                        slot,
                        1,
                        machine
                    ]
                    -
                    X[
                        slot,
                        1,
                        machine - 1
                    ]
                    >=
                    gp.quicksum(
                        W[
                            slot,
                            family
                        ]
                        *
                        t[
                            family,
                            1,
                            machine
                        ]
                        for family in P
                    )
                )


        # ====================================================
        # SLOT COMPLETION
        # ====================================================

        for slot in I:

            for machine in K:

                model.addConstr(
                    C[
                        slot,
                        machine
                    ]
                    >=
                    X[
                        slot,
                        1,
                        machine
                    ]
                )


        # ====================================================
        # SOLVE
        # ====================================================

        print(
            f"Solving {folder}/{instance} "
            f"with EVERY JOB as its own family..."
        )


        model.optimize()


        status_code = model.Status

        status_name = get_status_name(
            status_code
        )

        result[
            "status_code"
        ] = status_code

        result[
            "status"
        ] = status_name

        result[
            "solution_count"
        ] = model.SolCount

        result[
            "runtime_seconds"
        ] = round(
            model.Runtime,
            4
        )

        result[
            "node_count"
        ] = round(
            model.NodeCount,
            4
        )

        result[
            "iteration_count"
        ] = round(
            model.IterCount,
            4
        )


        # Best bound
        try:

            best_bound = model.ObjBound

            if math.isfinite(
                best_bound
            ):

                result[
                    "best_bound"
                ] = round(
                    best_bound,
                    6
                )

        except gp.GurobiError:
            pass


        # ====================================================
        # NO FEASIBLE SOLUTION
        # ====================================================

        if model.SolCount == 0:

            result[
                "feasible_solution_found"
            ] = False

            print(
                f"No feasible solution found "
                f"for {folder}/{instance}"
            )

            return result


        # ====================================================
        # FEASIBLE SOLUTION EXISTS
        # ====================================================

        result[
            "feasible_solution_found"
        ] = True

        result[
            "objective"
        ] = round(
            model.ObjVal,
            6
        )


        try:

            mip_gap = model.MIPGap

            if math.isfinite(
                mip_gap
            ):

                result[
                    "gap_decimal"
                ] = round(
                    mip_gap,
                    8
                )

                result[
                    "gap_percent"
                ] = round(
                    mip_gap * 100,
                    6
                )

        except gp.GurobiError:
            pass


        # ====================================================
        # EXTRACT JOB ORDER
        # ====================================================

        sequence = extract_family_order(
            I,
            P,
            W
        )


        # Because each family = one job,
        # the family sequence IS the job sequence.

        job_sequence = [
            sequence[slot]
            for slot in I
        ]


        # Also show original family associated
        # with every job.

        original_family_sequence = [
            job_original_family[
                job
            ]
            for job in job_sequence
        ]


        result[
            "job_family_order"
        ] = json.dumps(
            job_sequence
        )

        result[
            "original_family_order"
        ] = json.dumps(
            original_family_sequence
        )


        # ====================================================
        # PRINT RESULT
        # ====================================================

        print(
            f"\nInstance: "
            f"{folder}/{instance}"
        )

        print(
            f"Status: {status_name}"
        )

        print(
            f"Best makespan found: "
            f"{model.ObjVal:.2f}"
        )


        if result[
            "gap_percent"
        ] != "":

            print(
                f"Gap: "
                f"{float(result['gap_percent']):.2f}%"
            )


        print(
            f"Runtime: "
            f"{model.Runtime:.2f} seconds"
        )


        print(
            "\nJob order "
            "(each job is its own family):"
        )

        print(
            job_sequence
        )


        print(
            "\nOriginal family of each job:"
        )

        print(
            original_family_sequence
        )


        # ====================================================
        # GANTT CHART
        # ====================================================

        if SAVE_GANTT_CHARTS:

            try:

                save_gantt_chart(
                    folder=folder,
                    instance=instance,
                    I=I,
                    K=K,
                    sequence=sequence,
                    pt=pt,
                    X=X,
                    O=O,
                    job_original_family=(
                        job_original_family
                    ),
                    output_path=gantt_path
                )


                result[
                    "gantt_file"
                ] = gantt_path


                print(
                    f"Gantt saved: "
                    f"{gantt_path}"
                )


            except Exception as chart_error:

                print(
                    "Gantt chart could not "
                    "be created:"
                )

                print(
                    chart_error
                )


        return result


    except Exception as error:

        result["status"] = "ERROR"

        result["error"] = (
            f"{type(error).__name__}: "
            f"{error}"
        )


        print(
            f"\nERROR for "
            f"{folder}/{instance}"
        )

        print(
            result["error"]
        )

        traceback.print_exc()

        return result


    finally:

        if model is not None:

            model.dispose()


# ============================================================
# MAIN
# ============================================================

def main():

    create_output_directories()

    initialize_results_csv(
        RESULTS_CSV
    )


    if not os.path.exists(
        ZIP_PATH
    ):

        print(
            f"ZIP file not found: "
            f"{ZIP_PATH}"
        )

        sys.exit(1)


    instances = find_instances_in_zip(
        ZIP_PATH,
        FOLDER
    )


    if not instances:

        print(
            f"No D instances found "
            f"in folder {FOLDER}"
        )

        sys.exit(1)


    print("=" * 70)

    print(
        "EACH JOB AS ITS OWN FAMILY EXPERIMENT"
    )

    print("=" * 70)

    print(
        f"Folder: {FOLDER}"
    )

    print(
        f"Instances: {instances}"
    )

    print(
        f"Time limit per instance: "
        f"{TIME_LIMIT} seconds"
    )

    print(
        f"Target gap: "
        f"{TARGET_MIP_GAP * 100:.2f}%"
    )

    print(
        f"Results file: "
        f"{RESULTS_CSV}"
    )


    consecutive_no_solution = 0

    attempted = 0


    for instance in instances:

        result = solve_instance(
            zip_path=ZIP_PATH,
            folder=FOLDER,
            instance=instance,
            time_limit=TIME_LIMIT,
            target_mip_gap=(
                TARGET_MIP_GAP
            )
        )


        append_result_to_csv(
            RESULTS_CSV,
            result
        )


        attempted += 1


        print(
            f"\nResult saved for "
            f"{instance}"
        )


        if result[
            "feasible_solution_found"
        ]:

            consecutive_no_solution = 0

        else:

            consecutive_no_solution += 1


        if (
            STOP_AFTER_CONSECUTIVE_NO_SOLUTION
            is not None
            and
            consecutive_no_solution
            >=
            STOP_AFTER_CONSECUTIVE_NO_SOLUTION
        ):

            print(
                "\nStopping because "
                f"{consecutive_no_solution} "
                "consecutive instances had "
                "no feasible solution."
            )

            break


    print("\n" + "=" * 70)

    print(
        "EXPERIMENT FINISHED"
    )

    print("=" * 70)

    print(
        f"Instances attempted: "
        f"{attempted}"
    )

    print(
        f"Results: "
        f"{RESULTS_CSV}"
    )

    print(
        f"Logs: "
        f"{LOG_DIRECTORY}"
    )

    if SAVE_GANTT_CHARTS:

        print(
            f"Gantt charts: "
            f"{GANTT_DIRECTORY}"
        )


if __name__ == "__main__":
    main()