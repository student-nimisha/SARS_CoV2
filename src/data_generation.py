"""


This script uses three real datasets:

1. Genomic Data → NCBI Virus Database
   (You need to download sequences.fasta manually)

2. Contact Data → POLYMOD study (Mossong et al. 2008)
   Embedded directly in the code

3. NPI Timeline → Oxford OxCGRT Denmark dataset
   Embedded values for Jun–Oct 2021

Steps to download sequences.fasta:
1. Go to: https://www.ncbi.nlm.nih.gov/labs/virus/vssi/#/
2. Search: SARS-CoV-2
3. Apply filters:
   - Complete sequences
   - Region: Europe / Denmark
   - Date: 2021-06-01 to 2021-10-31
4. Download FASTA file
5. Save as sequences.fasta
6. Run: python main.py
"""

import numpy as np
import pandas as pd
import os, re
from datetime import date, timedelta

RNG = np.random.default_rng(42)

FASTA_FILE       = "sequences.fasta"
TARGET_SEQUENCES = 2000
GENOME_SUBSAMPLE = 300
START_DATE       = date(2021, 6, 1)
END_DATE         = date(2021, 10, 31)

AGE_GROUPS  = ["0-10", "11-18", "19-39", "40-59", "60+"]
AGE_WEIGHTS = [0.115, 0.095, 0.275, 0.290, 0.225]

# POLYMOD contact matrix (European average)
POLYMOD_RAW = np.array([
    [3.07, 0.91, 0.98, 1.77, 1.67, 0.74, 0.29, 0.10],
    [0.91, 7.65, 1.22, 0.79, 1.39, 1.07, 0.29, 0.07],
    [0.81, 0.91, 5.08, 1.49, 0.95, 0.87, 0.29, 0.12],
    [1.19, 0.63, 1.23, 4.57, 1.52, 0.68, 0.34, 0.13],
    [1.01, 0.97, 0.73, 1.50, 3.87, 1.01, 0.44, 0.14],
    [0.57, 0.89, 0.76, 0.62, 1.20, 2.95, 0.61, 0.22],
    [0.30, 0.32, 0.31, 0.38, 0.61, 0.76, 1.83, 0.35],
    [0.12, 0.10, 0.16, 0.18, 0.23, 0.34, 0.47, 0.91],
], dtype=float)

# Convert 8 age groups → 5 groups
def _collapse_polymod(M):
    groups = [[0], [1], [2,3], [4,5], [6,7]]
    C = np.zeros((5,5))
    for i,ri in enumerate(groups):
        for j,cj in enumerate(groups):
            C[i,j] = M[np.ix_(ri,cj)].mean()
    return C / C.sum(axis=1, keepdims=True)

POLYMOD_5 = _collapse_polymod(POLYMOD_RAW)

# OxCGRT Denmark values (simplified)
OXCGRT_DNK = [
    ("2021-06-01", 0, 0, 1, 0, 2, 1),
    ("2021-06-15", 0, 0, 1, 0, 2, 1),
    ("2021-07-01", 0, 0, 0, 0, 1, 0),
    ("2021-08-01", 0, 0, 1, 0, 1, 0),
    ("2021-08-15", 1, 0, 1, 0, 1, 1),
    ("2021-09-01", 1, 0, 2, 0, 2, 1),
    ("2021-09-15", 1, 0, 2, 0, 2, 2),
    ("2021-10-01", 1, 1, 2, 0, 2, 2),
    ("2021-10-15", 2, 1, 3, 0, 2, 2),
    ("2021-10-31", 2, 1, 3, 0, 2, 2),
]
OXCGRT_MAX = [3, 3, 4, 3, 3, 4]

SCHOOL_HOLIDAYS = [("2021-06-25","2021-08-08"), ("2021-10-11","2021-10-17")]


# -------------------------------
# 1. Load FASTA sequences
# -------------------------------

def load_fasta(path):
    records, cur_id, cur_seq = [], None, []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line: continue
            if line.startswith(">"):
                if cur_id: records.append((cur_id, "".join(cur_seq)))
                cur_id, cur_seq = line[1:].split()[0], []
            else:
                cur_seq.append(line.upper())
    if cur_id: records.append((cur_id, "".join(cur_seq)))
    print(f"  Parsed {len(records)} sequences")
    return records

def extract_date(header):
    m = re.search(r"(\d{4}-\d{2}-\d{2})", header)
    if m:
        try: return date.fromisoformat(m.group(1))
        except: pass
    m = re.search(r"(\d{4}-\d{2})", header)
    if m:
        try:
            y,mo = m.group(1).split("-")
            return date(int(y),int(mo),15)
        except: pass
    return None

def sequences_to_matrix(records, n_keep=TARGET_SEQUENCES, n_pos=GENOME_SUBSAMPLE):
    base_map = {"A":0,"T":1,"G":2,"C":3}
    complete = [(s,q) for s,q in records if len(q)>=25000] or records

    if len(complete) > n_keep:
        idx = RNG.choice(len(complete), size=n_keep, replace=False)
        complete = [complete[i] for i in sorted(idx)]

    min_len   = min(len(q) for _,q in complete)
    positions = np.linspace(0, min_len-1, n_pos, dtype=int)

    G = np.zeros((len(complete), n_pos), dtype=np.int8)
    ids, dates = [], []

    for i,(sid,seq) in enumerate(complete):
        ids.append(sid)
        dates.append(extract_date(sid))
        for j,p in enumerate(positions):
            G[i,j] = base_map.get(seq[p], 0)

    return G, ids, dates


# -------------------------------
# 2. Build population
# -------------------------------

def build_population(seq_ids, seq_dates):
    n = len(seq_ids)

    test_dates = []
    for d in seq_dates:
        if d and START_DATE <= d <= END_DATE:
            test_dates.append(d)
        else:
            test_dates.append(START_DATE + timedelta(
                days=int(RNG.integers(0,(END_DATE-START_DATE).days))))

    age_groups = list(RNG.choice(AGE_GROUPS, size=n, p=AGE_WEIGHTS))
    regions    = list(RNG.choice(
        ["Hovedstaden","Midtjylland","Nordjylland","Syddanmark","Sjælland"],
        size=n))

    vacc = []
    for ag, td in zip(age_groups, test_dates):
        if td < date(2021,7,1):
            vacc.append("one_dose" if ag in ["60+","40-59"] else "unvaccinated")
        else:
            vacc.append("two_doses")

    VACC_MULTIPLIER = {"unvaccinated": 1.00, "one_dose": 0.845, "two_doses": 0.765}
    vacc_mult = [VACC_MULTIPLIER[v] for v in vacc]

    pop = pd.DataFrame({
        "individual_id":  range(n),
        "sequence_id":    seq_ids,
        "age_group":      age_groups,
        "region":         regions,
        "test_date":      test_dates,
        "variant":        "Delta",
        "vacc_status":    vacc,
        "vacc_mult":      vacc_mult,
    })

    return pop


# -------------------------------
# Fallback if FASTA not found
# -------------------------------

def _simulated_fallback(n):
    print("Using simulated genomes (FASTA not found)")
    consensus = RNG.integers(0,4,size=GENOME_SUBSAMPLE,dtype=np.int8)
    G = np.zeros((n,GENOME_SUBSAMPLE),dtype=np.int8)
    for i in range(n):
        G[i] = consensus
    return G


# -------------------------------
# Main function
# -------------------------------

def generate_all(fasta_path=None):

    if fasta_path is None:
        fasta_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), FASTA_FILE)

    if os.path.exists(fasta_path):
        records = load_fasta(fasta_path)
        genomes,ids,dts  = sequences_to_matrix(records)
        pop = build_population(ids, dts)
    else:
        n       = TARGET_SEQUENCES
        genomes = _simulated_fallback(n)
        ids     = [f"SIM_{i:05d}" for i in range(n)]
        pop     = build_population(ids, [None]*n)

    return pop, genomes


if __name__ == "__main__":
    pop, genomes = generate_all()