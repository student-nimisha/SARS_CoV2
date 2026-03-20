# data/

Put your downloaded NCBI FASTA file here.

## How to get sequences.fasta

1. Go to: https://www.ncbi.nlm.nih.gov/labs/virus/vssi/#/
2. Search: SARS-CoV-2
3. Set filters:
   - Nucleotide Completeness = Complete
   - Host = Homo sapiens
   - Geographic Region = Europe
   - Collection Date = 2021-06-01 to 2021-10-31
4. Click Download All Results
5. Step 1: Select Sequence Data (FASTA) → Nucleotide
6. Step 2: Download a randomized subset → type 2000
7. Step 3: Use default → Download
8. Rename the downloaded file to: sequences.fasta
9. Place it in this folder (data/sequences.fasta)

## Without the file

If sequences.fasta is not here, the pipeline automatically
uses simulated genomes and prints a warning. All other
analyses (POLYMOD network, OxCGRT NPIs) still run normally.
