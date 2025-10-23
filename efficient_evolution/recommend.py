import argparse
from pathlib import Path
import datetime
import csv
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO
try:
    from efficient_evolution.amis_new import reconstruct_multi_models
except ImportError:
    from amis_new import reconstruct_multi_models


if __name__ == '__main__':
    p = argparse.ArgumentParser(
        description='Run reconstruct_multi_models using a specified model set from config/models.yaml'
    )

    p.add_argument('sequence', help='Input amino acid sequence to reconstruct')

    p.add_argument('--alpha', type=float, default=1.0,
                   help='Alpha threshold for soft reconstruction (default=1.0)')

    p.add_argument('--model-set', type=str, default='original_esm',
                   help='Model set key defined in config/models.yaml (e.g., esm2_original_debug, esm2_finetuned)')

    args = p.parse_args()

    seq = args.sequence.strip().upper()
    model_set = args.model_set

    print(f'Running reconstruct_multi_models on sequence: {seq}')
    print(f'Using model set: {model_set} (defined in config/models.yaml)')

    mutations_counts, mutations_model_names = reconstruct_multi_models(
        seq,
        model_set=model_set,
        alpha=args.alpha,
        return_names=True
    )
    
    out_root = Path(__file__).resolve().parent.parent / 'output'
    ts = datetime.datetime.now().strftime('%y%m%d%H%M')
    out_dir = out_root / ts
    out_dir.mkdir(parents=True, exist_ok=True)

    seq_short = seq[:10].replace('/', '_')
    base_name = f'r{model_set}_{seq_short}'

    txt_path = out_dir / f"{base_name}.txt"
    csv_path = out_dir / f"{base_name}.csv"
    fasta_path = out_dir / f"{base_name}.fasta"

    # Write human-readable text file and CSV summary
    with open(txt_path, 'w') as fh_txt, open(csv_path, 'w', newline='') as fh_csv:
        fh_csv_writer = csv.writer(fh_csv)
        fh_txt.write(f'Sequence: {seq}\n')
        fh_txt.write(f'Model set: {model_set}\n')
        fh_txt.write(f'Alpha: {args.alpha}\n')
        fh_txt.write(f'Timestamp: {ts}\n\n')
        fh_txt.write('=== Mutations aggregated across models ===\n')

        fh_csv_writer.writerow(['pos', 'wt', 'mt', 'count', 'models'])

        for mut, count in sorted(mutations_counts.items(), key=lambda x: (-x[1], x[0])):
            pos, wt, mt = mut
            models_list = mutations_model_names.get(mut, [])
            line = f'{wt}{pos+1}{mt}\t(count={count})\tfrom {models_list}'
            print(line)
            fh_txt.write(line + '\n')
            fh_csv_writer.writerow([pos + 1, wt, mt, count, ';'.join(models_list)])

    print(f'[INFO] Wrote summary files: {txt_path} and {csv_path}')

    # === Write FASTA with wild-type and mutated sequences ===
    if False:
        seq_records = []
        # Wild-type
        seq_records.append(SeqRecord(Seq(seq),
                                    id=f"{base_name}_WT",
                                    description="wild-type reference"))

        # Mutants
        for (pos, wt, mt), count in sorted(mutations_counts.items(), key=lambda x: (-x[1], x[0])):
            if pos < 0 or pos >= len(seq):
                continue
            if seq[pos] != wt:
                print(f"[WARN] WT mismatch at {pos}: expected {wt}, found {seq[pos]}")
                continue

            mutated_seq = seq[:pos] + mt + seq[pos + 1:]
            models_list = mutations_model_names.get((pos, wt, mt), [])
            desc = f"mutation={wt}{pos+1}{mt};count={count};models={';'.join(models_list)}"

            rec = SeqRecord(Seq(mutated_seq),
                            id=f"{base_name}_{wt}{pos+1}{mt}",
                            description=desc)
            seq_records.append(rec)

        SeqIO.write(seq_records, fasta_path, "fasta")
        print(f'[INFO] Wrote mutated FASTA to: {fasta_path}')


# python recommend.py MQWQTNLPLIAILRGITPDEALAHVGAVIDAGFDAVEIPLNSPQWEKSIPQVVDAYGEQALIGAGTVLQPEQVDRLAAMGCRLIVTPNIQPEVIRRAVGYGMTVCPGCATASEAFSALDAGAQALKIFPSSAFGPDYIKALKAVLPPEVPVFAVGGVTPENLAQWINAGCVGAGLGSDLYRAGQSVERTAQQAAAFVKAYREAVK