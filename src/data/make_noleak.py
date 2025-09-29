# src/data/make_noleak.py
import re, argparse, pandas as pd

# LEAKY_NAME_PAT = re.compile(r"(fraud|label|target)", re.I)
LEAKY_NAME_PAT = re.compile(r"(fraud|label|target|ae_|recon|anomaly|cluster)", re.I)

NON_FEATURE_IDS = {"transaction_id","card_id","merchant_id","device_id","ip","timestamp"}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    df = pd.read_parquet(args.input)

    # drop obvious IDs
    drop = set(c for c in df.columns if c in NON_FEATURE_IDS)

    # drop anything that *looks* like it encodes the label (besides the label itself)
    drop |= set(c for c in df.columns if c != "fraud" and LEAKY_NAME_PAT.search(c))

    kept = [c for c in df.columns if c not in drop]
    df[kept].to_parquet(args.out, index=False)
    print(f"OK no-leak -> {args.out} {df[kept].shape}  dropped: {sorted(drop)}")

if __name__ == "__main__":
    main()
