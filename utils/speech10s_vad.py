import os
import glob
import soundfile as sf
import numpy as np
from scipy.signal import resample
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from tqdm import tqdm

# ---- Params ----
SR_TARGET = 48000
MAX_SEG_LEN_SEC = 8.0
MAX_SEG_LEN_SAMPLES = int(MAX_SEG_LEN_SEC * SR_TARGET)
FOLDER = "/scratch/profdj_root/profdj0/shared_data/DNS-Challenge/datasets_fullband/noise_fullband"
OUTPUT_TXT = "/scratch/profdj_root/profdj0/sidcs/codebase/or_se/utils/segments_noise10s.txt"

# Set this True if you want exact 10s segment lines; False to only list files >=10s



def process_one(wav: str):

    try:
        audio, sr = sf.read(wav)

        # print(f"Read {wav}, {len(audio)} samples, sr={sr}")
    except Exception:
        return []  # unreadable -> skip


    # Short-circuit if shorter than 10s
    if len(audio) < int(sr*8.0):
        return []
    else:
        # print(f"Processing {wav}, {len(audio)/sr:.1f}s, sr={sr}")
        return [f"{wav}\n"]



def main():
    all_files = glob.glob(os.path.join(FOLDER, "**", "*.wav"), recursive=True)
    # for wavs in all_files:
    #     process_one(wavs)
    worker = partial(process_one,)

    results = []
    # Use up to CPU count processes; tune chunksize for fewer scheduler calls
    print(f"Using num_workers = {os.cpu_count()}")
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as ex:
        futures = [ex.submit(worker, w) for w in all_files]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
            try:
                results.extend(fut.result())
            except Exception:
                pass  # robust to any worker crash on a single file

    # Write once to avoid lock contention
    os.makedirs(os.path.dirname(OUTPUT_TXT), exist_ok=True)
    with open(OUTPUT_TXT, "w") as f:
        f.writelines(results)

if __name__ == "__main__":
    main()
