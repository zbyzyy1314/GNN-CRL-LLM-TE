"""
Convert Abilene TM files to CFR-RL compatible format.
Merges individual 5-min files into a single training/test file.

Format conversion:
  - Input: comma-separated, Gbytes/sec, with 13 header lines
  - Output: space-separated, kbps-equivalent (like convert_geant.py)
  - Scale: file_value = original_kbps * 375 (matching convert_geant.py)
"""

import os
import numpy as np

MEASURED_DIR = 'Abilene/2004/Measured'
OUT_DIR = 'data'
SCALE = 100.0
SAMPLES = None  # None = all files; set to e.g. 20000 for subset

def main():
    # Collect all TM files
    files = sorted(f for f in os.listdir(MEASURED_DIR) if f.startswith('tm.') and f.endswith('.dat'))
    if SAMPLES:
        files = files[:SAMPLES]
    print(f'[*] Processing {len(files)} TM files...')

    matrices = []
    for i, fname in enumerate(files):
        fpath = os.path.join(MEASURED_DIR, fname)
        with open(fpath) as f:
            lines = f.readlines()

        # Skip 13 header lines
        data_lines = [l for l in lines if not l.startswith('#')]
        if len(data_lines) != 12:
            continue

        # Parse 12×12 matrix
        tm = np.zeros((12, 12))
        for r, line in enumerate(data_lines):
            vals = [float(v.strip()) for v in line.strip().split(',')]
            tm[r, :] = vals[:12]

        # Zero diagonal (self-traffic)
        np.fill_diagonal(tm, 0)

        # Convert Gbytes/sec → kbps → file_format
        # 1 Gbyte/sec = 8 Gbps = 8,000,000 kbps
        tm_kbps = tm * 8_000_000

        # Apply same scaling as convert_geant.py:
        # file_value = original_kbps * 375 (for scale=100)
        conversion = 300 * 1000 / (SCALE * 8)  # = 375
        tm_file = tm_kbps * conversion

        matrices.append(tm_file)

        if (i+1) % 5000 == 0:
            print(f'  {i+1}/{len(files)}')

    matrices = np.stack(matrices)  # (T, 12, 12)
    print(f'[*] Total: {matrices.shape[0]} valid TMs, shape: {matrices.shape}')

    # Train/test split
    split = int(matrices.shape[0] * 0.8)
    train = matrices[:split]
    test = matrices[split:]

    # Write
    for name, data in [('AbileneTM', train), ('AbileneTM2', test)]:
        path = os.path.join(OUT_DIR, name)
        print(f'[*] Writing {path} ({data.shape[0]} TMs)...')
        with open(path, 'w') as f:
            for i in range(data.shape[0]):
                flat = data[i].flatten()
                line = ' '.join(f'{v:.6f}' for v in flat)
                f.write(line + '\n')
        size_mb = os.path.getsize(path) / (1024*1024)
        print(f'    Done: {size_mb:.1f} MB')

    print('[✓] Done!')

if __name__ == '__main__':
    main()
