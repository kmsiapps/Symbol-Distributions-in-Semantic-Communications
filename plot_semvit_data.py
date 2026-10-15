#%%
import os
import csv
import numpy as np
import matplotlib.pyplot as plt

# Define the SNRs and compression ratios you want to plot
SNRs = [0, 10, 20, 'inf']
Ms   = [256, 512, 1024]
#%%
fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(12, 6), sharex=True)

# Loop over rows (compression ratios) and columns (SNRs)
for row_idx, M in enumerate(Ms):
    for col_idx, snr in enumerate(SNRs):
        ax = axes[row_idx, col_idx]

        # Construct filenames
        noKL_filename_train = f"SemViT-{snr}dB-{M}_train.csv"
        KL_filename_train   = f"SemViT-{snr}dB-KL-{M}_train.csv"
        newKL_filename_train   = f"nSemViT_{snr}dB_{M}sym_train.csv"
        noKL_filename_valid = f"SemViT-{snr}dB-{M}_validation.csv"
        KL_filename_valid   = f"SemViT-{snr}dB-KL-{M}_validation.csv"
        newKL_filename_valid   = f"nSemViT_{snr}dB_{M}sym_validation.csv"

        # Read w/o KL data
        with open(os.path.join('semvit-csv-data', noKL_filename_train), 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            rows = list(reader)[1:]   # Skip header row
            x_noKL_train = np.array(rows, dtype=float)[:, 1]
            y_noKL_train = np.array(rows, dtype=float)[:, 2]
            
        with open(os.path.join('semvit-csv-data', noKL_filename_valid), 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            rows = list(reader)[1:]   # Skip header row
            x_noKL_val = np.array(rows, dtype=float)[:, 1]
            y_noKL_val = np.array(rows, dtype=float)[:, 2]

        # Read w/ KL data
        with open(os.path.join('semvit-csv-data', KL_filename_train), 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            rows = list(reader)[1:]   # Skip header row
            x_KL_train = np.array(rows, dtype=float)[:, 1]
            y_KL_train = np.array(rows, dtype=float)[:, 2]
        
        with open(os.path.join('semvit-csv-data', KL_filename_valid), 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            rows = list(reader)[1:]   # Skip header row
            x_KL_val = np.array(rows, dtype=float)[:, 1]
            y_KL_val = np.array(rows, dtype=float)[:, 2]

        # Read w/ new KL data
        with open(os.path.join('semvit-csv-data', newKL_filename_train), 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            rows = list(reader)[1:]   # Skip header row
            x_newKL_train = np.array(rows, dtype=float)[:, 1]
            y_newKL_train = np.array(rows, dtype=float)[:, 2]
        
        with open(os.path.join('semvit-csv-data', newKL_filename_valid), 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            rows = list(reader)[1:]   # Skip header row
            x_newKL_val = np.array(rows, dtype=float)[:, 1]
            y_newKL_val = np.array(rows, dtype=float)[:, 2]


        # Plot raw data with lower alpha
        ax.plot(x_noKL_train, y_noKL_train, color='tab:blue', linewidth=2, label='Baseline')
        ax.plot(x_KL_train, y_KL_train, color='tab:orange', linewidth=2, label='w/ Proposed Loss')
        ax.plot(x_newKL_train, y_newKL_train, color='tab:green', linewidth=2, label='w/ Factorized Loss')

        ax.plot(x_noKL_val, y_noKL_val, color='tab:blue', alpha=0.3, label=None)
        ax.plot(x_KL_val, y_KL_val, color='tab:orange', alpha=0.3, label=None)
        ax.plot(x_newKL_val, y_newKL_val, color='tab:green', alpha=0.3, label=None)

        # Label columns by SNR on the top row
        if row_idx == 0:
            ax.set_title(f"{snr} dB")

        # Label rows by M on the leftmost column
        if col_idx == 0:
            ax.set_ylabel(f"CBR={M/32/32/3:.2f}")
        
        a = np.quantile(y_KL_train, 0.05)
        b = np.max(y_KL_train) + np.std(y_KL_train) * 0.2
        ax.set_ylim([a, b])

# Add a legend to the first subplot (or global legend)
axes[0, 0].legend(loc='best')

# Common X and Y labels
fig.text(0.5, 0.04, 'Training Epoch', ha='center')
fig.text(0.04, 0.5, 'Reconstructed Image Quality (PSNR, dB)', va='center', rotation='vertical')

# If you want a custom y-limit or annotation, uncomment below:
# for ax in axes.flat:
#     ax.set_ylim([0, 0.25])
#     # or any range you want
# # Example of placing a vertical text or arrow on the left
# fig.text(0.02, 0.85, 'Better?', rotation='vertical', fontsize=12)

fig.tight_layout(rect=[0.05, 0.05, 1, 1])
plt.savefig('./images/CIFAR_KL.png')
plt.show

#%%
# NTSCC training curve
plt.figure(figsize=(4, 3))

# Construct filenames
noKL_filename_train = f"NTSCC_NAT_o72_train.csv"
KL_filename_train   = f"NTSCC_RRC_Lo72_KLdv_K1_train.csv"
noKL_filename_valid = f"NTSCC_NAT_o72_valid.csv"
KL_filename_valid   = f"NTSCC_RRC_Lo72_KLdv_K1_valid.csv"

# Read w/o KL data
with open(os.path.join('semvit-csv-data', noKL_filename_train), 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    rows = list(reader)[1:]   # Skip header row
    x_noKL_train = np.array(rows, dtype=float)[:, 1]
    y_noKL_train = np.array(rows, dtype=float)[:, 2]
    
with open(os.path.join('semvit-csv-data', noKL_filename_valid), 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    rows = list(reader)[1:]   # Skip header row
    x_noKL_val = np.array(rows, dtype=float)[:, 1]
    y_noKL_val = np.array(rows, dtype=float)[:, 2]

# Read w/ KL data
with open(os.path.join('semvit-csv-data', KL_filename_train), 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    rows = list(reader)[1:]   # Skip header row
    x_KL_train = np.array(rows, dtype=float)[:, 1]
    y_KL_train = np.array(rows, dtype=float)[:, 2]

with open(os.path.join('semvit-csv-data', KL_filename_valid), 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    rows = list(reader)[1:]   # Skip header row
    x_KL_val = np.array(rows, dtype=float)[:, 1]
    y_KL_val = np.array(rows, dtype=float)[:, 2]


# Plot raw data with lower alpha
plt.plot(x_noKL_train, y_noKL_train, color='tab:blue', linewidth=2, label='Baseline')
plt.plot(x_KL_train, y_KL_train, color='tab:orange', linewidth=2, label='w/ Proposed Loss')

plt.plot(x_noKL_val, y_noKL_val, color='tab:blue', alpha=0.3, label=None)
plt.plot(x_KL_val, y_KL_val, color='tab:orange', alpha=0.3, label=None)

plt.title(f"NTSCC (PSNR)")
plt.ylabel(f"Decoded Image Quality (PSNR, dB)")
plt.xlabel('Training Epoch')

# a = np.quantile(y_KL_train, 0.05)
# b = np.max(y_KL_train) + np.std(y_KL_train) * 0.2
# plt.ylim([a, b])
plt.legend(loc='best')

plt.tight_layout()
plt.savefig('./images/NTSCC_training_curves_PSNR.png')
plt.show()

#%%
# NTSCC training curve (CBR)

plt.figure(figsize=(4, 3))

# Construct filenames
noKL_filename_train = f"NTSCC_NAT_o72_train.csv"
KL_filename_train   = f"NTSCC_RRC_Lo72_KLdv_K1_train.csv"
noKL_filename_valid = f"NTSCC_NAT_o72_valid.csv"
KL_filename_valid   = f"NTSCC_RRC_Lo72_KLdv_K1_valid.csv"

# CBR
with open(os.path.join('semvit-csv-data', 'CBR_' + noKL_filename_train), 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    rows = list(reader)[1:]   # Skip header row
    x_CBR_noKL_train = np.array(rows, dtype=float)[:, 1]
    y_CBR_noKL_train = np.array(rows, dtype=float)[:, 2]
    
with open(os.path.join('semvit-csv-data', 'CBR_' + noKL_filename_valid), 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    rows = list(reader)[1:]   # Skip header row
    x_CBR_noKL_val = np.array(rows, dtype=float)[:, 1]
    y_CBR_noKL_val = np.array(rows, dtype=float)[:, 2]

# Read w/ KL data
with open(os.path.join('semvit-csv-data', 'CBR_' + KL_filename_train), 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    rows = list(reader)[1:]   # Skip header row
    x_CBR_KL_train = np.array(rows, dtype=float)[:, 1]
    y_CBR_KL_train = np.array(rows, dtype=float)[:, 2]

with open(os.path.join('semvit-csv-data', 'CBR_' + KL_filename_valid), 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    rows = list(reader)[1:]   # Skip header row
    x_CBR_KL_val = np.array(rows, dtype=float)[:, 1]
    y_CBR_KL_val = np.array(rows, dtype=float)[:, 2]

plt.plot(x_CBR_noKL_train, y_CBR_noKL_train, color='tab:blue', linewidth=2, label='Baseline')
plt.plot(x_CBR_KL_train, y_CBR_KL_train, color='tab:orange', linewidth=2, label='w/ Proposed Loss')

plt.plot(x_CBR_noKL_val, y_CBR_noKL_val, color='tab:blue', alpha=0.3, label=None)
plt.plot(x_CBR_KL_val, y_CBR_KL_val, color='tab:orange', alpha=0.3, label=None)

plt.title(f"NTSCC (Symbol Rate)")
plt.ylabel(f"Channel Bandwidth Ratio")
plt.xlabel('Training Epoch')

# a = np.quantile(y_KL_train, 0.05)
# b = np.max(y_KL_train) + np.std(y_KL_train) * 0.2
# plt.ylim([a, b])
plt.legend(loc='best')

plt.tight_layout()
plt.savefig('./images/NTSCC_training_curves_CBR.png')
plt.show()

#%%
# DeepJSCC training curve (PSNR)
plt.figure(figsize=(4, 3))

# Construct filenames
noKL_filename_train = f"SemELIC_small_10dB_log_train.csv"
KL_filename_train   = f"SemELIC_small_10dB_kl_L10000_train.csv"
noKL_filename_valid = f"SemELIC_small_10dB_log_valid.csv"
KL_filename_valid   = f"SemELIC_small_10dB_kl_L10000_valid.csv"

# Read w/o KL data
with open(os.path.join('semvit-csv-data', noKL_filename_train), 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    rows = list(reader)[1:]   # Skip header row
    x_noKL_train = np.array(rows, dtype=float)[:, 1]
    y_noKL_train = np.array(rows, dtype=float)[:, 2]
    
with open(os.path.join('semvit-csv-data', noKL_filename_valid), 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    rows = list(reader)[1:]   # Skip header row
    x_noKL_val = np.array(rows, dtype=float)[:, 1]
    y_noKL_val = np.array(rows, dtype=float)[:, 2]

# Read w/ KL data
with open(os.path.join('semvit-csv-data', KL_filename_train), 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    rows = list(reader)[1:]   # Skip header row
    x_KL_train = np.array(rows, dtype=float)[:, 1]
    y_KL_train = np.array(rows, dtype=float)[:, 2]

with open(os.path.join('semvit-csv-data', KL_filename_valid), 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    rows = list(reader)[1:]   # Skip header row
    x_KL_val = np.array(rows, dtype=float)[:, 1]
    y_KL_val = np.array(rows, dtype=float)[:, 2]

# Plot raw data with lower alpha
plt.plot(x_noKL_train, y_noKL_train, color='tab:blue', linewidth=2, label='Baseline')
plt.plot(x_KL_train, y_KL_train, color='tab:orange', linewidth=2, label='w/ Proposed Loss')

plt.plot(x_noKL_val, y_noKL_val, color='tab:blue', alpha=0.3, label=None)
plt.plot(x_KL_val, y_KL_val, color='tab:orange', alpha=0.3, label=None)

plt.title(f"DeepJSCC (PSNR)")
plt.ylabel(f"Decoded Image Quality (PSNR, dB)")
plt.xlabel('Training Epoch')

a = np.quantile(y_KL_val, 0.05)
b = np.max(y_KL_train) + np.std(y_KL_train) * 0.2
plt.ylim([a, b])
plt.legend(loc='best')

plt.tight_layout()
plt.savefig('./images/DeepJSCC_training_curves_PSNR.png')
plt.show()

# %%
