import scipy.stats
import torch

# Experiment: steps=1470, frac=0.60 (8 samples)
exp_losses = [3.2787, 3.2796, 3.2775, 3.2796, 3.2804, 3.2795, 3.2793, 3.2809]
exp_times = [91.487, 91.421, 91.517, 91.443, 91.427, 91.677, 91.576, 91.587]

# Baseline: runs 1,2,5,6,7,8 visible (runs 3,4 interleaved/corrupted in output)
baseline_losses = [3.2789, 3.2789, 3.2793, 3.2813, 3.2806, 3.2788]
baseline_times = [92.170, 92.370, 92.276, 92.381, 92.200, 92.292]

print("=== Experiment (frac=0.60) ===")
print("p=%.4f" % scipy.stats.ttest_1samp(exp_losses, 3.28, alternative="less").pvalue)
print("losses:", torch.std_mean(torch.tensor(exp_losses)))
print("times:", torch.std_mean(torch.tensor(exp_times)))

print("\n=== Baseline ===")
print("p=%.4f" % scipy.stats.ttest_1samp(baseline_losses, 3.28, alternative="less").pvalue)
print("losses:", torch.std_mean(torch.tensor(baseline_losses)))
print("times:", torch.std_mean(torch.tensor(baseline_times)))
