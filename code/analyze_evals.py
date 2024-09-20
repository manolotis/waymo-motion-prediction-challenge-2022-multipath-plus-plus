# prints some basic stats of the evaluations

import os
import numpy as np

evaluations_path = "/home/manolotis/sandbox/temporal-consistency-tests/multipathPP/evals/"

evaluation_files = sorted([os.path.join(evaluations_path, file) for file in os.listdir(evaluations_path)])
evals = [np.load(file) for file in evaluation_files]
file2eval = {}
for f, e in zip(evaluation_files, evals):
    file2eval[f] = e
n_evals = len(evals)

# keys = ['rank_switches_counter', 'top1_rank_switches', 'top1_rank_switches_ade_sum', 'top1_rank_switches_ades', 'top1_rank_switches_counter']
keys = list(evals[0].keys())
all_values = {}

counter_top1_rank_switches = 0 # is it above mean value?
counter_ade_mean = 0 # is it above mean value?
counter_all = 0

for eval in evals:
    if eval["top1_rank_switches"] > 2* 9 and eval["top1_rank_switches_ade_mean"] > 2 * 1.3 and eval["top1_rank_switches_counter"] > 2* 6.5:
        counter_all+= 1

    for key in keys:
        if key == "top1_rank_switches_ades":
            continue
        if key not in all_values:
            all_values[key] = []


        value = eval[key].item()



        if key == "top1_rank_switches" and value >= 2 * 9:
            counter_top1_rank_switches += 1

        if key == "top1_rank_switches_ade_mean" and value > 2 * 1.3:
            counter_ade_mean += 1



        all_values[key].append(value)



print("Num evaluations: ", n_evals)
for key in keys:
    if key == "top1_rank_switches_ades":
        continue
    print(key)
    print("\t-min", np.min(all_values[key]))
    print("\t-max", np.max(all_values[key]))
    print("\t-mean", np.mean(all_values[key]))
    print("\t-std", np.std(all_values[key]))


print("counter_top1_rank_switches", counter_top1_rank_switches)
print("counter_ade_mean", counter_ade_mean)
print("counter_both", counter_all)