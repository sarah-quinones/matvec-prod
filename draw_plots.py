from sys import argv
from itertools import accumulate
from pathlib import Path
from json import load, JSONDecodeError

import numpy as np

from matplotlib import pyplot as plt

Path("out_plots").mkdir(exist_ok=True)
plt.style.use("ggplot")

results = {}
for arg in argv[1:]:
    with open(arg) as f:
        try:
            data = load(f)
        except JSONDecodeError:
            continue
    data = data["benchmarks"]

    for d in data:
        # name ~ bm_method<type, rows, cols>
        # method = blaze | eigen | simd_

        method = d["name"][3:8]
        params = d["name"][9:-1].split(",")
        dtype = params[0]
        n_rows = int(params[1])
        n_cols = int(params[2])

        key = (dtype, n_cols)

        results[key] = results.get(
            key, {"time_b": [[], []], "time_e": [[], []], "time_s": [[], []],},
        )

        t = d["cpu_time"]

        if method == "blaze":
            results[key]["time_b"][0].append(n_rows)
            results[key]["time_b"][1].append(t)
        if method == "eigen":
            results[key]["time_e"][0].append(n_rows)
            results[key]["time_e"][1].append(t)
        if method == "simd_":
            results[key]["time_s"][0].append(n_rows)
            results[key]["time_s"][1].append(t)

for key in results:
    for a in ["time_s", "time_e", "time_b"]:
        results[key][a] = np.array(results[key][a])

for key in results:
    dtype, n_cols = key
    res = results[key]

    res_e = res["time_e"]
    res_b = res["time_b"]
    res_s = res["time_s"]

    sort = np.argsort(res_e[0])
    res_e[0] = res_e[0][sort]
    res_e[1] = res_e[1][sort]

    sort = np.argsort(res_b[0])
    res_b[0] = res_b[0][sort]
    res_b[1] = res_b[1][sort]

    sort = np.argsort(res_s[0])
    res_s[0] = res_s[0][sort]
    res_s[1] = res_s[1][sort]

    n_rows = res_s[0]

    n_skip = 6
    e = np.exp(np.log(res_e[1][n_skip:] / res_s[1][n_skip:]).mean())
    b = np.exp(np.log(res_b[1][n_skip:] / res_s[1][n_skip:]).mean())

    print(f"[{dtype}][n×{n_cols}][{n_cols}] => n : geometric average:")
    print(f"{e:<4.3} faster than eigen")
    print(f"{b:<4.3} faster than blaze")

    fig, axes = plt.subplots(ncols=2, figsize=(10, 5))

    abs_plot = axes[0]
    rel_plot = axes[1]

    rel_plot.plot(n_rows, res_e[1] / res_s[1])
    rel_plot.plot(n_rows, res_b[1] / res_s[1])
    rel_plot.plot(n_rows, len(n_rows) * [1])
    abs_plot.legend(["eigen", "blaze", "simd", "simd (unroll 2)", "simd (unroll 4)"])
    rel_plot.set_ylim(ymin=0)
    rel_plot.set_xlabel("n n_rows")
    rel_plot.set_title("relative time")

    abs_plot.plot(*res_e)
    abs_plot.plot(*res_b)
    abs_plot.plot(*res_s)
    abs_plot.legend(["eigen", "blaze", "simd", "simd (unroll 2)", "simd (unroll 4)"])
    abs_plot.set_ylim(ymin=0)
    abs_plot.set_xlabel("n rows")
    abs_plot.set_title("absolute time")

    fig.suptitle(f"[{dtype}][n×{n_cols}][{n_cols}] => n ")
    fig.savefig(f"out_plots/{dtype}_{n_cols}_cols.pdf")
