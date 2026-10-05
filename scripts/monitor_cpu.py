"""Run an experiment and report its CPU use as % of the whole machine, plus peak RAM.

Used to check the compute limit (fam/compute.py) is respected:
    python scripts/monitor_cpu.py experiments/01_num_addon_services.py
prints e.g. "CPU avg 25% | p95 29% | max 29% of machine | peak RAM 1.15 GB".
"""

import subprocess
import sys
import time

import psutil

proc = subprocess.Popen([sys.executable] + sys.argv[1:])
p = psutil.Process(proc.pid)
cores = psutil.cpu_count()
samples, peak_ram = [], 0.0
while proc.poll() is None:
    try:
        tree = [p] + p.children(recursive=True)
        for q in tree:
            q.cpu_percent(None)
        time.sleep(1)
        cpu = sum(q.cpu_percent(None) for q in tree if q.is_running()) / cores
        ram = sum(q.memory_info().rss for q in tree if q.is_running()) / 1024**3
    except psutil.NoSuchProcess:
        break
    samples.append(cpu)
    peak_ram = max(peak_ram, ram)
busy = sorted(samples)
p95 = busy[int(len(busy) * 0.95) - 1] if busy else 0
print(f"\n[monitor] {len(samples)}s | CPU avg {sum(samples) / max(len(samples), 1):.0f}% | p95 {p95:.0f}% "
      f"| max {max(samples, default=0):.0f}% of machine | peak RAM {peak_ram:.2f} GB")
