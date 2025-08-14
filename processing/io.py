import os

def ensure_results_dir(path):
    os.makedirs(path, exist_ok=True)
    return path

def _fmt_number(x):
    try:
        xf = float(x)
        if xf.is_integer():
            return str(int(xf))
        return f"{xf:.15g}"
    except Exception:
        return str(x)

def save_result(operation, indices, values, results_dir):
    os.makedirs(results_dir, exist_ok=True)
    result_file = os.path.join(results_dir, f"{operation}-result.txt")
    with open(result_file, 'w') as f:
        f.write("0\n")
        f.write("0\n")
        f.write(f"{len(indices)}\n")
        for i, v in zip(indices, values):
            f.write(f"{_fmt_number(i)} {_fmt_number(v)}\n")
    return result_file
