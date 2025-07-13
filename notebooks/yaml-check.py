import os
import nbformat
import yaml
import re
from collections import defaultdict

def extract_yaml_from_notebook(nb_path):
    try:
        with open(nb_path, 'r', encoding='utf-8') as f:
            nb = nbformat.read(f, as_version=4)
        if nb.cells and nb.cells[0].cell_type == 'raw':
            return nb.cells[0].source
    except Exception as e:
        return None
    return None

def analyze_yaml(yaml_str):
    issues = []
    # Check number of YAML fences
    num_fences = len(re.findall(r'^---\s*$', yaml_str, flags=re.MULTILINE))
    if num_fences > 2:
        issues.append(f"❌ Multiple '---' blocks found ({num_fences}). Only one block is allowed.")

    # Check for bad indentation
    for i, line in enumerate(yaml_str.splitlines(), start=1):
        if re.match(r'^\s{1,}[a-zA-Z0-9_-]+:', line):
            issues.append(f"❌ Line {i}: Extra leading spaces before key: '{line.strip()}'")

    # Try YAML parse
    try:
        yaml.safe_load(yaml_str)
    except yaml.YAMLError as e:
        issues.append(f"❌ YAML parsing error: {str(e).strip()}")

    return issues

def scan_notebooks_for_yaml_issues(path='.'):
    issues_by_file = defaultdict(list)
    for root, _, files in os.walk(path):
        for file in files:
            if file.endswith('.ipynb'):
                full_path = os.path.join(root, file)
                yaml_str = extract_yaml_from_notebook(full_path)
                if yaml_str:
                    issues = analyze_yaml(yaml_str)
                    if issues:
                        issues_by_file[full_path] = issues
    return issues_by_file

if __name__ == "__main__":
    issues = scan_notebooks_for_yaml_issues('./notebooks')  # Change path if needed
    if not issues:
        print("✅ No YAML issues found.")
    else:
        for file, errs in issues.items():
            print(f"\n🔍 Issues in: {file}")
            for err in errs:
                print(f"   {err}")
