import subprocess

def run_script(script_name):
    print(f"\n>>> Running {script_name}...")
    result = subprocess.run(['python', script_name], capture_output=False, text=True)
    if result.returncode != 0:
        print(f"Error running {script_name}")
        return False
    return True

if __name__ == "__main__":
    scripts = [
        'data_setup.py',
        'traditional_model.py',
        'deep_model.py',
        'evaluate.py'
    ]

    for script in scripts:
        if not run_script(script):
            print("Pipeline failed at " + script)
            break
    else:
        print("\n" + "=" * 50)
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print("Final report is in evaluation_report.txt")
        print("=" * 50)