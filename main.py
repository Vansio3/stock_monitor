# main.py - The single command to run the entire data pipeline. 

import time
import sys
import traceback
import importlib.util

# --- Configuration ---
# List of scripts and the main function to call within each script.
# This makes the pipeline easy to modify, reorder, or extend.
PIPELINE_STEPS = [
    ('1_data_collector.py', 'download_stock_data'),
    ('2_model_trainer.py', 'train_all_models'),
    ('3_automated_backtest.py', 'run_all_backtests'),
    ('4_generate_predictions.py', 'generate_all_predictions'),
    ('5_export_for_web.py', 'export_data_for_web'),
]

def run_script(script_path, function_name):
    """Dynamically loads a Python script and runs a specified function."""
    try:
        # Create a module specification from the file path
        spec = importlib.util.spec_from_file_location(script_path, script_path)
        # Create a new module based on the spec
        module = importlib.util.module_from_spec(spec)
        # Execute the module in its own namespace. This is equivalent to running the script.
        spec.loader.exec_module(module)
        
        # Get the main function from the now-loaded module
        main_function = getattr(module, function_name)
        
        # Call the function
        main_function()
        
    except FileNotFoundError:
        print(f"  ERROR: Script not found at '{script_path}'. Please check the file name.")
        raise
    except AttributeError:
        print(f"  ERROR: Function '{function_name}' not found in script '{script_path}'.")
        raise
    except Exception as e:
        print(f"  ERROR: An unexpected error occurred while running '{script_path}'.")
        raise

def run_full_pipeline():
    """
    Executes the entire data processing pipeline from start to finish.
    """
    start_time = time.time()
    print("--- 🚀 STARTING FULL DATA PIPELINE 🚀 ---")

    for i, (script, function) in enumerate(PIPELINE_STEPS):
        step_num = i + 1
        print(f"\n[STEP {step_num}/{len(PIPELINE_STEPS)}] Executing: {script}")
        try:
            run_script(script, function)
            print(f"✅ STEP {step_num} COMPLETE: {script} finished successfully.")
        except Exception as e:
            print(f"\n--- ❌ PIPELINE FAILED AT STEP {step_num} ({script}) ❌ ---")
            traceback.print_exc()
            sys.exit(1) # Exit with an error code

    end_time = time.time()
    print("\n\n--- 🎉 PIPELINE FINISHED SUCCESSFULLY 🎉 ---")
    print(f"Total execution time: {end_time - start_time:.2f} seconds.")
    print("Your 'data.json' file is now up-to-date and ready for the web.")

if __name__ == "__main__":
    run_full_pipeline()