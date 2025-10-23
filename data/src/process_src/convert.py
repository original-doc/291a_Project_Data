import json
import os

# --- Configuration ---

# The directory where the source .jsonl files are located
# Based on your image, this is 'pl_dataset/argumented'
SOURCE_DIR = os.path.join('pl_dataset', 'augmented')

# The four source files to process
SOURCE_FILES = [
    'pytorch_lightning_acs_0.jsonl',
    'pytorch_lightning_acs_1.jsonl',
    'pytorch_lightning_acs_2.jsonl',
    'pytorch_lightning_acs_4.jsonl',
    'pytorch_lightning_acs_5.jsonl',
]

# The directory to save the new dataset in
OUTPUT_DIR = 'src_dataset'

# The name of the new output file (now .json for readability)
OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'filtered_data.json')

# List of path strings to filter for.
# We use raw strings (r"...") to handle the backslashes correctly.
FILTER_PATHS_RAW = [
    r"src\lightning\pytorch\callbacks",
    "core",
    "loops",
    #"strategies",
    "trainer",
    "tuner",
    "utilities",
]

# For consistent matching, we'll normalize all paths to use forward slashes
FILTER_PATHS = [p.replace('\\', '/') for p in FILTER_PATHS_RAW]

# --- Main Script ---

def create_filtered_dataset():
    """
    Loads multiple source JSONL files, filters based on path,
    and writes to a single JSON file in the target schema.
    """
    print("Starting conversion...")

    # Ensure the output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_filtered_data = []  # This list will hold all matching records
    new_index = 0
    total_lines_processed = 0

    # Loop through each source file
    for filename in SOURCE_FILES:
        source_path = os.path.join(SOURCE_DIR, filename)
        
        try:
            print(f"\nProcessing '{source_path}'...")
            lines_in_file = 0
            
            # Open the source file for reading
            with open(source_path, 'r', encoding='utf-8') as f_in:
                
                for line in f_in:
                    total_lines_processed += 1
                    lines_in_file += 1
                    try:
                        # Load the data from one line
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        print(f"Warning: Skipping malformed JSON line {lines_in_file} in {filename}")
                        continue

                    original_path = data.get('path')
                    if not original_path:
                        print(f"Warning: Skipping record {lines_in_file} in {filename} due to missing 'path'")
                        continue

                    # Normalize the path from the data for comparison
                    normalized_path = original_path.replace('\\', '/')

                    # Check if the normalized path contains ANY of the filter strings
                    if any(filter_p in normalized_path for filter_p in FILTER_PATHS):
                        
                        # --- 1. Construct the new 'text' field ---
                        meta_data_parts = [
                            f"Repo: {data.get('repo', 'N/A')}",
                            f"Path: {original_path}",
                            f"Function Name: {data.get('func_name', 'N/A')}",
                            f"Language: {data.get('language', 'N/A')}",
                            f"Partition: {data.get('partition', 'N/A')}"
                        ]
                        meta_data_str = "\n".join(meta_data_parts)

                        text_content = (
                            f"--- Meta Data ---\n"
                            f"{meta_data_str}\n\n"
                            f"--- Docstring ---\n"
                            f"{data.get('docstring', '')}\n\n"
                            f"--- Code ---\n"
                            f"{data.get('code', '')}\n\n"
                            f"--- Original String ---\n"
                            f"{data.get('original_string', '')}"
                        )

                        # --- 2. Create the new record in the target schema ---
                        new_record = {
                            "label": "src_data",
                            "file": original_path,  # The original path
                            "index": new_index,      # The new, sequential index
                            "text": text_content.strip() # The combined text block
                        }
                        
                        # --- 3. Add the new record to our master list ---
                        all_filtered_data.append(new_record)
                        new_index += 1

        except FileNotFoundError:
            print(f"Error: Source file not found at '{source_path}'")
            print("Please make sure the file exists in the specified directory.")
            continue  # Skip to the next file
        except Exception as e:
            print(f"An error occurred while processing {source_path}: {e}")
            continue # Skip to the next file

    # --- 4. Write the final combined list to the output JSON file ---
    if not all_filtered_data:
        print("\nNo matching data was found. Output file will not be created.")
        return

    try:
        print(f"\nWriting {len(all_filtered_data)} records to '{OUTPUT_FILE}'...")
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f_out:
            # Use json.dump (not dumps) to write the whole list to the file
            # indent=2 makes it human-readable (pretty-printed)
            json.dump(all_filtered_data, f_out, indent=2)
            
        print("\n--- Conversion Complete ---")
        print(f"Total lines processed from all files: {total_lines_processed}")
        print(f"Total records kept (filtered): {len(all_filtered_data)}")
        print(f"New dataset saved to: '{OUTPUT_FILE}'")

    except Exception as e:
        print(f"An error occurred while writing the final JSON file: {e}")


# Run the conversion function when the script is executed
if __name__ == "__main__":
    create_filtered_dataset()