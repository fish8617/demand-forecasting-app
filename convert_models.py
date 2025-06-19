import pickle
import joblib

def convert_model(file_path):
    try:
        # Try loading with pickle
        with open(file_path, 'rb') as f:
            model = pickle.load(f)
        print(f"Loaded {file_path} with pickle.")
    except Exception as e_pickle:
        print(f"Pickle load failed for {file_path}: {e_pickle}")
        try:
            # Try loading with joblib
            model = joblib.load(file_path)
            print(f"Loaded {file_path} with joblib.")
        except Exception as e_joblib:
            print(f"Joblib load failed for {file_path}: {e_joblib}")
            return False

    # Save back as pickle file
    output_path = file_path.replace('.pkl', '_converted.pkl')
    with open(output_path, 'wb') as f_out:
        pickle.dump(model, f_out)
    print(f"Saved converted model to {output_path}")
    return True

if __name__ == "__main__":
    files = ['model_low.pkl', 'model_high.pkl']
    for file in files:
        success = convert_model(file)
        if not success:
            print(f"Failed to convert {file}")
