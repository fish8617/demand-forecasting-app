import pickle

def print_model_feature_names(model_path):
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        if hasattr(model, 'feature_names_in_'):
            print(f"Feature names for {model_path}: {model.feature_names_in_}")
        else:
            print(f"No feature_names_in_ attribute found in {model_path}")
    except Exception as e:
        print(f"Error loading model {model_path}: {e}")

if __name__ == "__main__":
    print_model_feature_names('model_low_converted.pkl')
    print_model_feature_names('model_high_converted.pkl')
