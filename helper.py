import os
import json
import joblib
import zipfile

def zip_files(files, output_path):
    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file in files:
            zipf.write(file, arcname=file)


def unzip_files(zip_path, extract_to):
    with zipfile.ZipFile(zip_path, 'r') as zipf:
        zipf.extractall(extract_to)

def update_baseline_metadata(model_data):
    dataset_metadata_path = f"processed_data/{model_data['data_version']}/metadata.json"
    dataset_metadata = {}
    with open(dataset_metadata_path, 'r') as f:
        dataset_metadata = json.load(f)

    if os.path.exists("Baseline_model_metadata.json"):
        with open("Baseline_model_metadata.json", 'r') as f:
            baseline_metadata = json.load(f)
    else:
        baseline_metadata = {}

    if "Best_model" not in baseline_metadata:
        baseline_metadata["Best_model"] = {}

   
    if "test_r2" in baseline_metadata["Best_model"] and baseline_metadata["Best_model"]["test_r2"] < model_data["test_r2"]:
   
        baseline_metadata["Best_model"]["train_rmse"] = model_data["train_rmse"]
        baseline_metadata["Best_model"]["train_r2"] = model_data["train_r2"]
        baseline_metadata["Best_model"]["test_r2"] = model_data["test_r2"]
        baseline_metadata["Best_model"]["test_rmse"] = model_data["test_rmse"]
        baseline_metadata["Best_model"]["selected_feature_names"] = model_data["selected_feature_names"]
        baseline_metadata["Best_model"]["dataset_metadata"] = dataset_metadata        
        with open("Baseline_model_metadata.json", 'w') as f:
            json.dump(baseline_metadata, f, indent=4)

    else:
        # Manage history if current model is not the best
        history = {}
        if os.path.exists("history.json"):
            with open("history.json", 'r') as f:
                history = json.load(f)
        
        id = history.get("counter", 0)
        history["counter"] = id + 1
        history[id] = {
            'train_rmse': model_data["train_rmse"],
            'train_r2':   model_data["train_r2"],
            'test_rmse':  model_data["test_rmse"],
            'test_r2':    model_data["test_r2"],
            'selected_feature_names': model_data["selected_feature_names"],
            'dataset_metadata': dataset_metadata
        }

        # Save updated history
        with open("history.json", 'w') as f:
            json.dump(history, f, indent=4)


def load_model(file_path):
    try:
        model = joblib.load(file_path)
        return model
    except FileNotFoundError:
        print(f"Error: The file at {file_path} was not found.")
    except joblib.externals.loky.process_executor.TerminatedWorkerError:
        print("Error: The file could not be loaded.")
    except Exception as e:
        print(f"An error occurred: {e}")

def unzip_all_folders():
    for path_names in ["processed_data/0/train.csv.zip", "processed_data/1/train.csv.zip","split/train.csv.zip","split/val.csv.zip" , "processed_data/0/val.csv.zip", "processed_data/1/val.csv.zip"]:
        unzip_files(path_names,path_names[:-4])

def zip_all_folder():
      for path_names in ["processed_data/0/train.csv", "processed_data/1/train.csv","split/train.csv","split/val.csv" , "processed_data/0/val.csv", "processed_data/1/val.csv"]:
        zip_files(path_names,path_names+".zip")



if __name__ == "__main__":
    ...


  
    
    
   

