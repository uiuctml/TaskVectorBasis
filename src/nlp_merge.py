import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import SpectralClustering
from scipy.linalg import eigh
from transformers import RobertaModel
from sklearn.metrics.pairwise import cosine_similarity
import json
import pandas as pd

def select_roberta_parameters(model):
    params = {}
    for n, p in model.named_parameters():
        if n.startswith("roberta.encoder.layer"):
            params[n] = p
    return params

def load_finetuned_checkpoint(checkpoint_path, model_type):
    if model_type == "classification":
        from transformers import RobertaForSequenceClassification
        model = RobertaForSequenceClassification.from_pretrained(checkpoint_path)
    elif model_type == "multiple_choice":
        from transformers import RobertaForMultipleChoice
        model = RobertaForMultipleChoice.from_pretrained(checkpoint_path)
    elif model_type == 'mlm':
        from transformers import RobertaForMaskedLM
        model = RobertaForMaskedLM.from_pretrained(checkpoint_path)
    else:
        raise ValueError("Unsupported model type")
    encoder_params = select_roberta_parameters(model)
    return encoder_params

def load_task_vector(task_path, pretrained_model, model_type):
    finetuned_params = load_finetuned_checkpoint(task_path, model_type)
    return create_task_vector(pretrained_model, finetuned_params) 

def create_task_vector(pretrained_model, finetuned_params):
    pretrained_params = select_roberta_parameters(pretrained_model)
    task_vector = {}
    for name, param_pretrained in pretrained_params.items():
        if name in finetuned_params:
            task_vector[name] = finetuned_params[name].data - param_pretrained.data
    return task_vector

def get_pretrained_model():
    pretrained_checkpoint_path = f"/data/common/cindy2000_sh/tangent_task_arithmetic/{model_name}/pretrained_backbone_checkpoint.pt"
    if os.path.exists(pretrained_checkpoint_path):
        print("Loading pretrained backbone checkpoint.")
        return torch.load(pretrained_checkpoint_path)
    else:
        print("Saving pretrained backbone checkpoint.")
        from transformers import RobertaForMaskedLM
        model = RobertaForMaskedLM.from_pretrained(model_name) 
        torch.save(model, pretrained_checkpoint_path)
        return model

def save_head():
    base_dir = f"/data/common/cindy2000_sh/tangent_task_arithmetic/{model_name}"
    task_folders = [
        folder for folder in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, folder))
        and not (folder == "swag" or folder.startswith("winogrande") or folder == "superglue_wsc" or folder == 'pubmed_qa')
    ]

    for task_name in task_folders:
        task_path = os.path.join(base_dir, task_name)

        checkpoint_folders = [f for f in os.listdir(task_path) if f.startswith("checkpoint-")]
        if not checkpoint_folders:
            continue
        last_step_num = max(int(folder.split("-")[1]) for folder in checkpoint_folders)
        checkpoint_path = os.path.join(task_path, f"checkpoint-{last_step_num}")

        _, head_params = load_finetuned_checkpoint(checkpoint_path, model_type="classification")

        torch.save(head_params, f"{checkpoint_path}/head_parameters.pt")
        print(f"Saved head parameters for task {task_name} at checkpoint {last_step_num}.") 


def compute_cosine_similarity_matrix(task_vectors, save_path):
    if os.path.exists(save_path):
        print(f"Loading existing cosine similarity matrix from {save_path}")
        cosine_matrix = np.load(save_path)
    else:
        print("Computing cosine similarity matrix...")
        flat_vectors = [torch.cat([param.flatten() for param in vec.values()]).numpy() for vec in task_vectors]
        flat_vectors = np.stack(flat_vectors)
        cosine_matrix = cosine_similarity(flat_vectors)
        
        np.save(save_path, cosine_matrix)
        print(f"Cosine similarity matrix saved to {save_path}")
    
    return cosine_matrix


def plot_heatmap(matrix, task_names, base_dir, cluster_order, k):
    ordered_matrix = matrix[cluster_order][:, cluster_order]
    ordered_task_names = [task_names[i].split('-')[0] for i in cluster_order]
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(ordered_matrix, xticklabels=ordered_task_names, yticklabels=ordered_task_names, cmap="coolwarm", annot=False)
    plt.title(f"Task Vector Cosine Similarity Heatmap (Clustered k = {k})")
    
    heatmap_path = os.path.join(base_dir, f"cosine_similarity_heatmap_clustered_{k}_{args.seed}.png")
    plt.savefig(heatmap_path)
    plt.close()
    print(f"Heatmap saved at {heatmap_path}")

def spectral_clustering_from_cosine(matrix, base_dir, task_names, k=None):
    if k is None:
        laplacian = np.diag(matrix.sum(axis=1)) - matrix
        eigenvalues, _ = eigh(laplacian)
        
        eigengaps = np.diff(eigenvalues)
        k = np.argmax(eigengaps) + 1
    
    clustering = SpectralClustering(n_clusters=k, affinity='precomputed')
    clusters = clustering.fit_predict(matrix)

    sorted_indices = sorted(range(len(task_names)), key=lambda i: clusters[i])
    sorted_task_names = [task_names[i] for i in sorted_indices]
    
    cluster_path = os.path.join(base_dir, f"cluster_assignments_{k}_{args.seed}.json")
    with open(cluster_path, "w") as f:
        json.dump(dict(zip(sorted_task_names, clusters[sorted_indices].tolist())), f)
    print(f"Cluster assignments saved at {cluster_path}")
    
    return clusters, sorted_indices

def recover_model_from_task_vector(task_vector, pretrained_model):
    model_state_dict = pretrained_model.state_dict()

    for name, delta in task_vector.items():
        if name in model_state_dict:
            model_state_dict[name] += delta
        else:
            print(f"Warning: {name} not found in model parameters; skipping.")

    pretrained_model.load_state_dict(model_state_dict)
    return pretrained_model

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed',type=int)
    args = parser.parse_args()


    model_name = 'roberta-base'
    base_dir = f"/data-4/common/cindy2000_sh/tangent_task_arithmetic/{model_name}"

    csv_path = os.path.join(f"/data/common/cindy2000_sh/tangent_task_arithmetic/{model_name}", "accuracy_comparison.csv")
    accuracy_df = pd.read_csv(csv_path)
    accuracy_df = accuracy_df[accuracy_df['fine_tune_accuracy'] >= accuracy_df['majority_label_accuracy']]

    task_folders = [
        folder for folder in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, folder))
    ]

    accuracy_dataset_names = set(accuracy_df['dataset_name'])


    task_folders = [
        folder for folder in task_folders
        if folder in accuracy_dataset_names  
    ]


    print(f"Number of datasets left after filtering: {len(task_folders)}")


    model_types = dict()

    pretrained_model = get_pretrained_model()
    
    task_vectors = []
    task_names = []
    for task_name in task_folders:
        task_path = os.path.join(base_dir, task_name, str(args.seed))
        
        checkpoint_path = os.path.join(task_path, 'best_model')
        if os.path.exists(checkpoint_path):
            task_vector = load_task_vector(checkpoint_path, pretrained_model, 'mlm') 
            task_vectors.append(task_vector)
            task_names.append(task_name)

    cosine_sim_matrix = compute_cosine_similarity_matrix(task_vectors, os.path.join(base_dir, f"cosine_similarity_matrix_{args.seed}.npy"))

    for k in [1, 3, 5, 10, 15, 20]:
        print('k', k)
        clusters, sorted_indices = spectral_clustering_from_cosine(cosine_sim_matrix, base_dir, task_names, k)
        print("Cluster assignments:", dict(zip(task_names, clusters)))
        plot_heatmap(cosine_sim_matrix, task_names, base_dir, sorted_indices, k)