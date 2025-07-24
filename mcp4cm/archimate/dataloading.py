import os
import json
from typing import List, Optional
from tqdm.auto import tqdm

from mcp4cm.base import Dataset, Model



class ArchimateModel(Model):
    """
    Class representing an ArchiMate model.
    
    This class extends the base Model class with ArchiMate-specific attributes
    and functionality.
    """
    
    language: Optional[str] = None
    names_with_types: Optional[List[str]] = None
    names_with_layers_and_types: Optional[List[str]] = None


class ArchimateDataset(Dataset):
    models: List[ArchimateModel]
    

def get_hash_str(string):
    """
    Generate a hash string from the given string.
    
    Args:
        string (str): The input string to hash.
        
    Returns:
        str: The hash string.
    """
    return str(hash(string))


def load_dataset(dataset_dir: str) -> ArchimateDataset:
    """
    Load the ArchiMate dataset from the specified directory.
    This function reads ArchiMate model files from a dataset directory, processes them, and returns an ArchimateDataset object.
    Args:
        dataset_dir (str): Path to the dataset directory.
    Returns:
        ArchimateDataset: The loaded dataset containing ArchiMate models.
    """
    
    data_path = os.path.join(dataset_dir, 'processed-models')
    model_dirs = os.listdir(data_path)
    models = list()
    for model_dir in tqdm(model_dirs, desc=f'Loading Archimate Dataset @ {dataset_dir}'):
        model_dir = os.path.join(data_path, model_dir)
        if os.path.isdir(model_dir):
            model_file = os.path.join(model_dir, 'model.json')
            if os.path.isdir(model_dir):
                    model_file = os.path.join(model_dir, 'model.json')
                    if os.path.exists(model_file):
                        model = json.load(open(model_file))
                        model_id = model['identifier'].split('/')[-1]
                        names = [e['name'] for e in model['elements']]
                        names_with_types = [f"{e['name']}:{e['type']}" for e in model['elements']]
                        names_with_types_and_layers = [f"{e['name']}:{e['type']}:{e['layer']}" for e in model['elements']]
                        
                        models.append(
                            ArchimateModel(
                                id=model_id,
                                hash=get_hash_str("\n".join(names)),
                                file_path=model_file,
                                model_json=model,
                                model_txt="\n".join(names),
                                tags=model['tags'],
                                language=model['language'],
                                names=names,
                                names_with_types=names_with_types,
                                names_with_layers_and_types=names_with_types_and_layers,
                            )
                        )
    dataset = ArchimateDataset(
        name="Archimate",
        models=models
    )
    return dataset
