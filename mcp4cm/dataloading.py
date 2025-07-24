from mcp4cm.uml.dataloading import load_dataset as load_uml_dataset
from mcp4cm.archimate.dataloading import load_dataset as load_archimate_dataset
from mcp4cm.base import DatasetType, Dataset


def load_dataset(
    dataset_type: str, 
    path: str = 'data/modelset',
    uml_type: str = 'genmymodel',
    language_csv_path: str = 'categories_uml.csv'
) -> Dataset:
    """
    Load a dataset based on the dataset type and path.
    
    This function serves as a central entry point for loading different types 
    of model datasets. It dispatches to specific loaders based on the dataset type.
    
    Args:
        dataset_type (str): The type of dataset to load. Currently supports "modelset".
        path (str): The path to the dataset directory. Defaults to 'modelset'.
        uml_type (str): The type of UML models to load. Currently only supports 'genmymodel'.
        language_csv_path (str): Path to a CSV file containing language information for models.
            Defaults to 'categories_uml.csv'.
    
    Returns:
        Dataset: The loaded dataset containing models.
        
    Raises:
        ValueError: If an unknown dataset type is provided.
        FileNotFoundError: If the dataset path or language CSV file does not exist.
    
    Example:
        >>> dataset = load_dataset("modelset", path="path/to/modelset")
        >>> print(f"Loaded {len(dataset.models)} models")
    """
    if dataset_type == DatasetType.MODELSET:
        return load_uml_dataset(path, uml_type, language_csv_path)
    elif dataset_type == DatasetType.EAMODELSET:
        return load_archimate_dataset(path)
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
    