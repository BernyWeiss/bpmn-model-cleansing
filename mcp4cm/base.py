from enum import Enum
from pydantic import BaseModel
from typing import List, Optional, Union, Dict

class Model(BaseModel):
    """
    Base class for model objects.
    
    This class represents a single model within a dataset and contains
    common attributes shared by all model types.
    
    Attributes:
        id (str): Unique identifier for the model.
        file_path (str): Path to the model file on disk.
        hash (str): Hash value of the model for duplicate detection.
        model_json (Optional[Union[List, Dict]]): Model data in JSON format, if available.
        model_xmi (Optional[str]): Model data in XMI format, if available.
        model_txt (Optional[str]): Model data in plain text format, if available.
        category (Optional[str]): Category or classification of the model.
        tags (Optional[List[str]]): List of tags associated with the model.
    """
    id: str
    file_path: str
    hash: str
    language: Optional[str] = None
    model_json: Optional[Union[List, Dict]] = None
    model_xmi: Optional[str] = None
    model_txt: Optional[str] = None
    names: Optional[List[str]] = None
    category: Optional[str] = None
    tags: Optional[List[str]] = None
    
    
    def __str__(self) -> str:
        """
        String representation of the model.
        
        Returns:
            str: Model ID and file path.
        """
        
        category_str = f"({self.category})" if self.category else ""
        tags_str = f"[{', '.join(self.tags)}]" if self.tags else ""
        names_str = f"names={len(self.names)}" if self.names else ""
        
        return f"Model({names_str}{category_str} {tags_str})"


    def __repr__(self) -> str:
        """
        String representation of the model for debugging.
        
        Returns:
            str: Model ID and file path.
        """
        names_str = f"names={len(self.names)}" if self.names else ""
        category_str = ", " + f"({self.category})" if self.category else ""
        tags_str = ", " + f"[{', '.join(self.tags)}]" if self.tags else ""
        
        
        return f"Model({names_str}{category_str}{tags_str})"


class Dataset(BaseModel):
    """
    Base class for datasets.
    
    This class represents a collection of models and provides common
    attributes and methods for working with model datasets.
    
    Attributes:
        name (str): Name of the dataset.
        models (List[Model]): List of models contained in the dataset.
    """
    name: str
    models: List[Model]
    
    def __getitem__(self, index: int) -> Model:
        """
        Get a model by its index.
        
        Args:
            index (int): Index of the model to retrieve.
        
        Returns:
            Model: The model at the specified index.
        """
        return self.models[index]


    def __len__(self) -> int:
        """
        Get the number of models in the dataset.
        
        Returns:
            int: Number of models in the dataset.
        """
        return len(self.models)
    
    
    def __str__(self) -> str:
        """
        String representation of the dataset.
        
        Returns:
            str: Name of the dataset and number of models it contains.
        """
        return f"Dataset(name={self.name}, models={len(self.models)})"


    def __repr__(self) -> str:
        """
        String representation of the dataset for debugging.
        
        Returns:
            str: Name of the dataset and number of models it contains.
        """
        return f"Dataset(name={self.name}, models={len(self.models)})"


    def __iter__(self):
        """
        Iterate over the models in the dataset.
        
        Yields:
            Model: Each model in the dataset.
        """
        return iter(self.models)

    @staticmethod
    def apply_filters(dataset, filters: List[callable], verbose=False) -> 'Dataset':
        """
        Apply a list of filters to the dataset.
        
        Args:
            filters (List[callable]): List of filter functions to apply.
        
        Returns:
            Dataset: A new dataset containing only models that pass all filters
        """
        for filter_func in filters:
            if verbose:
                print(f"Applying filter: {filter_func.__name__}")
                print("Before filter:", len(dataset))
            
            filter_func(dataset, inplace=True)
            if verbose:
                print("After filter:", len(dataset))
        
        return dataset
    
    @staticmethod
    def to_csv(dataset: 'Dataset', fp: str):
        """
        Save the dataset to a CSV file.
        
        Args:
            fp (str): File path to save the CSV.
        """
        import pandas as pd
        
        data = [model.model_dump() for model in dataset.models]
        df = pd.DataFrame(data)
        df.to_csv(fp, index=False)
    

class DatasetType(Enum):
    """
    Enum for different dataset types.
    
    This enumeration defines the valid dataset types that can be loaded
    by the library. Currently supports:
    
    - MODELSET: Standard model dataset format
    """
    MODELSET = "modelset"
    EAMODELSET = "eamodelset"
