import json
import os
import sqlite3
from typing import List, Optional
from tqdm.auto import tqdm
import numpy as np
import pandas as pd
from mcp4cm.base import Dataset, Model


class UMLModel(Model):
    """
    Class representing a UML model.
    
    This class extends the base Model class with UML-specific attributes
    and functionality.
    
    Attributes:
        diagram_types (Optional[List[str]]): Types of diagrams present in the model
            (e.g., 'Class Diagram', 'Activity Diagram', etc.).
        names (Optional[List[str]]): Extracted element names from the model.
        names_with_types (Optional[List[str]]): Element names with their types 
            (e.g., 'class: Customer', 'actor: User', etc.).
    """
    diagram_types: Optional[List[str]] = None
    names_with_types: Optional[List[str]] = None


class UMLDataset(Dataset):
    """
    Class representing a dataset of UML models.
    
    This class extends the base Dataset class to work specifically with
    UML models and provides UML-specific operations.
    
    Attributes:
        models (List[UMLModel]): List of UML models in the dataset.
    """
    models: List[UMLModel]

    def __getitem__(self, index: int) -> UMLModel:
        """
        Get a UML model by index.
        
        Args:
            index (int): Index of the model to retrieve.
        
        Returns:
            UMLModel: The UML model at the specified index.
        """
        return self.models[index]
    
    

def load_dataset(
    dataset_path: str = 'modelset', 
    uml_type: str = 'genmymodel', 
    language_csv_path: str = 'categories_uml.csv'
) -> UMLDataset:
    """
    Load the modelset dataset from the specified path.
    
    This function loads UML models from a modelset dataset directory.
    It reads model files in various formats (XMI, JSON, TXT) and creates
    UMLModel objects with metadata.
    
    Args:
        dataset_path (str): Path to the modelset directory. Defaults to 'modelset'.
        uml_type (str): Type of UML models to load. Currently only supports 'genmymodel'.
        language_csv_path (str): Path to a CSV file containing language information for models.
            Defaults to 'categories_uml.csv'.
        
    Returns:
        Dataset: The loaded dataset containing UML models.
        
    Raises:
        AssertionError: If an unsupported UML type is provided.
        FileNotFoundError: If the dataset path, language CSV file, or expected database files
            do not exist.
        NotADirectoryError: If the dataset path is not a directory.
        
    Example:
        >>> dataset = load_dataset("path/to/modelset", "genmymodel")
        >>> print(f"Loaded {len(dataset.models)} UML models")
    """
    
    def connect_to_db(path):
        """ Create a connection to the SQLite database specified by the db_file path """
        conn = sqlite3.connect(path)
        return conn
    
    def extract_tags(row):
        json_data = json.loads(row['json'])
        # Extract tags and join them with '|'
        tags = json_data.get('tags', [])
        if not tags:  
            return np.nan  # Return NaN if no tags
        return tags

    assert uml_type == 'genmymodel', f"Unsupported UML type: {uml_type}. Only 'genmymodel' is supported."
    
    # Check if the language_csv_path is 
    language_csv_path = os.path.join(dataset_path, language_csv_path)
    if language_csv_path and not os.path.exists(language_csv_path):
        raise FileNotFoundError(f"Language CSV path does not exist: {language_csv_path}")
    
    # Check if the dataset path exists
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")
    # Check if the dataset path is a directory
    if not os.path.isdir(dataset_path):
        raise NotADirectoryError(f"Dataset path is not a directory: {dataset_path}")
    # Check if the dataset path contains the expected structure
    if not os.path.exists(os.path.join(dataset_path, 'datasets/dataset.genmymodel/data/genmymodel.db')):
        raise FileNotFoundError(f"Expected database file not found in dataset path: {dataset_path}")
    # Check if the dataset path contains the expected structure
    if not os.path.exists(os.path.join(dataset_path, 'datasets/dataset.genmymodel/data/analysis.db')):
        raise FileNotFoundError(f"Expected database file not found in dataset path: {dataset_path}")

    prefix = 'datasets/dataset.genmymodel/data/'
    # Paths to your databases
    uml_db_path = os.path.join(
        dataset_path,
        f'{prefix}/genmymodel.db'
    )
    
    analysis_db_path = os.path.join(
        dataset_path,
        f'{prefix}/analysis.db'
    )

    uml_conn = connect_to_db(uml_db_path)
    analysis_conn = connect_to_db(analysis_db_path)
    
    uml_query_metadata = "SELECT * FROM metadata;"
    uml_df_metadata = pd.read_sql_query(uml_query_metadata, uml_conn)
    
    uml_df_metadata['category'] = uml_df_metadata['metadata'].str.extract(r'category:\s*([^\,]+)')
    uml_df_metadata['tags'] = uml_df_metadata.apply(extract_tags, axis=1)
    
    analysis_query_models = "SELECT * FROM models;"
    analysis_df_models = pd.read_sql_query(analysis_query_models, analysis_conn)

    
    analysis_query_stats = "SELECT * FROM stats;"
    analysis_df_stats = pd.read_sql_query(analysis_query_stats, analysis_conn) ## Contains diagram_type
    
    diagram_type_map = {
        'diagram_ad': 'Activity Diagram',
        'diagram_cd': 'Class Diagram',
        'diagram_comp': 'Component Diagram',
        'diagram_sm': 'State Machine Diagram',
        'diagram_usecase': 'Use Case Diagram',
        'diagram_interaction': 'Interaction Diagram'
    }
    analysis_df_stats['type'] = analysis_df_stats['type'].map(diagram_type_map)
    diagram_type_df = analysis_df_stats.loc[analysis_df_stats['type'] != 'elements']
    
    
    order_df = analysis_df_stats[['id']].drop_duplicates().reset_index(drop=True).reset_index()
    order_df.columns = ['order', 'id']
    analysis_df_stats_pivot = analysis_df_stats.pivot(index='id', columns='type', values='count').reset_index()
    analysis_df_stats_pivot = analysis_df_stats_pivot.merge(order_df, on='id')
    analysis_df_stats_pivot = analysis_df_stats_pivot.sort_values('order').drop('order', axis=1)
    analysis_df_stats_pivot = analysis_df_stats_pivot.reset_index(drop=True)
    
    merged_df = pd.merge(analysis_df_stats_pivot, uml_df_metadata[['id', 'category', 'tags']], on='id', how='left')
    # print(merged_df.shape)
    if language_csv_path:
        language_df = pd.read_csv(language_csv_path)
        merged_df = merged_df.merge(language_df[['id', 'language']], on='id', how='left')

    
    data_prefix = 'repo-genmymodel-uml/data'
    graph_data_dir_path = os.path.join(dataset_path, 'graph', data_prefix)
    xmi_data_dir_path = os.path.join(dataset_path, 'raw-data', data_prefix)
    text_data_dir_path = os.path.join(dataset_path, 'txt', data_prefix)
    
    models = list()
    for _, i in tqdm(uml_df_metadata.iterrows(), total=uml_df_metadata.shape[0], desc="Loading UML models"):
        model_id = i['id'].split('/')[-1]
        model_name = model_id.split('.xmi')[0]
        json_data = json.load(open(os.path.join(graph_data_dir_path, model_id, f"{model_name}.json"), encoding='utf-8'))
        if json_data is None:
            print(f"Error loading JSON data for model ID: {model_id}")
            continue
        text_data = open(os.path.join(text_data_dir_path, model_id, f"{model_name}.txt"), encoding='utf-8').read()
        if text_data is None:
            print(f"Error loading text data for model ID: {model_id}")
            continue
        xmi_data = open(os.path.join(xmi_data_dir_path, model_id), encoding='utf-8').read()
        if xmi_data is None:
            print(f"Error loading XMI data for model ID: {model_id}")
            continue
        diagram_type = diagram_type_df.loc[diagram_type_df['id'] == model_id, 'type'].values.tolist()
        if not diagram_type:
            diagram_type = None
        
        model_hash = analysis_df_models.loc[analysis_df_models['id'] == i['id'], 'hash'].values[0]
        fp = analysis_df_models.loc[analysis_df_models['id'] == i['id'], 'relative_file'].values[0]
        language = language_df.loc[language_df['id'] == i['id'], 'language'].values[0]
        model = UMLModel(
            id=model_id,
            file_path=fp,
            hash=model_hash,
            model_json=json_data,
            model_xmi=xmi_data,
            model_txt=text_data,
            category=i['category'],
            tags=i['tags'] if not isinstance(i['tags'], float) else None,
            diagram_types=diagram_type,
            language=language,
        )
        
        models.append(model)
    
    dataset = UMLDataset(
        name='modelset',
        models=models
    )
           

    # Close the database connections
    uml_conn.close()
    analysis_conn.close()
    
    return dataset
