import hashlib
import json
from collections import deque
from functools import partial
from typing import List

from mcp4cm.bpmn.json_model import Shape
from mcp4cm.uml.filtering_patterns import empty_name_pattern


def compute_hash_of_modeldict(modeldict: dict) -> str:
    return hashlib.sha256(json.dumps(modeldict).encode(encoding='utf-8', errors='strict')).hexdigest()




def extract_names_from_models(dataset: 'BPMNDataset',
                              use_types: bool = False,
                              empty_name_pattern: str = "empty name") -> None:


    column = 'names'
    if use_types:
        column = 'names_with_types'

    name_extraction = partial(_extract_names_from_shape, use_types=use_types, empty_name_pattern=empty_name_pattern)

    dataset.models[column] = dataset.models['model_json'].apply(name_extraction)

    print(f"Extracting {column} from raw model done.")


def _extract_names_from_shape(model_json_string: str,
                              use_types: bool = False,
                              empty_name_pattern: str = "empty name") -> list[str]:
    # TODO: Continue to with Dict/List type instead of str
    bpmn_model_shape = Shape(**json.loads(model_json_string))
    names = list()
    names_with_types = list()

    stack = deque([bpmn_model_shape])

    while len(stack) > 0:
        element = stack.pop()
        for child in element.childShapes:
            stack.append(child)

        # TODO: add proper check for None, Emtpy and whitespace
        name = empty_name_pattern if element.properties is None or element.properties.name is None else element.properties.name

        if use_types:
            if element.stencil is not None:
                if element.stencil.id is not None:
                    names_with_types.append(f"{element.stencil.id}: {name}")
                    continue

            names_with_types.append(f"unknown type: {name}")
        else:
            names.append(name)


    if use_types:
        return names_with_types
    else: return names


def extract_names_from_model(
    model: 'BPMNModel',
    use_types: bool = False) -> 'BPMNModel':
    """
    TODO: Write docstring
    This method was adapted from the sapsam BpmnModelParser _get_elements_flat method.


    Returns the model with model.names or names_with_types set (depending on use_types).
    """


    model_dict = model.model_json
    names = list()
    names_with_types = list()

    stack = deque([model_dict])

    while len(stack) > 0:
        element = stack.pop()

        for c in element.get("childShapes", []):
            c["parent"] = element["resourceId"]
            stack.append(c)

        # don't parse the root element.
        if element["resourceId"] == model_dict["resourceId"]:
            continue

        name = element["properties"].get("name")
        if use_types:
            element_type = element["stencil"].get("id") if "stencil" in element else None
            names_with_types.append(f"{element_type}: {name}")
        else:
            names.append(name)

    if use_types:
        model.names_with_types = names_with_types
    else:
        model.names = names

    return model



