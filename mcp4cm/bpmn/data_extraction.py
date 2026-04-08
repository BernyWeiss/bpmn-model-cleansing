from collections import deque
from functools import partial
from typing import List, Dict

from pydantic.v1.utils import get_model

from mcp4cm.bpmn.json_model import Shape
from mcp4cm.bpmn.dataloading import BPMNDataset
from mcp4cm.generic.language_detection import get_model_language
from mcp4cm.generic.utils import join_texts, get_text_language

translation_table = str.maketrans({'\n': ' '})


def extract_names_from_models(dataset: BPMNDataset,
                              use_types: bool = False,
                              empty_name_pattern: str = "empty name") -> None:
    column = 'names'
    if use_types:
        column = 'names_with_types'

    name_extraction = partial(_extract_names_from_shape, use_types=use_types, empty_name_pattern=empty_name_pattern)

    dataset.models[column] = dataset.models['model_json'].apply(name_extraction)

    print(f"Extracting {column} from raw model done.")


def _extract_names_from_shape(model_json: List | Dict,
                              use_types: bool = False,
                              empty_name_pattern: str = "empty name") -> list[str]:
    bpmn_model_shape = Shape.model_validate(model_json)
    names = list()
    names_with_types = list()

    stack = deque([bpmn_model_shape])

    if use_types:
        while len(stack) > 0:
            element = stack.pop()
            for child in element.childShapes:
                stack.append(child)
            if element.properties:
                name = None
                if element.properties.name:
                    name = element.properties.name.strip()
                    name = name.translate(translation_table)
                if not name:
                    name = empty_name_pattern

            if element.stencil:
                if element.stencil.id:
                    names_with_types.append(f"{element.stencil.id}: {name}")
            else:
                names_with_types.append(f"unknown type: {name}")
        return names_with_types

    else:
        while len(stack) > 0:
            element = stack.pop()
            for child in element.childShapes:
                stack.append(child)

            if element.properties:
                name = None
                if element.properties.name:
                    name = element.properties.name.strip()
                    name = name.translate(translation_table)
                if not name:
                    name = empty_name_pattern
                names.append(name)
            else:
                names.append(empty_name_pattern)
        return names


def filter_empty_models(dataset: BPMNDataset, key: str = 'names', inplace: bool = False,
                        empty_name: str = "empty name") -> BPMNDataset:
    empty_models = dataset.models[key].apply(lambda names: all(name == empty_name for name in names))
    non_empty_models = ~empty_models
    print(f'Found {sum(empty_models)} models with empty names.')
    if inplace:
        dataset.models = dataset.models[non_empty_models]
        return dataset
    return BPMNDataset(name=dataset.name, models=dataset.models[non_empty_models])


def extract_model_languages(dataset: BPMNDataset, key: str = 'names', empty_name: str = "empty name"):
    dataset.models['language'] = dataset.models[key].apply(
        lambda text: get_text_language(join_texts(text, empty_name=empty_name)))
