from collections import deque, Counter
from functools import partial
from typing import List, Dict


from mcp4cm.bpmn.json_model import Shape
from mcp4cm.bpmn.dataloading import BPMNDataset

from mcp4cm.bpmn.data_extraction import translation_table


# TODO: Remove this temporary file which was used for analysis of the models.

def extract_names_from_models(dataset: BPMNDataset,
                              use_types: bool = False,
                              empty_name_pattern: str = "empty name") -> None:
    column = 'names'
    if use_types:
        column = 'names_with_types'
    empty_counter = Counter()
    set_counter = Counter()
    text_counter = Counter()
    docu_counter = Counter()
    name_extraction = partial(_extract_names_from_shape_with_counts, set_counter=set_counter, empty_counter=empty_counter, text_counter=text_counter, docu_counter=docu_counter, use_types=use_types, empty_name_pattern=empty_name_pattern)
    dataset.models[column] = dataset.models['model_json'].apply(name_extraction)

    print("Text counter")
    print(text_counter)
    print("Documentation counter")
    print(docu_counter)
    print(f"Extracting {column} from raw model done.")
    print("Empty Counter")
    print(empty_counter)
    print("Set Counter")
    print(set_counter)
    print("Net names")
    set_counter.subtract(empty_counter)
    print(set_counter)


def _extract_names_from_shape_with_counts(model_json: List | Dict,
                                          empty_counter: Counter,
                                          set_counter: Counter,
                                          text_counter: Counter,
                                          docu_counter: Counter,
                                          use_types: bool = False,
                                          empty_name_pattern: str = "empty name",
                                          empty_type_pattern: str = "unknown type") -> list[str]:
    bpmn_model_shape = Shape.model_validate(model_json)
    names = list()
    names_with_types = list()

    stack = deque([bpmn_model_shape])

    if use_types:
        while len(stack) > 0:
            element = stack.pop()
            for child in element.childShapes:
                stack.append(child)
            name = None
            if element.stencil and element.stencil.id:
                node_type = element.stencil.id
            else:
                node_type = empty_type_pattern
            if element.properties:
                if element.properties.name:
                    name = element.properties.name.strip()
                    name = name.translate(translation_table)
                if not name:
                    name = empty_name_pattern
                    empty_counter[node_type] += 1
                else:
                    set_counter[node_type] += 1

                if element.properties.documentation:
                    docu = element.properties.documentation.strip()
                    docu = docu.translate(translation_table)
                    if docu:
                        docu_counter[node_type] += 1

                if element.properties.text:
                    text = element.properties.text.strip()
                    text = text.translate(translation_table)
                    if text:
                        print(text)
                        text_counter[node_type] += 1


            else:
                empty_counter[node_type] +=1
                name = empty_name_pattern

            names_with_types.append(f"{node_type}: {name}")
        return names_with_types

    else:
        while len(stack) > 0:
            element = stack.pop()
            for child in element.childShapes:
                stack.append(child)
            node_type = (element.stencil and element.stencil.id) or empty_type_pattern
            if element.properties:
                name = None
                if element.properties.name:
                    name = element.properties.name.strip()
                    name = name.translate(translation_table)
                if not name:
                    name = empty_name_pattern
                    empty_counter[node_type] += 1
                else:
                    set_counter[node_type] += 1
                names.append(name)

                if element.properties.documentation:
                    docu = element.properties.documentation.strip()
                    docu = docu.translate(translation_table)
                    if docu:
                        docu_counter[node_type] += 1

                if element.properties.text:
                    text = element.properties.text.strip()
                    text = text.translate(translation_table)
                    if text:
                        text_counter[node_type] += 1
                        # if name == empty_name_pattern:
                        #     docu_counter[node_type] += 1

            else:
                empty_counter[node_type] +=1
                names.append(empty_name_pattern)
        return names