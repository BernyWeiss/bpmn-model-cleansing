# Artifact Badge Application Status

Badges Requested:
- Artifact Reusable

Artifact Description:
Part 1: Model Cleansing Pipeline for Conceptual Models  
  • Implemented as a Python library  
  • Publicly archived on Zenodo (DOI: [10.5281/zenodo.16365384](https://doi.org/10.5281/zenodo.16365384))  
  • Packaged and published on PyPI (pip install mcp4cm)

Part 2: Comparative Reproducibility Package  
  • Contains all code and data needed to reproduce evaluation  
  • Publicly archived on Zenodo ([10.5281/zenodo.16285770] (https://doi.org/10.5281/zenodo.16285770))

Reasons for Badges:
1. Artifact Resusable  
   • The Python library is fully functional and can be installed via pip.  
   • Comprehensive documentation available in README and docstrings.  
   • Example notebooks demonstrate usage and functionality.  
   • All code, data, and documentation are included or linked (Zenodo DOI, PyPI link).  
   • No proprietary dependencies; all software is open-source under MIT license.  

2. Artifact Reusable  
   • Two reproducibility studies included:  
     - Explainable AI for Model Comprehension  
     - MM Classification  
   • Each study has its own directory with all necessary code and data.  
   • Clear instructions for running each study provided in README files.  
   • All dependencies are documented and can be installed via pip.
   • Clear, step-by-step setup and usage instructions in README.  
   • Includes example notebooks and test suite verifying end-to-end functionality.  


Rationale for badges

We apply for the following artifact evaluation badges:
1. **Available**:
    * The artifacts are available on Zenodo with a DOI: [Python Library](https://zenodo.org/records/16365384) and [Reproducibility Package](https://zenodo.org/records/16285770). This is also specified in the README.md file of the repository.
    * A license file is provided in the repository, as described in the LICENSE.md file of the Python Library.
    * Author information for this artifact is provided in the README.md file of the repository.
2. **Functional**:
    * The artifact can be run with the provided instructions in the README.md file of the repository.
    * All required dependencies for the artifact and uses cases are specified in the `requirements.txt` file for each use case.
    * The instructions to install setting up the environment, installing dependencies, and running the use cases are provided in the README.md file for the python library and both the reproducibility studies.
    * Instructions for running the notebooks to produce the results in the paper are provided in the README.md file for both reproducibility studies.
3. **Reusable**:
    * The README.md files contain an organization section that describes the structure of the code and data in the repository.
    * There are two artifacts and they are organized in a way that allows for easy reuse of the code and data. 
    * The reproducibility package is structured in a way that allows execution of both the reproducibility studies independently.
    * The code is organized into directories for each use case, with a clear structure that allows for easy navigation and understanding of the code.
    * A readme.md file for both reproducibility studies provides description about how to run the code and how to re-use the python library developed for model cleansing using code snippets in the data_generation.ipynb and test_mcp4cm.ipynb.
    * Usages for the python library is clearly provided in the python library test_mcp4cm.ipynb as well as in data_generation.ipynb.