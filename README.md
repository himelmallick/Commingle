# Commingle
This is a repository to create an attention based framework for diagonal integration of large single cell omics data

## Requirements

The Python libraries required for executing the various phases of this pipeline are listed in the `environment.yml` file located in the corresponding folder. Please ensure that these dependencies are installed before running the pipeline.

## Usage

### Executing the Complete Pipeline

To execute the complete pipeline, run the script `run_commingle.sh` with the required arguments appended to the `python main.py` command for the corresponding phase. Modify line 4 of `run_commingle.sh` to `abc` or `harmony` to specify preprocessing using ABC or Harmony, respectively.

### Executing Individual Phases 
 - To execute one of the phase individually:
    ```
     $ cd <PhaseName>
     
     $ python main.py --<Argument1> <Value1> --<Argument2> <Value2> ...
    ```

 - For executing the Commingle code first time, execute the following command:
    
    ```
    $ python main.py --first_run
    ```
 - For executing the downstream code with attention, execute the following command:
    
    ```
    $ python main.py --attention
    ```
     
**Note:** 

1.) The implementation currently supports `.h5ad` file as input.  
2.) Please delete the existing `saved_data` and `tmp` folder in `Phase_2-Commingle/` before training on new dataset.  

