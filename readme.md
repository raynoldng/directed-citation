# Citation Networks as Directed Graphs

Citation network datasets commonly used in node classification benchmark but with directed edges (u -> y) if paper u cites paper v. 

Example of using this dataset with PyTorch Geometric is in `dataset.py` 

`process_data.py` preprocesses data in the `raw_data`, taken from https://linqs.soe.ucsc.edu/data, and outputs them in `data`