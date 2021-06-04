# DFVAE
 Federated Learning for potential fairness and accuracy
## Running Details of Code
### Simulation Configuration
We provide configuration instructions in the python terminal to help our readers run DFVAE demo quickly.
After pulling our code to local path, readers can start the project by:  
```python
python mainfederated.py [others]
```
The [others] are:
```python
'--dataset', required = True, help = 'celebA | GENKI | UTKface | DSprites'
'--mode',required = True, help = 'Random | Up | Down'
'--samrounds',required = True, help = 'Up to 500'
```

The first is four legal datasets. The second represents a scale changing mode, whether we want our contribution proportion to change in a relatively fixed direction or not. The third is our federated framework duration length. You can type 'help' in the terminal to fetch further information.
### Before Running the Code
Before running our project, there's still some modules our readers can adjust by themselves.  
(1) Datasets. Readers must download at least one dataset of the four celebA, GENKI, UTKface and dSprites. There are some examples with incomplete annotations in GENKI and UTKface that we skip when making dataloaders.  
(2) The pretrain module. The module is default noted because pretraining models may performs too well to reappear in federated scenarios.   
(3) VAE network structure. The module is default set as fully connected. Readers can also switch to a CNN version that does not interfere with the generating results too much. 
(4) Plotting. Our code provides no plotting and visual modules due to a potential conflict of visual packages. Readers can add this module by their own for recurrent results in this paper.  
