This is the repository connected to the paper "Grouped Multi-Layer Echo State Networks withSelf-Normalizing Activations". Below, we present additional information and experiments that were performed but did not fit into the paper.

- [Architecture overview](#architecture-overview)
  * [Deep ESN](#deep-esn)
  * [Grouped ESN](#grouped-esn)
  * [Grouped Deep ESN](#grouped-deep-esn)
- [Grid Search Configuration](#grid-search-configuration)
- [Best Architecture Configuration](#best-architecture-configuration)
  * [ESN](#esn)
  * [LSTM](#lstm)
- [Memory Capacity](#memory-capacity)
  * [Results by architecture type](#results-by-architecture-type)
  * [Input scaling vs Activation Radius](#input-scaling-vs-activation-radius)


## Architecture overview
Four different types of Echo State Networks were tested: shallow ESN, deep ESN, grouped ESN and the generalisation of all of them which is grouped deep Echo State Network.

### Deep ESN
Briefly, Deep Echo State Network stacks several reservoirs one on top of another. The difference to classical deep neural network is that the output of all intermediate layers is concatenated giving the final result.

<img src="https://user-images.githubusercontent.com/32894690/113476313-0c1c4680-947b-11eb-9d57-83485a946f65.png" width="400">

### Grouped ESN
Grouped ESN consist of a group of shallow ESNs whose outputs are concatenated to create final output.

<img src="https://user-images.githubusercontent.com/32894690/113476317-163e4500-947b-11eb-9eb4-9827d691c0e7.png" width="200">

### Grouped Deep ESN
Grouped Deep ESN puts both these approaches together by creating a group of Deep ESNs The output of gdESN is concatenated output of all its reservoirs.

<img src="https://user-images.githubusercontent.com/32894690/113476592-b5b00780-947c-11eb-93b0-d312c8accb41.png" width="400">


## Grid Search Configuration
Tanh activation function highly  depends  on  input  scaling,  while  for  SNA(self-normalizing activation) its  effect  is  reduced  due  to  normalization  factor.This is why we used two different hyperparameter setups as an input to grid search. For each configuration, 5 concrete models were generated with different random seeds applied for weight initialization. For *gESN* and *dESN* architectures N<sub>g</sub>,N<sub>l</sub>∈ {2,3,4,5,10}, where N<sub>g</sub> is number of groups and N<sub>l</sub> number of layers. For *gdESN* architectures each of{(2,2),(2,3),(2,4),(2,5),(3,2),(3,3),(3,4),(4,2),(4,3),(5,2)}configurations of (groups, layers) was used.Each  tested  model,  shallow  or  decoupled  had  the  total  number  of  1000  neurons  (with  the  small deviations  resulting  from  subreservoir  integer  sizes).  Grid  search  optimization  was  performed  on all these hyperparameters and architectures. The best configuration was selected based on minimal NRMSE score obtained on validation set. In the main part of experiment, which includes 1-step ahead  prediction  of  time-series,  the  average and minimal NRMSE  on  the  test  set  was calculated for each architecture and the target hyperparameter set.

<table>
    <tr>
        <td>Hyperparameter</td>
        <td>tanh</td>
        <td>self-normalizing</td>
    </tr>
    <tr>
        <td>Input Scaling s</td>
        <td>{0.1,0.5,1.0,10}</td>
        <td>1.0</td>
    </tr>
    <tr>
        <td>Spectral Radius ρ</td>
        <td>{0.7,0.8,0.9,1.0}</td>
        <td>1.0</td>
    </tr>
    <tr>
        <td>Leaking Rate α</td>
        <td>{0.7,0.8,0.9,1.0}</td>
        <td>1.0</td>
    </tr>
    <tr>
        <td>Regularization β</td>
        <td>{0.5,1,2}</td>
        <td>{0.5,1,2}</td>
    </tr>
    <tr>
        <td>Actiation Radius r</td>
        <td>-</td>
        <td>{50k | k&#8712;1,2,3...30}</td>
    </tr>
    <tr>
        <td>Washout</td>
        <td colspan="2" style="text-align: center; vertical-align: middle;">100</td>
    </tr>
    <tr>
        <td>Total neurons</td>
       <td colspan="2">1000</td>
    </tr>
    <tr>
        <td>Sparsity</td>
        <td colspan="2">10%</td>
    </tr>
    <tr>
        <td>Weight distribution</td>
        <td colspan="2">uniform, centered around 0</td>
    </tr>
</table>

## Best Architecture configuration
### ESN

Hyperparameter sets for best results in different tasks are listed below:

MackeyGlass Series one step ahead prediction:
| Architecture         | Layers | Groups | Activation Radius | Regularization |
|----------------------|--------|--------|-------------------|----------------|
| *ESN tanh*           | 1      | 1      | None              | 0.5            |
| *dESN tanh*           | 4      | 1      | None              | 0.5            |
|  *gESN tanh*              | 1      | 2      | None              | 0.5            |
|  *gdESN tanh*       | 2      | 4      | None              | 0.5            |
|----------------------|--------|--------|-------------------|----------------|
| *ESN SNA*                  | 1      | 1      | 100               | 0.5            |
| *dESN SNA*         | 3      | 1      | 100               | 0.5            |
| *gESN SNA*      | 1      | 10     | 200               | 0.5            |
| *gdESN SNA* | 3      | 3      | 50                | 0.5            |



Multiple Superimposed Oscillators one step ahead prediction:
| Architecture                 | Layers | Groups | Activation Radius | Regularization |
|------------------------------|--------|--------|-------------------|----------------|
| *ESN tanh*                    | 1      | 1      | None              | 0.5            |
|*dESN tanh*                  | 2      | 1      | None              | 0.5            |
| *gESN tanh*                  | 1      | 2      | None              | 0.5            |
| *gdESN tanh*            | 3      | 3      | None              | 0.5            |
|----------------------|--------|--------|-------------------|----------------|
| *ESN SNA*                           | 1      | 1      | 1400              | 0.5            |
| *dESN SNA*                  | 4      | 1      | 1200              | 1.0            |
| *gESN SNA*             | 1      | 3      | 1300              | 1.0            |
| 	*gdESN SNA*  | 3      | 2      | 1400              | 1.0            |

Sunspot Series one step ahead prediction:

| Architecture                    | Layers | Groups | Activation Radius | Regularization |
|---------------------------------|--------|--------|-------------------|----------------|
|*ESN tanh*                      | 1      | 1      | None              | 2.0            |
| *dESN tanh*                    | 2      | 1      | None              | 2.0            |
| *gESN tanh*                   | 1      | 20     | None              | 1.0            |
|*gdESN tanh*             | 3      | 3      | None              | 2.0            |
|----------------------|--------|--------|-------------------|----------------|
| *ESN SNA*                           | 1      | 1      | 850               | 1.0            |
|*dESN SNA*           | 3      | 1      | 1400              | 1.0            |
| *gESN SNA*                 | 1      | 8      | 400               | 2.0            |
| *gdESN SNA*   | 3      | 2      | 1450              | 1.0            |


### LSTM

Additionally, several experiments with moving average and LSTM networks were performed.

For LSTM 4 architectures were tested

Architecture 1

![m1](https://user-images.githubusercontent.com/32894690/113422089-71fdc500-93cc-11eb-8d12-3ed92ae2e03a.png)


Architecture 2

![m2](https://user-images.githubusercontent.com/32894690/113422180-8d68d000-93cc-11eb-8e64-1f9eb0c481ca.png)


Architecture 3

![m3](https://user-images.githubusercontent.com/32894690/113422194-93f74780-93cc-11eb-8276-869f084d0d43.png)


Architecture 4

![m4](https://user-images.githubusercontent.com/32894690/113422210-99549200-93cc-11eb-9a68-d2fdad84116d.png)


Each architecture was trained for 100 epochs with Adam optimiser.
Each model was trained 5 times for each learning rate of [0.001, 0.002, 0.0005,0.005].

Best results were obtained with:
* Architecture 2 and learning rate 0.002 for MackeGlass
* Architecture 3 and learning rate 0.002 for Sunspot
* Architecture 4 and learning rate 0.005 for Multiple Superimposed oscillators

## Memory Capacity
### Results by architecture type
### Input scaling vs Activation Radius











