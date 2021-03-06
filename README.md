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
