### Data generation for trajectories with more long-time correlations

```python
# Generation
python generate_data_lc.py --l 500 --N 1000000 --c 1

# Pre-Processing for training
python pre_process_data_lc.py --l 500 --c 1
```
*l* is the length, *N* is the number of trajectories. 

*c* is the type of long-time correlation:

> 1: Exponential decay (exp)
> 
> 2: Mittag-Leffler decay (M-L)
> 
> 3: Multi-exponential decay (multi-exp)
> 
> 4: Exponentially damped cosine wave (exp-cos)
