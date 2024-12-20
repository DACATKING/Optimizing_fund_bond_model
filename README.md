# Bond Fund Optimization Project

## Description

This project focuses on optimizing the training speed of an original project by leveraging **PyTorch's `torch.compile`** and **mixed precision training**. The original project can be found at [LUOFENGZHOU/bond_fund_skill](https://github.com/LUOFENGZHOU/bond_fund_skill).

The optimizations include:
- **PyTorch Compile (`torch.compile`)**: Enhances performance through graph-based optimization.
- **Automatic Mixed Precision (AMP)**: Reduces memory usage and increases speed by using mixed precision (combining float32 and float16).

## Outline

The repository includes the following files:

- `main.py`: Implements the optimized version of the project using PyTorch's `torch.compile`.
- `main_0.py`: Contains the original version of the project for benchmarking.
- `main_amp.py`: Runs the AMP (Automatic Mixed Precision) version of the project.
- `settings.py`: Contains configuration and hyperparameter settings for all scripts.
- `utils.py`: Utility functions used across the scripts.
- `bond_fundlog.log`: Logs the execution details and performance metrics for different runs.

## Example Commands

To execute the different versions of the code, use the following commands:

### Run the PyTorch Compile Version
```bash
python main.py
```
This will execute the optimized version using `torch.compile` and display the runtime.

### Run the Original Version
```bash
python main_0.py
```
This will execute the original version for benchmarking and comparison.

### Run the AMP Version
```bash
python main_amp.py
```
This will execute the version utilizing Automatic Mixed Precision.

## Results

### Performance Metrics
The following table summarizes the runtime results for each version:

| Version           | Speedup vs. Original  |
|-------------------|-----------------------|
| Original          | 1.0x                  |
| PyTorch Compile   | 1.14x                 |
| AMP               | 0.33x                 |

> **Note**: Replace `X.XX` with the actual runtime values from your experiments.

## Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA Toolkit (if running amp)

## References
- [LUOFENGZHOU/bond_fund_skill](https://github.com/LUOFENGZHOU/bond_fund_skill)
- [PyTorch Documentation](https://pytorch.org/docs/)

---
Feel free to submit issues or contribute to further optimizations!
