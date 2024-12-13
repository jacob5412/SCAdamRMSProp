# SCAdamRMSProp

This repository provides implementations of the Adam and RMSProp optimization algorithms, incorporating sufficient conditions to ensure their convergence.

## Overview

Adaptive optimization algorithms like Adam and RMSProp are widely used in training deep neural networks due to their efficient and effective convergence properties. However, certain conditions must be met to guarantee their convergence. This repository explores these conditions and offers implementations that adhere to them.

## Repository Structure

- **analysis/**: Contains scripts and notebooks analyzing the convergence properties of the implemented algorithms.
- **models/**: Includes neural network models used for testing and evaluation.
- **optimizers/**: Custom implementations of Adam and RMSProp with convergence guarantees.
- **results/**: Stores output data and results from experiments.
- **tests/**: Unit tests ensuring the correctness of the optimizer implementations.
- **utils/**: Utility functions supporting the main codebase.
- **.gitignore**: Specifies files and directories to be ignored by Git.
- **LICENSE**: The repository's licensing information.
- **requirements.txt**: Lists the Python dependencies required to run the code.

## Getting Started

To utilize the optimizers in this repository, follow these steps:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/jacob5412/SCAdamRMSProp.git
   cd SCAdamRMSProp
   ```

2. **Set up a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run tests**:
   Navigate to the `tests` directory and execute the test suite using [pytest](https://docs.pytest.org/en/stable/):
   ```bash
   pytest
   ```


5. **Run experiments**:
   Explore the `analysis` directory and execute the provided Jupyter notebooks to observe the convergence behavior of the optimizers.

## Usage

To integrate the custom optimizers into your project:

1. Import the optimizer classes from the `optimizers` module:
   ```python
   from optimizers import SCAdam, SCRMSProp
   ```

2. Initialize the optimizer with your model parameters:
   ```python
   optimizer = SCAdam(model.parameters(), lr=0.001)
   ```

Refer to the example notebooks in the `analysis` directory for detailed usage scenarios.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request with your enhancements. Ensure that your code adheres to the existing style and passes all tests.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

Special thanks to the authors of "[A Sufficient Condition for Convergences of Adam and RMSProp](https://openaccess.thecvf.com/content_CVPR_2019/papers/Zou_A_Sufficient_Condition_for_Convergences_of_Adam_and_RMSProp_CVPR_2019_paper.pdf)" for their foundational work that inspired this repository.
