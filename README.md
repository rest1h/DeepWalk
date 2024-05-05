# DeepWalk Implementation in Python

This repository contains a Python implementation of the DeepWalk algorithm, including the Skip-gram model. DeepWalk is a popular algorithm for learning latent representations of vertices in a network, which can be used for various downstream tasks such as node classification, link prediction, and community detection.

## Features

- Implementation of the DeepWalk algorithm based on the [karateclub](https://github.com/benedekrozemberczki/karateclub) library by Benedek Rozemberczki
- Skip-gram model for learning node embeddings
- Random walk generator for sampling node sequences from the graph
- Training loop with early stopping mechanism
- Visualization of learned node embeddings using t-SNE dimensionality reduction

## Dependencies

- Python 3.x
- NumPy
- NetworkX
- scikit-learn
- Matplotlib

## Usage

1. Clone the repository:  
```git clone https://github.com/your-username/deepwalk-implementation.git```

2. Install the required dependencies:
```pip install numpy networkx scikit-learn matplotlib```

3. Run the `main.ipynb` notebook to train the DeepWalk model on the Zachary's Karate Club dataset and visualize the learned node embeddings.

## Code Structure

- `activation.py`: Contains the softmax activation function.
- `deepwalk.py`: Implements the DeepWalk algorithm and the random walk generator.
- `loss.py`: Defines the cross-entropy loss function.
- `main.ipynb`: Notebook demonstrating the usage of the DeepWalk implementation on the Karate Club dataset.
- `skip_gram.py`: Implements the Skip-gram model for learning node embeddings.

## Limitations

- The current implementation does not include hierarchical softmax or negative sampling, which are techniques used in the original DeepWalk paper to improve training efficiency.

## References

- Perozzi, B., Al-Rfou, R., & Skiena, S. (2014). DeepWalk: Online Learning of Social Representations. In Proceedings of the 20th ACM SIGKDD international conference on Knowledge discovery and data mining (pp. 701-710).
- [karateclub library](https://github.com/benedekrozemberczki/karateclub) by Benedek Rozemberczki

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
