# Training Data Requirements Research

Based on research, there are several formulas and methods for determining the amount of data needed to train a model:

## Traditional Rules of Thumb

1. **10x Rule for Parameters**: A common guideline suggests having **10 times more training examples than model parameters** (degrees of freedom). However, this rule primarily works for small models and doesn't scale well to larger neural networks.

2. **10x Rule for Features**: Another guideline is having **10 times or more examples than features** in your dataset.

3. **100x Rule for Neural Networks**: For complex models, some suggest **10-100 times as many examples as model parameters**, though modern deep learning models with millions of parameters require substantially more data.

## Scaling Laws (Power Laws)

Recent research has identified **neural scaling laws** that describe how model performance scales with data:

- **Chinchilla Scaling Laws**: Suggests that **parameter count and dataset size should be scaled proportionally**. As you increase compute budget, increase both model size and training data roughly equally.

- **Performance follows power-law scalings** as a function of dataset size, model size, and compute budget.

- **Generalization error scales at ~1/√n** (where n = sample size) for ReLU neural networks, rather than the typical 1/n rate, indicating substantial data needs.

## Domain-Specific Guidelines

**Computer Vision**:
- **~1,000 images per class** for image classification with deep learning
- This can decrease significantly when using **pre-trained models**

**Time Series**:
- Should have **more observations than parameters**

**NLP/LLMs**:
- Complex tasks often require **millions of samples**
- Simple tasks (like MNIST) may need only a few thousand

## Statistical Methods

**Statistical Power Analysis**: Provides principled sample size estimation using:
- Confidence interval
- Accepted margin of error
- Standard deviation/population variance
- Minimum required effect size
- Acceptable probability of errors

## VC Dimension Theory

The **Vapnik-Chervonenkis (VC) dimension** measures model complexity. Training data requirements can be specified in terms of VC dimension, where higher complexity models need more data.

## Empirical Approach: Learning Curves

The most reliable method is **plotting learning curves**:
- Train models with increasing dataset sizes
- Plot training dataset size (x-axis) vs. model performance (y-axis)
- Observe where performance plateaus
- This reveals your specific problem's data requirements

## Key Insight

**No universal formula exists** - data requirements are problem-specific and must be discovered through empirical investigation. Factors affecting requirements include:
- Task complexity
- Model architecture
- Data quality and diversity
- Desired performance level

## Sources:
- [How Much Data is Needed to Train a (Good) Model? | DataRobot Blog](https://www.datarobot.com/blog/how-much-data-is-needed-to-train-a-good-model/)
- [How Much Data Is Required for Machine Learning? – PostIndustria](https://postindustria.com/how-much-data-is-required-for-machine-learning/)
- [How Do You Know You Have Enough Training Data? | Towards Data Science](https://towardsdatascience.com/how-do-you-know-you-have-enough-training-data-ad9b1fd679ee/)
- [How Much Data Is Required To Train ML Models in 2024?](https://www.akkio.com/post/how-much-data-is-required-to-train-ml)
- [How Much Training Data is Required for Machine Learning? - MachineLearningMastery.com](https://machinelearningmastery.com/much-training-data-required-machine-learning/)
- [Keras documentation: Estimating required sample size for model training](https://keras.io/examples/keras_recipes/sample_size_estimate/)
- [How much data is needed to train a neural network? - Milvus](https://milvus.io/ai-quick-reference/how-much-data-is-needed-to-train-a-neural-network)
- [How many samples are needed to train a deep neural network? - arXiv](https://arxiv.org/abs/2405.16696)
- [Scaling Laws | AI Safety, Ethics, and Society Textbook](https://www.aisafetybook.com/textbook/scaling-laws)
- [Neural scaling law - Wikipedia](https://en.wikipedia.org/wiki/Neural_scaling_law)
- [Scaling Laws for Neural Language Models - arXiv](https://arxiv.org/abs/2001.08361)
- [Explaining neural scaling laws - PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC11228526/)
- [Will we run out of data? Limits of LLM scaling](https://arxiv.org/html/2211.04325v2)
