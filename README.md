# Dine
Personal contributions to a collaborative budget search application:

- `getCombos.py` performs an iterative search through possible food item combinations given user-defined preferences and budget constraints
- `modelTrain.py` preprocesses text data from a CSV file for classification, applies dimensionality reduction with SVD, trains a logistic regression model to predict labels, selects an optimal threshold to minimize false positive rates, logs the model's performance metrics to a file, and exports model weights
- `computeError.py` is a helper function that computes a classifier's error over a random split of the data, averaged over ntrial runs
- `parser.py` loads menu data from a CSV file, utilizes an off-the-shelf language model to determine if prices precede food items, and extracts the corresponding price-item pairs, logging the results and related metrics to a text file
