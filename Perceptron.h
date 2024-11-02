#pragma once
/** Perceptron classifier
*
* Parameters
* -------------
* @param eta Learning rate (between 0.0 and 1.0)
* @param n_iter Passes over the training dataset.
*
* Attributes
* -------------
* @param weights weights after fitting
* @param errors number of misclassifications in every epoch
*/
#include <vector>
#include <string>
#include <iostream>

class Perceptron
{
private:
  double eta;
  int n_iter;
  std::vector<double> weights;
  std::vector<int> errors;

  // Calculate net input
  double netInput(const std::vector<double>& X) const;

  // checks sample vector's size
  void featureSizeCheck(const std::vector<std::vector<double>>& X, const std::vector<double>& y) const;

public:
  Perceptron() = delete;
  Perceptron(double eta, int n_iter) : eta{ eta }, n_iter{ n_iter } {}

  /** Fit training data.
  * Parameters
  * ------------
  * @param X, shape=[n_samples, n_features]
              Training vectors, where n_samples
              is the number of samples and
              n_features is the number of features.
  * @param y, shape = [n_samples]
              Target values.
  */
  void fit(std::vector<std::vector<double>>& X, std::vector<double>& y, bool verbose);

  // Return class label after unit step
  double predict(const std::vector<double>& X) const;
};