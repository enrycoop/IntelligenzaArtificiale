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

#include "Model.h"

class Perceptron : public Model
{
private:
  double eta;
  int n_iter;
  std::vector<double> weights;
  std::vector<int> errors;

  // Calculate net input
  double netInput(const std::vector<double>& X) const;

 public:
  Perceptron() = delete;
  Perceptron(double eta, int n_iter) : eta{ eta }, n_iter{ n_iter } {}
  
  void fit(std::vector<std::vector<double>>& X, std::vector<double>& y, bool verbose) override;

  double predict(const std::vector<double>& X) const override;
};