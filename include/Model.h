#pragma once

#include <vector>
#include <string>
#include <iostream>

class Model
{
protected:
  // checks sample vector's size
  void featureSizeCheck(const std::vector<std::vector<double>>& X, const std::vector<double>& y) const;

public:
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
  virtual void fit(std::vector<std::vector<double>>& X, std::vector<double>& y, bool verbose) =0;

  // Return class label after unit step
  virtual double predict(const std::vector<double>& X) const =0;
};
