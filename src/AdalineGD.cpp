#include "../include/AdalineGD.h"

void AdalineGD::fit(std::vector<std::vector<double>>& X, std::vector<double>& y, bool verbose)
{

  featureSizeCheck(X, y);

  this->weights = std::vector<double>(X.at(0).size() + 1);
  this->cost = {};

  for (int epoch = 1; epoch <= n_iter; ++epoch) // epoche
  {
    if (verbose)
      std::cout << "Epoch: " << epoch;

    double output = this->netInput(X)
    // TODO 
    this->cost.push_back(errors);
  }
}

double AdalineGD::predict(const std::vector<double>& X) const
{
  return activation(X) >= 0.0 ? 1.0 : -1.0;
}

double AdalineGD::activation(const std::vector<double>& X) const
{
  return this->netInput(X);
}

double AdalineGD::netInput(const std::vector<double>& X) const
{
  double result = 0;
  if (this->weights.size() - 1 != X.size())
  {
    throw std::length_error("Size of w [" + std::to_string(this->weights.size() - 1) +
      "] don't match with X[" + std::to_string(X.size()) + "]");
  }

  for (int i = 0; i < X.size(); ++i)
  {
    result += (X.at(i) * this->weights.at(i + 1));
  }
  return result + this->weights.at(0);
}
