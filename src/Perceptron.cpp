#include "../include/Perceptron.h"

void Perceptron::fit(std::vector<std::vector<double>>& X, std::vector<double>& y, bool verbose=false)
{
  featureSizeCheck(X, y);

  this->weights = std::vector<double>(X.at(0).size() + 1);
  this->errors = {};


  int errors = 0;
  for (int epoch = 1; epoch <= n_iter; ++epoch) // epoche
  {
    if (verbose)
      std::cout << "Epoch: " << epoch;

    errors = 0;
    for (int iSample = 0; iSample < X.size(); ++iSample)
    {
      // calcolo l'aggiornamento calcolando la distanza tra la classe e il valore predetto
      // per un certo coeff. eta
      double update = this->eta * (y.at(iSample) - predict(X.at(iSample)));

      // aggiorno i pesi in weights
      for (int iWeight = 1; iWeight < this->weights.size(); ++iWeight)
      {
        this->weights[iWeight] += update * X.at(iSample).at(iWeight - 1);
      }
      this->weights[0] += update;

      errors += int(update != 0.0);
    }
    if (verbose)
      std::cout << "  Errors: " << errors << std::endl;
    this->errors.push_back(errors);
  }
}

double Perceptron::netInput(const std::vector<double>& X) const
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

double Perceptron::predict(const std::vector<double>& X) const
{
  return netInput(X) >= 0.0 ? 1.0 : -1.0;
}
