#include "../include/Model.h"


void Model::featureSizeCheck(const std::vector<std::vector<double>>& X, const std::vector<double>& y) const
{
  if (X.empty())
  {
    throw std::length_error("Passed an empty sample vector!");
  }

  if (X.size() != y.size())
  {
    throw std::length_error("Size of X [" + std::to_string(X.size()) +
      "] don't match with y[" + std::to_string(y.size()) + "]");
  }

  size_t featureSize = X.at(0).size();
  for (int i = 0; i < X.size(); ++i)
  {
    if (X.at(i).size() != featureSize)
    {
      throw std::length_error("Size of X [" + std::to_string(X.size() - 1) +
        "] don't match with other sample size!");
    }
  }
}