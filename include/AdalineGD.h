#pragma once

#include "../include/Model.h"

class AdalineGD : public Model
{
private:
  double eta;
  int n_iter;
  std::vector<double> weights;
  std::vector<int> cost;

  double activation(const std::vector<double>& X) const;

  double netInput(const std::vector<double>& X) const;
public:
  AdalineGD() = delete;
  AdalineGD(double eta, int n_iter) : eta{ eta }, n_iter{ n_iter } {}

  void fit(std::vector<std::vector<double>>& X, std::vector<double>& y, bool verbose) override;

  double predict(const std::vector<double>& X) const override;

};