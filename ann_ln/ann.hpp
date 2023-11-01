#pragma once

#include "Eigen/Dense"
#include "Eigen/src/Core/Matrix.h"
#include <cstddef>
#include <functional>
#include <tuple>
#include <vector>

#define DEBUG_SEED 42

using Eigen::MatrixXd;
using Eigen::VectorXd;

using Activation = std::function<VectorXd(const VectorXd &)>;
using dActivation = std::function<VectorXd(const VectorXd &, const VectorXd &)>;
using Cost = std::function<double(const VectorXd &, const VectorXd &)>;
using dCost = std::function<VectorXd(const VectorXd &, const VectorXd &)>;

namespace ML {
class Net {
public:
  Net(const size_t, const size_t, const double, const Cost &, const dCost &);

  void AddDense(const size_t, const Activation &, const dActivation &);

  VectorXd Eval(const VectorXd &);

  void Fit(const std::vector<VectorXd> &, const std::vector<VectorXd> &);

private:
  void MPopulateRandom(MatrixXd &);
  void Backprop(const VectorXd &);
  void UpdateWeights(const size_t);
  void UpdateBiases(const size_t);
  void ClearDerivatives();

private:
  VectorXd inputV;

  std::vector<MatrixXd> lWeigths;
  std::vector<VectorXd> lBiases;

  std::vector<MatrixXd> lDWeights;
  std::vector<VectorXd> lDBiases;

  std::vector<VectorXd> aInputs;
  std::vector<VectorXd> activations;
  std::vector<VectorXd> dActivations;

  std::vector<std::pair<Activation, dActivation>> activationsF;

  Cost costF;
  dCost dCostF;

  size_t nEpochs;
  double lRate;
};
} // namespace ML
