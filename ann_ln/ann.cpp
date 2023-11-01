#include "ann.hpp"
#include "Eigen/src/Core/Matrix.h"
#include <cassert>
#include <iostream>
#include <random>
#include <utility>
#include <vector>

using namespace ML;

Net::Net(const size_t nInput, const size_t _nEpochs, const double _lRate,
         const Cost &_costF, const dCost &_dCostF)
    : inputV(nInput), costF(_costF), dCostF(_dCostF), nEpochs(_nEpochs),
      lRate(_lRate) {}

void Net::MPopulateRandom(MatrixXd &m) {
  static std::mt19937 gen(DEBUG_SEED);
  static std::uniform_real_distribution<double> dis(0., 1.);
  for (auto i = 0; i < m.rows(); ++i) {
    for (auto j = 0; j < m.cols(); ++j) {
      m(i, j) = dis(gen);
    }
  }
}

void Net::AddDense(const size_t nNodes, const Activation &activation,
                   const dActivation &dActivation) {
  auto m = inputV.size();
  if (!lWeigths.empty()) {
    m = lWeigths[lWeigths.size() - 1].cols();
  }

  auto weights = MatrixXd(m, nNodes);
  auto biases = VectorXd(nNodes);

  weights.setRandom();
  biases.setRandom();

  lWeigths.emplace_back(std::move(weights));
  lDWeights.emplace_back(m, nNodes);

  lBiases.emplace_back(std::move(biases));
  lDBiases.emplace_back(nNodes);

  activations.emplace_back(nNodes);
  dActivations.emplace_back(nNodes);

  activationsF.emplace_back(activation, dActivation);

  aInputs.emplace_back(nNodes);
}

VectorXd Net::Eval(const VectorXd &inputs) {
  assert(!lWeigths.empty());
  assert(inputs.size() == inputV.size());

  inputV = inputs;

  aInputs[0] = inputV.transpose() * lWeigths[0] + lBiases[0].transpose();
  activations[0] = activationsF[0].first(aInputs[0]);

  auto n = lWeigths.size();
  for (auto i = 1; i < n; ++i) {
    aInputs[i] =
        activations[i - 1].transpose() * lWeigths[i] + lBiases[i].transpose();
    activations[i] = activationsF[i].first(aInputs[i]);
  }

  return activations[activations.size() - 1];
}

void Net::Backprop(const VectorXd &targets) {
  auto n = lWeigths.size() - 1;

  dActivations[n] = dCostF(activations[n], targets)
                        .cwiseProduct(activationsF[n].second(
                            aInputs[n],
                            targets)); // Calculate derivative of cost with
                                       // respect to each input of activation
  for (auto i = 0; i < dActivations[n].size(); ++i) {
    lDWeights[n].col(i) +=
        (dActivations[n](i) * activations[n - 1]).transpose();
  }

  lDBiases[n] += dActivations[n];

  --n;

  for (; n > 0; --n) {
    VectorXd derActToIn = activationsF[n].second(aInputs[n], targets);

    for (auto i = 0; i < activations[n].size(); ++i) {
      VectorXd derToAct =
          dActivations[n + 1].cwiseProduct(lWeigths[n + 1].row(i).transpose());

      for (auto j = 0; j < derToAct.size(); ++j) {
        lDWeights[n].col(i) += derToAct(j) * derActToIn(i) * activations[n - 1];
      }

      lDBiases[n](i) += (derToAct * derActToIn(i)).sum();

      dActivations[n](i) = (derToAct * derActToIn(i)).sum();
    }
  }

  VectorXd derActToIn = activationsF[n].second(aInputs[n], targets);

  for (auto i = 0; i < activations[n].size(); ++i) {
    VectorXd derToAct =
        dActivations[n + 1].cwiseProduct(lWeigths[n + 1].row(i).transpose());

    for (auto j = 0; j < derToAct.size(); ++j) {
      lDWeights[n].col(i) += derToAct(j) * derActToIn(i) * inputV;
    }

    lDBiases[n](i) += (derToAct * derActToIn(i)).sum();
  }
}

void Net::Fit(const std::vector<VectorXd> &inputs,
              const std::vector<VectorXd> &targets) {
  auto n = inputs.size();
  double cost = 0.0;
  for (auto it = 0; it < nEpochs; ++it) {
    for (auto i = 0; i < n; ++i) {
      Eval(inputs[i]);
      Backprop(targets[i]);
      cost += costF(activations[activations.size() - 1], targets[i]);
    }
    UpdateWeights(n);
    UpdateBiases(n);
    ClearDerivatives();
    std::cout << "Iteration/Cost is:\t" << it << "/" << cost / n << std::endl;
    cost = 0;
  }
}

void Net::UpdateWeights(const size_t n) {
  for (auto i = 0; i < lDWeights.size(); ++i) {
    lWeigths[i] -= lRate * lDWeights[i] / static_cast<double>(n);
  }
}

void Net::UpdateBiases(const size_t n) {
  for (auto i = 0; i < lDBiases.size(); ++i) {
    lBiases[i] -= lRate * lDBiases[i] / static_cast<double>(n);
  }
}

void Net::ClearDerivatives() {
  for (auto i = 0; i < lDWeights.size(); ++i) {
    lDWeights[i].setZero();
    lDBiases[i].setZero();
    dActivations[i].setZero();
  }
}
