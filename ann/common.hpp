#pragma once
#include "layer.hpp"

namespace ML {

static double ReLU(const double val) { return val > 0.0 ? val : 0.0; }
static double dReLU(const double val) { return val > 0.0 ? 1.0 : 0.0; }

static double Empty(const double val) { return val; }
static double dEmpty(const double val) { return 1.0; }

static double Sigmoid(const double val) { return 1.0 / (1.0 + std::exp(-val)); }
static double dSigmoid(const double val) {
  return Sigmoid(val) * (1.0 - Sigmoid(val));
}

static std::vector<double> SoftMax(const std::vector<double> &values) {
  auto rValues = values;
  static auto softsum = [](double sum, double val) {
    return sum + std::exp(val);
  };
  auto summed = std::accumulate(values.begin(), values.end(), 0.0, softsum);
  for (auto &x : rValues) {
    x = std::exp(x) / summed;
  }
  return rValues;
}

static std::vector<double> dSoftMax(const std::vector<double> &values,
                                    const size_t id) {
  auto softMax = SoftMax(values);
  auto main = softMax[id];
  for (auto i = 0; i < softMax.size(); ++i) {
    if (i == id) {
      softMax[i] = main * (1.0 - main);
    } else {
      softMax[i] = -softMax[i] * main;
    }
  }
  return softMax;
}

static double MSE(const std::vector<std::shared_ptr<Node>> &nodes,
                  const double target) {
  double error = 0.0;

  for (const auto &node : nodes) {
    error += std::pow(target - node->GetOut(), 2);
  }

  return error / nodes.size();
}

static double dMSE(const std::vector<std::shared_ptr<Node>> &nodes,
                   const double target, const size_t) {
  double dError = 0.0;

  for (const auto &node : nodes) {
    dError += target - node->GetOut();
  }

  return -2 * dError / nodes.size();
}

static std::vector<double> softMax;
static std::vector<double> dsoftMax;

static double CE(const std::vector<std::shared_ptr<Node>> &nodes,
                 const double target) {
  std::vector<double> values;
  values.reserve(nodes.size());

  for (auto &n : nodes) {
    values.emplace_back(n->GetOut());
  }
  softMax = SoftMax(values);
  dsoftMax = dSoftMax(values, static_cast<size_t>(target));

  return -std::log(softMax[target]);
}

static double dCE(const std::vector<std::shared_ptr<Node>> &nodes,
                  const double target, const size_t nodeId) {
  std::vector<double> values;
  values.reserve(nodes.size());

  for (auto &n : nodes) {
    values.emplace_back(n->GetOut());
  }

  return (-1.0 / softMax[static_cast<size_t>(target)]) * (dsoftMax[nodeId]);
}

} // namespace ML
