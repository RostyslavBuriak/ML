#include "ann.hpp"
#include <cassert>
#include <iostream>
#include <memory>
#include <random>
#include <vector>

using namespace ML;

Net::Net(const size_t nEp, const double lRate,
         const std::function<double(const std::vector<std::shared_ptr<Node>> &,
                                    const double)> &cost,
         const std::function<double(const std::vector<std::shared_ptr<Node>> &,
                                    const double, const size_t)> &dCostF)
    : nEp(nEp), lRate(lRate), costF(cost), dCostF(dCostF) {}

void Net::AddDense(const size_t n,
                   const std::function<double(double)> &activation,
                   const std::function<double(double)> &dActivation) {
  auto size = layers.size();

  auto layer = std::make_shared<Layer>(
      n, size, size ? layers[size - 1] : nullptr, activation, dActivation);

  layers.emplace_back(layer);
}

void Net::AddInput(const std::vector<double> &input) {
  assert(!layers.empty());
  assert(input.size() == layers[0]->nodes.size());

  const auto &zLayer = layers[0];

  for (auto i = 0; i < input.size(); ++i) {
    zLayer->nodes[i]->out = input[i];
  }
}

void Net::AddInputs(const std::vector<std::vector<double>> &_inputs,
                    const std::vector<double> &_targets) {
  assert(inputs.size() == targets.size());

  inputs = _inputs;
  targets = _targets;
}

void Net::CalculateError(const double target) {
  auto outLayer = layers[layers.size() - 1];

  error += costF(outLayer->nodes, target);
}

void Net::Forward(const size_t i) {

  assert(!layers.empty());
  assert(!inputs.empty());

  for (auto j = 0; j < layers[0]->nodes.size(); ++j) {
    auto node = layers[0]->nodes[j];
    node->out = inputs[i][j];
  }

  for (auto j = 1; j < layers.size(); ++j) {
    layers[j]->CalculateNodes();
  }

  CalculateError(targets[i]);
}

void Net::Backprop(const size_t i) {
  auto outputL = layers[layers.size() - 1];

  for (auto k = 0; k < outputL->nodes.size(); ++k) {
    auto dCost = dCostF(outputL->nodes, targets[i], k);
    CalcDerRec(outputL->nodes[k], outputL, dCost);
  }
}

void Net::CalcDerRec(const std::shared_ptr<Node> &node,
                     const std::shared_ptr<Layer> &layer, const double der) {
  auto pLayer = layer->pLayer;
  if (!pLayer) {
    return;
  }
  auto n = node->weights.size();
  const auto tempDer = der * node->dActivation(node->in);
  node->dBias += tempDer;
  for (auto i = 0; i < n; ++i) {
    const auto weight = node->weights[i];
    node->dWeights[i]->val += tempDer * pLayer->GetNode(weight->fromNode)->out;
    CalcDerRec(pLayer->nodes[i], pLayer, tempDer * node->weights[i]->val);
  }
}

void Net::Learn() {
  const auto n = inputs.size();
  for (auto i = 0; i < nEp; ++i) {
    error = 0;
    for (auto j = 0; j < n; ++j) {
      Forward(j);
      Backprop(j);
    }

    for (auto &l : layers) {
      if (l->pLayer) {
        for (auto &node : l->nodes) {
          node->UpdateWeights(n, lRate);
          node->UpdateBias(n, lRate);
        }
      }
    }
    error /= static_cast<double>(n);
    std::cout << "iteration/error: " << i << "/" << error << std::endl;
  }
}

std::vector<double> Net::Predict(const std::vector<double> &instance) {
  AddInput(instance);

  for (auto j = 1; j < layers.size(); ++j) {
    layers[j]->CalculateNodes();
  }
  const auto &nodes = layers[layers.size() - 1]->nodes;
  std::vector<double> outputs;
  outputs.reserve(nodes.size());

  for (const auto &n : nodes) {
    outputs.emplace_back(n->GetOut());
  }

  return outputs;
}

double Net::GetCost() const { return error; }

// END NET
