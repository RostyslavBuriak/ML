#include "layer.hpp"
#include <cassert>
#include <random>

using namespace ML;

Layer::Layer(const size_t n, const size_t sN, const std::shared_ptr<Layer> l,
             const std::function<double(double)> &activation,
             const std::function<double(double)> &dActivation)
    : seqN(sN), pLayer(l) {

  nodes.reserve(n);

  std::mt19937 gen(42);
  std::uniform_real_distribution<double> dis(
      0.3, 0.7); // decided to use 0.3 - 0.7 range since some extreme values
                 // could potentially affect weights

  for (auto i = 0; i < n; ++i) {
    auto node = std::make_shared<Node>(i, activation, dActivation);

    if (pLayer) {
      for (auto j = 0; j < pLayer->nodes.size(); ++j) {
        weights.emplace_back(node->AddWeight(dis(gen), j));
      }
      node->SetBias(dis(gen));
    }

    nodes.emplace_back(node);
  }
}

std::shared_ptr<Node> Layer::GetNode(const size_t id) const {
  assert(id < nodes.size());
  return nodes[id];
}

void Layer::CalculateNodes() {
  for (const auto &node : nodes) {
    node->Calculate(pLayer);
  }
}
