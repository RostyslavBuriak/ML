#include "node.hpp"
#include "layer.hpp"
#include <cassert>

using namespace ML;

Node::Node(const size_t n, const std::function<double(double)> &activation,
           const std::function<double(double)> &dActivation)
    : n(n), activation(activation), dActivation(dActivation) {}

std::shared_ptr<Weight> Node::AddWeight(const double val, const size_t from) {
  auto weight = std::make_shared<Weight>();
  weight->val = val;
  weight->fromNode = from;
  weight->toNode = n;
  weights.emplace_back(weight);
  dWeights.emplace_back(std::make_shared<Weight>());
  return weight;
}

void Node::Calculate(const std::shared_ptr<Layer> l) {
  assert(l);
  double temp = 0.0;
  for (const auto &weight : weights) {
    temp += l->GetNode(weight->fromNode)->out * weight->val;
  }
  in = temp + bias;
  out = activation(in);
}

void Node::UpdateWeights(const size_t num, const double lRate) {
  for (auto i = 0; i < weights.size(); ++i) {
    weights[i]->val -= lRate * (dWeights[i]->val / num);
    dWeights[i]->val = 0;
  }
}

void Node::UpdateBias(const size_t num, const double lRate) {
  bias -= lRate * (dBias / static_cast<double>(num));
  dBias = 0;
}

void Node::SetBias(const double b) { bias = b; }

double Node::GetIn() const { return in; }

double Node::GetOut() const { return out; }

double Node::GetBias() const { return bias; }
