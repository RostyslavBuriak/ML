#pragma once
#include "node.hpp"

namespace ML {
class Net;

class Layer {
public:
  Layer(const size_t, const size_t, const std::shared_ptr<Layer>,
        const std::function<double(double)> &,
        const std::function<double(double)> &);

  std::shared_ptr<Node> GetNode(const size_t) const;

  void CalculateNodes();

private:
  std::vector<std::shared_ptr<Node>> nodes;
  std::vector<std::shared_ptr<Weight>> weights;

  std::shared_ptr<Layer> pLayer;

  const size_t seqN;

  friend class Net;
};
} // namespace ML
