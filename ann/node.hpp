#pragma once

#include <cmath>
#include <functional>
#include <memory>
#include <numeric>
#include <type_traits>
#include <vector>

namespace ML {

class Layer;
class Net;

struct Weight {
  double val;
  size_t fromNode;
  size_t toNode;
};

class Node {
public:
  Node(const size_t, const std::function<double(double)> &activation,
       const std::function<double(double)> &dActivaion);

  std::shared_ptr<Weight> AddWeight(const double, const size_t);

  void Calculate(const std::shared_ptr<Layer>);
  void SetBias(const double);
  void UpdateWeights(const size_t, const double);
  void UpdateBias(const size_t, const double);

  double GetIn() const;
  double GetOut() const;
  double GetBias() const;

private:
  std::vector<std::shared_ptr<Weight>> weights;
  std::vector<std::shared_ptr<Weight>> dWeights;

  std::function<double(double)> activation;
  std::function<double(double)> dActivation;

  double in{};
  double out{};
  double bias{};
  double dBias{};

  const size_t n;

  friend class Net;
};

} // namespace ML
