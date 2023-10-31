#pragma once
#include "layer.hpp"

namespace ML {

class Net {
public:
  Net(const size_t, const double,
      const std::function<double(const std::vector<std::shared_ptr<Node>> &,
                                 const double)> &,
      const std::function<double(const std::vector<std::shared_ptr<Node>> &,
                                 const double, const size_t)> &);

  void AddDense(const size_t, const std::function<double(double)> &,
                const std::function<double(double)> &);
  void AddInput(const std::vector<double> &);
  void AddInputs(const std::vector<std::vector<double>> &,
                 const std::vector<double> &);

  void CalculateError(const double);

  void Forward(const size_t);
  void Backprop(const size_t);
  void Learn();
  void CalcDerRec(const std::shared_ptr<Node> &, const std::shared_ptr<Layer> &,
                  const double);

  std::vector<double> Predict(const std::vector<double> &);

  double GetCost() const;

private:
  std::function<double(const std::vector<std::shared_ptr<Node>> &,
                       const double)>
      costF;

  std::function<double(const std::vector<std::shared_ptr<Node>> &, const double,
                       const size_t)>
      dCostF;

  std::vector<std::shared_ptr<Layer>> layers;
  std::vector<std::vector<double>> inputs;
  std::vector<double> targets;

  double error{};
  double lRate;

  size_t nEp;
};

} // namespace ML
