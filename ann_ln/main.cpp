#include "ann.hpp"
#include <cmath>
#include <fstream>
#include <iostream>
#include <ostream>
#include <random>
#include <vector>

VectorXd Sigmoid(const VectorXd &v) {
  VectorXd rV(v.size());
  for (auto i = 0; i < v.size(); ++i) {
    rV(i) = 1. / (1. + std::exp(-v(i)));
  }
  return rV;
}

VectorXd dSigmoid(const VectorXd &v, const VectorXd &targets) {
  VectorXd sigmoid_v = Sigmoid(v);
  VectorXd rV(v.size());
  for (auto i = 0; i < v.size(); ++i) {
    rV(i) = sigmoid_v(i) * (1. - sigmoid_v(i));
  }
  return rV;
}

double MSE(const VectorXd &outputs, const VectorXd &targets) {
  return std::pow(targets(0) - outputs(0), 2);
}

VectorXd dMSE(const VectorXd &outputs, const VectorXd &targets) {
  VectorXd rV(1);
  rV << (-2 * (targets(0) - outputs(0)));
  return rV;
}

VectorXd SoftMax(const VectorXd &v) {
  VectorXd rV(v.size());
  double summed = 0;
  for (auto i = 0; i < v.size(); ++i) {
    summed += std::exp(v(i));
  }
  for (auto i = 0; i < v.size(); ++i) {
    rV(i) = std::exp(v(i)) / summed;
  }
  return rV;
}

VectorXd dSoftMax(const VectorXd &v, const VectorXd &targets) {
  VectorXd softmax = SoftMax(v);
  size_t mainId = targets(0);
  double main = softmax(mainId);
  for (auto i = 0; i < softmax.size(); ++i) {
    if (i == mainId) {
      softmax(i) = main * (1.0 - main);
    } else {
      softmax(i) = -softmax(i) * main;
    }
  }
  return softmax;
}

VectorXd ReLU(const VectorXd &v) {
  static auto lrelu = [](const double val) { return val > 0.0 ? val : 0.0; };

  VectorXd rV = v;
  rV = rV.unaryExpr(lrelu);

  return rV;
}

VectorXd dReLU(const VectorXd &v, const VectorXd &targets) {
  static auto ldrelu = [](const double val) { return val > 0.0 ? 1.0 : 0.0; };

  VectorXd rV = v;
  rV = rV.unaryExpr(ldrelu);

  return rV;
}

double CE(const VectorXd &v, const VectorXd &targets) {
  VectorXd softmax = SoftMax(v);
  double target = softmax((size_t)targets(0));
  return -std::log(target);
}

VectorXd dCE(const VectorXd &v, const VectorXd &targets) {
  VectorXd softmax = SoftMax(v);

  VectorXd rV(v.size());

  rV.setConstant(-1.0 / softmax((size_t)targets(0)));
  return rV;
}

std::vector<std::vector<double>> readCsv() {
  std::ifstream in{"iris.data"};
  std::string line;
  std::vector<std::vector<double>> out;
  while (std::getline(in, line)) {
    std::vector<double> instance;
    instance.push_back(std::stof(line.substr(0, 3)));
    instance.push_back(std::stof(line.substr(4, 7)));
    instance.push_back(std::stof(line.substr(8, 11)));
    instance.push_back(std::stof(line.substr(12, 15)));

    instance.push_back(out.size() < 50 ? 0 : out.size() < 100 ? 1 : 2);
    out.emplace_back(std::move(instance));
  }

  in.close();

  return out;
}

std::vector<std::vector<double>> GetData(std::vector<double> &targets) {
  std::random_device rd;
  std::mt19937 gen(rd());
  auto data = readCsv();
  std::shuffle(data.begin(), data.end(), gen);
  for (auto &in : data) {
    targets.push_back(in[in.size() - 1]);
    in.pop_back();
  }
  return data;
}

using namespace ML;
int main() {

  Net net(4, 20000, 0.01, CE, dCE);

  net.AddDense(4, ReLU, dReLU);
  net.AddDense(4, ReLU, dReLU);
  net.AddDense(3, SoftMax, dSoftMax);

  std::vector<double> targets;
  auto inputs = GetData(targets);

  std::vector<VectorXd> eigenInputs;

  for (auto i = 0; i < inputs.size(); ++i) {
    VectorXd v =
        Eigen::Map<Eigen::VectorXd>(inputs[i].data(), inputs[i].size());
    eigenInputs.emplace_back(std::move(v));
  }

  std::vector<VectorXd> eigenTargets;

  for (auto i = 0; i < targets.size(); ++i) {
    VectorXd v(1);
    v << targets[i];
    eigenTargets.emplace_back(v);
  }

  auto n = 100;

  net.Fit({eigenInputs.begin(), eigenInputs.begin() + n},
          {eigenTargets.begin(), eigenTargets.begin() + n});

  size_t correct = 0;
  for (auto i = n; i < 150; ++i) {
    auto predicted = net.Eval(eigenInputs[i]);
    double maxVal = 0;
    size_t maxId = 0;
    for (auto j = 0; j < predicted.size(); ++j) {
      if (predicted(j) > maxVal) {
        maxVal = predicted(j);
        maxId = j;
      }
    }
    if (maxId == targets[i])
      ++correct;
  }

  std::cout << std::endl
            << "\tSuccesses/rate\t" << static_cast<double>(correct) << "/"
            << static_cast<double>(correct) / (150. - static_cast<double>(n))
            << std::endl
            << std::endl;

  return 0;
}

//  Iteration/Cost is:	19999/0.572858
//
//	Successes/rate	49/0.98
