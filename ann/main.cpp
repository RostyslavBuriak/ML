#include "ann.hpp"
#include "common.hpp"
#include <fstream>
#include <iostream>
#include <ostream>
#include <random>
#include <vector>

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

int main() {
  ML::Net net(20000, 0.01, ML::CE, ML::dCE);
  std::vector<double> targets;
  auto inputs = GetData(targets);
  const size_t n = 120;
  net.AddInputs({inputs.begin(), inputs.begin() + n},
                {targets.begin(), targets.begin() + n});
  net.AddDense(4, ML::ReLU, ML::dReLU);
  net.AddDense(4, ML::ReLU, ML::dReLU);
  net.AddDense(4, ML::ReLU, ML::dReLU);
  net.AddDense(3, ML::Empty, ML::dEmpty);

  net.Learn();

  size_t correct = 0;
  for (auto i = n; i < 150; ++i) {
    auto predicted = net.Predict(inputs[i]);
    double maxVal = 0;
    size_t maxId = 0;
    for (auto j = 0; j < predicted.size(); ++j) {
      if (predicted[j] > maxVal) {
        maxVal = predicted[j];
        maxId = j;
      }
    }
    if (maxId == targets[i])
      ++correct;
  }

  ML::Net xorNet(5000, 0.5, ML::MSE, ML::dMSE);

  xorNet.AddDense(2, ML::Sigmoid, ML::dSigmoid);
  xorNet.AddDense(2, ML::Sigmoid, ML::dSigmoid);
  xorNet.AddDense(1, ML::Sigmoid, ML::dSigmoid);
  std::vector<std::vector<double>> xorInputs{{0, 0}, {1, 0}, {0, 1}, {1, 1}};
  xorNet.AddInputs(xorInputs, {0, 1, 1, 0});

  xorNet.Learn();

  ML::Net orNet(2000, 0.5, ML::MSE, ML::dMSE);

  orNet.AddDense(2, ML::Sigmoid, ML::dSigmoid);
  orNet.AddDense(2, ML::Sigmoid, ML::dSigmoid);
  orNet.AddDense(1, ML::Sigmoid, ML::dSigmoid);
  std::vector<std::vector<double>> orInputs{{0, 0}, {1, 0}, {0, 1}, {1, 1}};
  orNet.AddInputs(orInputs, {0, 1, 1, 1});

  orNet.Learn();

  std::cout << std::endl
            << "\tSuccesses/rate\t" << static_cast<double>(correct) << "/"
            << static_cast<double>(correct) / (150. - static_cast<double>(n))
            << std::endl
            << std::endl;

  for (const auto &v : xorInputs) {
    const auto predicted = xorNet.Predict(v)[0];
    std::cout << v[0] << " " << v[1] << "\t" << predicted << std::endl;
  }

  std::cout << std::endl;

  for (const auto &v : orInputs) {
    const auto predicted = orNet.Predict(v)[0];
    std::cout << v[0] << " " << v[1] << "\t" << predicted << std::endl;
  }
}

// iteration/error: 1999/0.00182907

//        Successes/rate  30/1

// 0 0     0.0502368
// 1 0     0.954338
// 0 1     0.954319
// 1 1     0.0489037

// 0 0     0.0671991
// 1 0     0.963002
// 0 1     0.962961
// 1 1     0.992638
