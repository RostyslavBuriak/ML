// Simple ANN for solving XOR, implemented for learning purpose and hardcoded
// most of the variables. ANN has 2 input nodes, 1 hidden layer with 2 nodes and
// 1 output node. It uses MSE as cost function, and sigmoid as activation.
// Tried to implement backpropagation

#include "Eigen/Dense"
#include "Eigen/src/Core/Matrix.h"
#include <array>
#include <cmath>
#include <iostream>
#include <random>

inline double Sigmoid(const double x) noexcept {
  return 1.0 / (1.0 + std::exp(-x));
}

inline double d_Sigmoid(const double x) noexcept {
  return Sigmoid(x) * (1.0 - Sigmoid(x));
}

using Eigen::MatrixXd;

MatrixXd input{{0, 0}, {1, 0}, {0, 1}, {1, 1}};

std::array<uint8_t, 4> output{0, 1, 1, 0};

std::array<double, 6> weights;

std::array<double, 3> biases;

const size_t iterations = 100000;

const double theta = 0.5;

double execute(char i1, char i2) {
  auto a1 = i1 * weights[0] + i2 * weights[2] + biases[0];
  auto a2 = i1 * weights[1] + i2 * weights[3] + biases[1];
  auto a1s = Sigmoid(a1);
  auto a2s = Sigmoid(a2);
  auto a3 = a1s * weights[4] + a2s * weights[5] + biases[2];
  return Sigmoid(a3);
}

int main() {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<double> dis(
      0.3, 0.7); // decided to use 0.3 - 0.7 range since some extreme values
                 // could potentially affect weights

  for (auto &x : weights) {
    x = dis(gen);
  }

  for (auto &x : biases) {
    x = dis(gen);
  }

  for (size_t i = 0; i < iterations; ++i) {

    std::array<double, 6> dWeights{};
    std::array<double, 3> dBiases{};

    for (size_t q = 0; q < 4; ++q) {
      auto a1 = weights[0] * input(q, 0) + weights[2] * input(q, 1) + biases[0];
      auto a2 = weights[1] * input(q, 0) + weights[3] * input(q, 1) + biases[1];

      auto a1s = Sigmoid(a1);
      auto a2s = Sigmoid(a2);

      auto a3 = weights[4] * a1s + weights[5] * a2s + biases[2];

      auto a3s = Sigmoid(a3);

      auto error = a3s - output[q];
      auto d_Sum = error * d_Sigmoid(a3);

      dWeights[4] += (d_Sum * a1s);
      dWeights[5] += (d_Sum * a2s);

      dWeights[0] += (d_Sum * weights[4] * d_Sigmoid(a1) * input(q, 0));
      dWeights[1] += (d_Sum * weights[5] * d_Sigmoid(a2) * input(q, 0));

      dWeights[2] += (d_Sum * weights[4] * d_Sigmoid(a1) * input(q, 1));
      dWeights[3] += (d_Sum * weights[5] * d_Sigmoid(a2) * input(q, 1));

      dBiases[2] += (error * d_Sigmoid(a3));
      dBiases[0] += (d_Sum * weights[4] * d_Sigmoid(a1));
      dBiases[1] += (d_Sum * weights[5] * d_Sigmoid(a2));
    }

    for (size_t q = 0; q < 6; ++q) {
      dWeights[q] /= 4.0;
      weights[q] -= theta * dWeights[q];
    }

    for (size_t q = 0; q < 3; ++q) {
      dBiases[q] /= 4.0;
      biases[q] -= theta * dBiases[q];
    }
  }

  std::cout << "0 0 = " << execute(0, 0) << std::endl;
  std::cout << "1 0 = " << execute(1, 0) << std::endl;
  std::cout << "0 1 = " << execute(0, 1) << std::endl;
  std::cout << "1 1 = " << execute(1, 1) << std::endl;

  return 0;
}
