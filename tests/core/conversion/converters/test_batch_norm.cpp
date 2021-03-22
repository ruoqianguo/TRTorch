#include <string>
#include "core/compiler.h"
#include "gtest/gtest.h"
#include "tests/util/util.h"
#include "torch/script.h"
#include "torch/csrc/jit/ir/irparser.h"

TEST(Converters, ATenBatchNormConvertsCorrectly) {
  const auto graph = R"IR(
      graph(%0 : Tensor,
            %1: Float(5:1),
            %2: Float(5:1),
            %3: Float(5:1),
            %4: Float(5:1)):
        %5 : bool = prim::Constant[value=0]()
        %6 : float = prim::Constant[value=1.0000000000000001e-05]()
        %7 : float = prim::Constant[value=0.10000000000000001]()
        %8 : Tensor = aten::batch_norm(%0, %1, %2, %3, %4, %5, %6, %7, %5)
        return (%8))IR";

  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, &*g);

  auto in = at::randint(1, 10, {1, 5, 5, 5}, {at::kCUDA});
  auto gamma = at::randint(1, 10, {5}, {at::kCUDA});
  auto beta = at::randint(1, 10, {5}, {at::kCUDA});
  auto mean = at::randint(1, 10, {5}, {at::kCUDA});
  auto var = at::randint(1, 10, {5}, {at::kCUDA});

  auto params = trtorch::core::conversion::get_named_params(g->inputs(), {gamma, beta, mean, var});
  auto jit_results = trtorch::tests::util::RunGraph(g, params, {in});

  params = trtorch::core::conversion::get_named_params(g->inputs(), {gamma, beta, mean, var});
  auto trt_results = trtorch::tests::util::RunGraphEngine(g, params, {in});

  ASSERT_TRUE(trtorch::tests::util::almostEqual(jit_results[0], trt_results[0].reshape_as(jit_results[0]), 2e-6));
}

TEST(Converters, ATenLayerNormConvertsCorrectly) {
  const auto graph = R"IR(
      graph(%0 : Tensor,
            %1: Tensor,
            %2: Tensor):
        %5 : int[] = prim::Constant[value=[768]]()
        %6 : float = prim::Constant[value=1.0000000000000001e-05]()
        %7 : bool = prim::Constant[value=0]()
        %8 : Tensor = aten::layer_norm(%0, %5, %1, %2, %6, %7)
        return (%8))IR";

  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, &*g);

  auto in = at::randn({8, 256, 768}, {at::kCUDA}).to(torch::kFloat);
  auto weight = at::randn({768}, {at::kCUDA}).to(torch::kFloat);
  auto bias = at::randn({768}, {at::kCUDA}).to(torch::kFloat);

  auto params = trtorch::core::conversion::get_named_params(g->inputs(), {});
  auto jit_results = trtorch::tests::util::RunGraph(g, params, {in, weight, bias});

  params = trtorch::core::conversion::get_named_params(g->inputs(), {});
  auto trt_results = trtorch::tests::util::RunGraphEngine(g, params, {in, weight, bias});

  ASSERT_TRUE(trtorch::tests::util::almostEqual(jit_results[0], trt_results[0].reshape_as(jit_results[0]), 2e-6));
}

TEST(Converters, ATenLayerNormWeightNoneConvertsCorrectly) {
  const auto graph = R"IR(
      graph(%0 : Tensor):
        %1 : int[] = prim::Constant[value=[5, 5]]()
        %2 : float = prim::Constant[value=1.0000000000000001e-05]()
        %3 : bool = prim::Constant[value=0]()
        %4 : None = prim::Constant()
        %5 : Tensor = aten::layer_norm(%0, %1, %4, %4, %2, %3)
        return (%5))IR";

  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, &*g);

  auto in = at::randint(1, 10, {1, 5, 5, 5}, {at::kCUDA});

  auto params = trtorch::core::conversion::get_named_params(g->inputs(), {});
  auto jit_results = trtorch::tests::util::RunGraph(g, params, {in});

  params = trtorch::core::conversion::get_named_params(g->inputs(), {});
  auto trt_results = trtorch::tests::util::RunGraphEngine(g, params, {in});

  ASSERT_TRUE(trtorch::tests::util::almostEqual(jit_results[0], trt_results[0].reshape_as(jit_results[0]), 2e-6));
}