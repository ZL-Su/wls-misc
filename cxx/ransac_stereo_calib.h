#pragma once

#include <string>
#include <vector>
#include <array>
#include <random>
#include <cmath>
#include <cstdlib>
#include <ctime>

using paired_param_t = std::pair<std::array<double, 10>, std::array<double, 10>>;
// Randomly sample `batch_size` pose images from the given image sequence
std::vector<size_t> gen_rand_samples(size_t num_poses, size_t batch_size)
{
   std::vector<size_t> indices(num_poses);
   std::iota(indices.begin(), indices.end(), 0); // Fill vector with 0, 1, ..., N-1

   std::random_device rd;
   std::mt19937 generator(rd());
   std::shuffle(indices.begin(), indices.end(), generator);
   indices.resize(batch_size);
   return indices;
}

// 计算left--right图像对的三维重构误差
double calc_error(const paired_param_t &query, std::string left, std::string right)
{
   double error = 0;

   return error;
}

paired_param_t calib_step(const std::vector<std::string> paths[2])
{
   // 1. Corner detection

   // 2. Calibration

   // 3. Convert rotation matrix to vector

   paired_param_t params;
   // 4. Fill calibrated results into params

   return params;
}

// 使用RANSAC标定
paired_param_t calib_ransac(const std::vector<std::string> paths[2], int iters, double error_threshold = 1e-2)
{
   srand(static_cast<unsigned>(time(nullptr)));
   const auto num_poses = std::min(paths[0].size(), paths[1].size());

   paired_param_t best;
   int best_inlier_count = 0;
   for (int i = 0; i < iters; ++i)
   {
      // 随机选择15张标定图
      const auto samples = gen_rand_samples(num_poses, 15 /*sample 15 poses each time*/);
      std::vector<std::string> sampled_paths[2];
      for (auto i = 0; i < samples.size(); ++i)
      {
         sampled_paths[0].push_back(paths[0][samples[i]]);
         sampled_paths[1].push_back(paths[1][samples[i]]);
      }

      // 标定一次
      auto candidate = calib_step(sampled_paths);

      // 计算内点数量， voting
      int inlier_count = 0;
      for (auto i = 0; i < num_poses; ++i)
      {
         const auto error = calc_error(candidate, paths[0][i], paths[1][i]);
         if (error < error_threshold)
         {
            inlier_count++;
         }
      }

      // 更新最佳模型
      if (inlier_count > best_inlier_count)
      {
         best_inlier_count = inlier_count;
         best = candidate;
      }

      // 设置终止条件
      if (best_inlier_count > num_poses * 0.9)
      {
         break;
      }
   }

   return best;
}