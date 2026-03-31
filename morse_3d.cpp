/*
 * morse_3d.cpp — Support 512³ with Zero-Allocation and Profiling (v3.0 Final)
 * =========================================================================
 * - Super-level set filtration (Max interpolation for Voxel Duality).
 * - 2N+1 Grid padding correctly implemented.
 * - Double precision keys to prevent float-truncation tie-breaker issues.
 * - Zero-persistence (Birth == Death) ghost pairs explicitly filtered.
 */

#include <algorithm>
#include <chrono>
#include <cmath>
#include <execution>
#include <iostream>
#include <limits>
#include <numeric>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include <unordered_map>
#include <omp.h>

namespace py = pybind11;

struct StaticVec {
    int32_t data[6];
    uint8_t sz = 0;
    inline void push_back(int32_t v) { data[sz++] = v; }
    inline void clear() { sz = 0; }
    inline size_t size() const { return sz; }
    inline bool empty() const { return sz == 0; }
    inline int32_t* begin() { return data; }
    inline int32_t* end() { return data + sz; }
    inline const int32_t* begin() const { return data; }
    inline const int32_t* end() const { return data + sz; }
    inline int32_t operator[](size_t i) const { return data[i]; }
};

inline int64_t get_global_id(int64_t i, int64_t j, int64_t k, int64_t M_cells,
                             int64_t L_cells) {
  return i * M_cells * L_cells + j * L_cells + k;
}

inline void get_ijk(int64_t g_id, int64_t M_cells, int64_t L_cells, int64_t &i,
                    int64_t &j, int64_t &k) {
  k = g_id % L_cells;
  int64_t temp = g_id / L_cells;
  j = temp % M_cells;
  i = temp / M_cells;
}

// 核心重构 1：Voxel Duality 的 Max 插值 (用于 Super-level set)
double get_cell_value(int64_t c_i, int64_t c_j, int64_t c_k, const double *grid,
                      int64_t N, int64_t M, int64_t L) {
  double max_val = -std::numeric_limits<double>::infinity();
  
  // 巧妙的边界自适应：奇数取自身，偶数(边界)跨越两个相邻 Voxel
  int64_t i_min = (c_i % 2 == 1) ? (c_i / 2) : std::max((int64_t)0, c_i / 2 - 1);
  int64_t i_max = (c_i % 2 == 1) ? (c_i / 2) : std::min(N - 1, c_i / 2);
  
  int64_t j_min = (c_j % 2 == 1) ? (c_j / 2) : std::max((int64_t)0, c_j / 2 - 1);
  int64_t j_max = (c_j % 2 == 1) ? (c_j / 2) : std::min(M - 1, c_j / 2);
  
  int64_t k_min = (c_k % 2 == 1) ? (c_k / 2) : std::max((int64_t)0, c_k / 2 - 1);
  int64_t k_max = (c_k % 2 == 1) ? (c_k / 2) : std::min(L - 1, c_k / 2);

  for (int64_t i = i_min; i <= i_max; ++i) {
    for (int64_t j = j_min; j <= j_max; ++j) {
      for (int64_t k = k_min; k <= k_max; ++k) {
        max_val = std::max(max_val, grid[(i * M + j) * L + k]);
      }
    }
  }
  return max_val;
}

py::dict extract_persistence_3d_morse(py::array_t<double> grid_array) {
  py::buffer_info buf = grid_array.request();
  if (buf.ndim != 3) throw std::runtime_error("Requires 3D numpy array");

  int64_t N = buf.shape[0];
  int64_t M = buf.shape[1];
  int64_t L = buf.shape[2];
  const double *grid = static_cast<const double *>(buf.ptr);

  // 核心重构 2：恢复 2N+1 的 Voxel 拓扑网格
  int64_t N_cells = 2 * N + 1;
  int64_t M_cells = 2 * M + 1;
  int64_t L_cells = 2 * L + 1;
  int64_t K = N_cells * M_cells * L_cells;

  auto t_start = std::chrono::high_resolution_clock::now();

  // Phase 1: Grid Init & Parallel Sort
  auto t1_start = std::chrono::high_resolution_clock::now();
  
  // 核心重构 3：使用 double 防止浮点截断导致假 Tie-breaker
  std::vector<double> keys(K);
  std::vector<uint8_t> dims(K);
  std::vector<int32_t> sorted_idx(K);

  #pragma omp parallel for collapse(3)
  for (int64_t i = 0; i < N_cells; ++i) {
    for (int64_t j = 0; j < M_cells; ++j) {
      for (int64_t k = 0; k < L_cells; ++k) {
        int64_t g_id = i * M_cells * L_cells + j * L_cells + k;
        dims[g_id] = (uint8_t)((i % 2) + (j % 2) + (k % 2));
        keys[g_id] = get_cell_value(i, j, k, grid, N, M, L);
      }
    }
  }

  std::iota(sorted_idx.begin(), sorted_idx.end(), 0);
  
  // 严谨的 Tie-breaker: 标量大优先 -> 维度低优先 -> 内存ID小优先
  std::sort(std::execution::par_unseq, sorted_idx.begin(), sorted_idx.end(),
      [&](int32_t a, int32_t b) {
          if (keys[a] != keys[b]) return keys[a] > keys[b];
          if (dims[a] != dims[b]) return dims[a] < dims[b];
          return a < b;
      }
  );

  std::vector<int32_t> global_to_sort(K);
  #pragma omp parallel for
  for (int32_t idx = 0; idx < K; ++idx) global_to_sort[sorted_idx[idx]] = idx;
  
  auto t1_end = std::chrono::high_resolution_clock::now();
  std::cout << "[DMT Profile] Phase 1 (Init & Sort) 耗时: " << std::chrono::duration<double>(t1_end - t1_start).count() << " 秒\n";

  auto get_boundary_static = [&](int32_t idx, StaticVec& out, const std::vector<int32_t>& g2s) {
    out.clear();
    int64_t g_id = sorted_idx[idx];
    int64_t c_i, c_j, c_k;
    get_ijk(g_id, M_cells, L_cells, c_i, c_j, c_k);
    if (c_i % 2 != 0) {
      out.push_back(g2s[get_global_id(c_i - 1, c_j, c_k, M_cells, L_cells)]);
      out.push_back(g2s[get_global_id(c_i + 1, c_j, c_k, M_cells, L_cells)]);
    }
    if (c_j % 2 != 0) {
      out.push_back(g2s[get_global_id(c_i, c_j - 1, c_k, M_cells, L_cells)]);
      out.push_back(g2s[get_global_id(c_i, c_j + 1, c_k, M_cells, L_cells)]);
    }
    if (c_k % 2 != 0) {
      out.push_back(g2s[get_global_id(c_i, c_j, c_k - 1, M_cells, L_cells)]);
      out.push_back(g2s[get_global_id(c_i, c_j, c_k + 1, M_cells, L_cells)]);
    }
    std::sort(out.begin(), out.end(), std::greater<int32_t>());
  };

  auto get_cofaces_static = [&](int32_t idx, StaticVec& out, const std::vector<int32_t>& g2s) {
    out.clear();
    int64_t g_id = sorted_idx[idx];
    int64_t c_i, c_j, c_k;
    get_ijk(g_id, M_cells, L_cells, c_i, c_j, c_k);
    if (c_i % 2 == 0) {
      if (c_i - 1 >= 0) out.push_back(g2s[get_global_id(c_i - 1, c_j, c_k, M_cells, L_cells)]);
      if (c_i + 1 < N_cells) out.push_back(g2s[get_global_id(c_i + 1, c_j, c_k, M_cells, L_cells)]);
    }
    if (c_j % 2 == 0) {
      if (c_j - 1 >= 0) out.push_back(g2s[get_global_id(c_i, c_j - 1, c_k, M_cells, L_cells)]);
      if (c_j + 1 < M_cells) out.push_back(g2s[get_global_id(c_i, c_j + 1, c_k, M_cells, L_cells)]);
    }
    if (c_k % 2 == 0) {
      if (c_k - 1 >= 0) out.push_back(g2s[get_global_id(c_i, c_j, c_k - 1, M_cells, L_cells)]);
      if (c_k + 1 < L_cells) out.push_back(g2s[get_global_id(c_i, c_j, c_k + 1, M_cells, L_cells)]);
    }
    std::sort(out.begin(), out.end());
  };

  // Phase 2: Discrete Gradient Field
  auto t2_start = std::chrono::high_resolution_clock::now();
  std::vector<int32_t> pair(K, -1);
  std::vector<bool> is_critical(K, false);
  
  StaticVec cof_cache, bnd_cache;
  for (int32_t i = 0; i < K; ++i) {
    if (pair[i] != -1) continue;
    get_cofaces_static(i, cof_cache, global_to_sort);
    int32_t best_beta = -1;
    for (size_t idx = 0; idx < cof_cache.size(); ++idx) {
      int32_t beta = cof_cache[idx];
      if (pair[beta] != -1) continue;
      get_boundary_static(beta, bnd_cache, global_to_sort);
      int32_t current_highest_unpaired_face = -1;
      for (size_t f_idx = 0; f_idx < bnd_cache.size(); ++f_idx) {
          int32_t f_prime = bnd_cache[f_idx];
          if (pair[f_prime] == -1) {
              if (current_highest_unpaired_face == -1 || f_prime > current_highest_unpaired_face) {
                  current_highest_unpaired_face = f_prime;
              }
          }
      }
      if (current_highest_unpaired_face == i) {
        best_beta = beta;
        break;
      }
    }
    if (best_beta != -1) {
      pair[i] = best_beta;
      pair[best_beta] = i;
    } else {
      is_critical[i] = true;
    }
  }
  auto t2_end = std::chrono::high_resolution_clock::now();
  std::cout << "[DMT Profile] Phase 2 (Pairing) 耗时: " << std::chrono::duration<double>(t2_end - t2_start).count() << " 秒\n";

  int32_t num_critical = 0;
  for (int32_t i = 0; i < K; ++i) if (is_critical[i]) num_critical++;
  std::cout << "[DMT Profile] Critical Cells 数量: " << num_critical << "\n";

  // Phase 3: Iterative V-Path DFS
  auto t3_start = std::chrono::high_resolution_clock::now();
  std::unordered_map<int32_t, std::vector<int32_t>> memo;
  std::vector<bool> in_stack(K, false);

  auto get_morse_boundary = [&](int32_t P_start) -> const std::vector<int32_t>& {
    if (memo.count(P_start)) return memo[P_start];
    struct Frame {
      int32_t P;
      uint8_t step;
      StaticVec faces;
      std::vector<int32_t> bnd; 
    };
    std::vector<Frame> st;
    StaticVec initial_faces;
    get_boundary_static(P_start, initial_faces, global_to_sort);
    st.push_back({P_start, 0, initial_faces, {}});
    while (!st.empty()) {
      auto &top = st.back();
      if (top.step == 0) in_stack[top.P] = true;
      int32_t paired_face = pair[top.P];
      bool moved_down = false;
      while (top.step < top.faces.size()) {
        int32_t f = top.faces[top.step++];
        if (f == paired_face) continue;
        if (is_critical[f]) {
          top.bnd.push_back(f);
        } else {
          int32_t next_P = pair[f];
          if (next_P != -1 && next_P != top.P && dims[sorted_idx[next_P]] == dims[sorted_idx[P_start]]) {
            if (in_stack[next_P]) continue;
            if (memo.count(next_P)) {
              for (int32_t cached_bnd : memo[next_P]) top.bnd.push_back(cached_bnd);
            } else {
              StaticVec next_faces;
              get_boundary_static(next_P, next_faces, global_to_sort);
              st.push_back({next_P, 0, next_faces, {}});
              moved_down = true;
              break;
            }
          }
        }
      }
      if (!moved_down) {
        in_stack[top.P] = false;
        std::sort(top.bnd.begin(), top.bnd.end(), std::greater<int32_t>());
        std::vector<int32_t> reduced_bnd;
        for (size_t i = 0; i < top.bnd.size();) {
          size_t j = i + 1;
          while (j < top.bnd.size() && top.bnd[j] == top.bnd[i]) ++j;
          if ((j - i) & 1) reduced_bnd.push_back(top.bnd[i]);
          i = j;
        }
        memo[top.P] = std::move(reduced_bnd);
        const auto& finished_bnd = memo[top.P];
        st.pop_back();
        if (!st.empty()) {
          for (int32_t b : finished_bnd) st.back().bnd.push_back(b);
        }
      }
    }
    return memo[P_start];
  };

  auto t3_end = std::chrono::high_resolution_clock::now();
  std::cout << "[DMT Profile] Phase 3 (V-Path Pathfinding) 耗时: " << std::chrono::duration<double>(t3_end - t3_start).count() << " 秒\n";

  // Phase 4: Reduction
  auto t4_start = std::chrono::high_resolution_clock::now();
  std::vector<int32_t> crit_indices;
  crit_indices.reserve(num_critical);
  for (int32_t i = 0; i < K; ++i) if (is_critical[i]) crit_indices.push_back(i);

  std::unordered_map<int32_t, int32_t> k_to_crit;
  for (size_t idx = 0; idx < crit_indices.size(); ++idx) k_to_crit[crit_indices[idx]] = (int32_t)idx;

  std::vector<std::vector<int32_t>> R(crit_indices.size());
  for (size_t idx = 0; idx < crit_indices.size(); ++idx) {
    const auto& morse_bnd = get_morse_boundary(crit_indices[idx]);
    R[idx].reserve(morse_bnd.size());
    for (int32_t b_idx : morse_bnd) R[idx].push_back(k_to_crit[b_idx]);
    
    // 保证边界矩阵每列按降序排列，确保 Pivot 位置准确
    std::sort(R[idx].begin(), R[idx].end(), std::greater<int32_t>());
  }

  std::unordered_map<int32_t, int32_t> pivot_to_col;
  std::vector<std::pair<int32_t, int32_t>> pairs;
  for (size_t j = 0; j < R.size(); ++j) {
    while (!R[j].empty()) {
      int32_t pivot = R[j].front();
      if (pivot_to_col.count(pivot)) {
        int32_t k = pivot_to_col[pivot];
        std::vector<int32_t> new_R;
        std::set_symmetric_difference(R[j].begin(), R[j].end(), R[k].begin(),
                                      R[k].end(), std::back_inserter(new_R),
                                      std::greater<int32_t>());
        R[j] = std::move(new_R);
      } else {
        pivot_to_col[pivot] = (int32_t)j;
        pairs.push_back({pivot, (int32_t)j});
        break;
      }
    }
  }
  auto t4_end = std::chrono::high_resolution_clock::now();
  std::cout << "[DMT Profile] Phase 4 (Reduction) 耗时: " << std::chrono::duration<double>(t4_end - t4_start).count() << " 秒\n";

  auto t_end = std::chrono::high_resolution_clock::now();
  std::cout << "[DMT Profile] 总耗时: " << std::chrono::duration<double>(t_end - t_start).count() << " 秒\n";

  std::vector<bool> is_paired_crit(crit_indices.size(), false);
  for (auto p : pairs) {
    is_paired_crit[p.first] = true;
    is_paired_crit[p.second] = true;
  }

  std::vector<int32_t> essential_h0;
  for (size_t i = 0; i < crit_indices.size(); ++i) {
    if (!is_paired_crit[i] && dims[sorted_idx[crit_indices[i]]] == 0) {
      essential_h0.push_back(crit_indices[i]);
    }
  }

  // 核心重构 4：严格过滤 Zero-Persistence 幽灵对
  std::vector<std::pair<int32_t, int32_t>> valid_pairs;
  for (auto p : pairs) {
    int32_t b_idx = crit_indices[p.first];
    int32_t d_idx = crit_indices[p.second];
    
    // 只有 Birth 和 Death 标量值不同，才是真实的拓扑特征
    if (keys[sorted_idx[b_idx]] != keys[sorted_idx[d_idx]]) {
      valid_pairs.push_back({b_idx, d_idx});
    }
  }

  size_t num_pairs_out = valid_pairs.size();
  size_t num_h0_out = essential_h0.size();
  size_t total_out = num_pairs_out + num_h0_out;

  py::array_t<double> b_vals_arr(total_out);
  py::array_t<double> d_vals_arr(total_out);
  py::array_t<int> dims_arr(total_out);
  py::array_t<double> b_coords_arr({total_out, (size_t)3});
  py::array_t<double> d_coords_arr({total_out, (size_t)3});

  auto b_v = b_vals_arr.mutable_unchecked<1>();
  auto d_v = d_vals_arr.mutable_unchecked<1>();
  auto d_m = dims_arr.mutable_unchecked<1>();
  auto b_c = b_coords_arr.mutable_unchecked<2>();
  auto d_c = d_coords_arr.mutable_unchecked<2>();

  for (size_t i = 0; i < num_pairs_out; ++i) {
    int32_t b_idx = valid_pairs[i].first;
    int32_t d_idx = valid_pairs[i].second;
    b_v(i) = keys[sorted_idx[b_idx]];
    d_v(i) = keys[sorted_idx[d_idx]];
    d_m(i) = dims[sorted_idx[b_idx]];
    int64_t gid_b = sorted_idx[b_idx], gid_d = sorted_idx[d_idx], bi, bj, bk, di, dj, dk;
    get_ijk(gid_b, M_cells, L_cells, bi, bj, bk); get_ijk(gid_d, M_cells, L_cells, di, dj, dk);
    
    // 根据 2N+1 坐标系，除以 2 并偏移 0.5 映射回原空间（如果需要的话，目前保留之前的除2逻辑）
    b_c(i, 0) = bi/2.0; b_c(i, 1) = bj/2.0; b_c(i, 2) = bk/2.0;
    d_c(i, 0) = di/2.0; d_c(i, 1) = dj/2.0; d_c(i, 2) = dk/2.0;
  }

  for (size_t i = 0; i < num_h0_out; ++i) {
    size_t out_idx = num_pairs_out + i;
    int32_t b_idx = essential_h0[i];
    b_v(out_idx) = keys[sorted_idx[b_idx]];
    d_v(out_idx) = -std::numeric_limits<double>::infinity(); // 超水平集 Essential H0 应该死于负无穷
    d_m(out_idx) = 0;
    int64_t gid = sorted_idx[b_idx], bi, bj, bk;
    get_ijk(gid, M_cells, L_cells, bi, bj, bk);
    b_c(out_idx, 0) = bi/2.0; b_c(out_idx, 1) = bj/2.0; b_c(out_idx, 2) = bk/2.0;
    d_c(out_idx, 0) = 0.0; d_c(out_idx, 1) = 0.0; d_c(out_idx, 2) = 0.0;
  }

  py::dict res;
  res["births"] = b_vals_arr; res["deaths"] = d_vals_arr; res["dims"] = dims_arr;
  res["birth_coords"] = b_coords_arr; res["death_coords"] = d_coords_arr;
  return res;
}

PYBIND11_MODULE(morse_3d, m) {
  m.def("extract_persistence_3d_morse", &extract_persistence_3d_morse,
        "3D Cubical Complex DMT Persistence Engine v3.0 (Final Voxel-Max Setup)");
}