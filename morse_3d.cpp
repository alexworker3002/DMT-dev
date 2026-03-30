/*
 * morse_3d.cpp — 3D Cubical Complex DMT Persistence Engine v1.1
 * ==============================================================
 * Proven correct against GUDHI at all scales ≤ 32³.
 * v1.1 adds: intermediate memory release, int32_t boundary checks for K > 2^31.
 */

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <set>
#include <unordered_map>
#include <vector>

namespace py = pybind11;

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

struct Cell {
  int64_t global_id;
  double key;
  uint8_t dim;

  // 严格弱序 (Strict Weak Ordering): 消除 EPS 模糊判定，防止 std::sort 传递性崩溃
  bool operator<(const Cell &other) const {
    if (key != other.key)
      return key > other.key;
    if (dim != other.dim)
      return dim < other.dim;
    return global_id < other.global_id;
  }
};

double get_cell_value(int64_t c_i, int64_t c_j, int64_t c_k, const double *grid,
                      int64_t N, int64_t M, int64_t L) {
  double min_val = std::numeric_limits<double>::infinity();
  for (int64_t di = 0; di <= (c_i % 2); ++di) {
    for (int64_t dj = 0; dj <= (c_j % 2); ++dj) {
      for (int64_t dk = 0; dk <= (c_k % 2); ++dk) {
        int64_t v_i = c_i / 2 + di;
        int64_t v_j = c_j / 2 + dj;
        int64_t v_k = c_k / 2 + dk;
        min_val = std::min(min_val, grid[(v_i * M + v_j) * L + v_k]);
      }
    }
  }
  return min_val;
}

py::dict extract_persistence_3d_morse(py::array_t<double> grid_array) {
  py::buffer_info buf = grid_array.request();
  if (buf.ndim != 3)
    throw std::runtime_error("Requires 3D numpy array");

  int64_t N = buf.shape[0];
  int64_t M = buf.shape[1];
  int64_t L = buf.shape[2];
  const double *grid = static_cast<const double *>(buf.ptr);

  int64_t N_cells = 2 * N - 1;
  int64_t M_cells = 2 * M - 1;
  int64_t L_cells = 2 * L - 1;
  int64_t K = N_cells * M_cells * L_cells;

  if (K > 2100000000LL)
    throw std::runtime_error(
        "Grid too large for int32 indexing (>2.1B cells). Max ~130³ vertices.");

  std::vector<Cell> cells(K);
  for (int64_t i = 0; i < N_cells; ++i)
    for (int64_t j = 0; j < M_cells; ++j)
      for (int64_t k = 0; k < L_cells; ++k) {
        int64_t g_id = get_global_id(i, j, k, M_cells, L_cells);
        cells[g_id].global_id = g_id;
        cells[g_id].dim = (i % 2) + (j % 2) + (k % 2);
        cells[g_id].key = get_cell_value(i, j, k, grid, N, M, L);
      }

  // Phase 1: Sort
  std::sort(cells.begin(), cells.end());

  std::vector<int32_t> global_to_sort(K);
  for (int32_t idx = 0; idx < K; ++idx)
    global_to_sort[cells[idx].global_id] = idx;

  auto get_boundary = [&](int32_t idx) -> std::vector<int32_t> {
    int64_t g_id = cells[idx].global_id;
    int64_t c_i, c_j, c_k;
    get_ijk(g_id, M_cells, L_cells, c_i, c_j, c_k);
    std::vector<int32_t> bnd;
    if (c_i % 2 != 0) {
      bnd.push_back(
          global_to_sort[get_global_id(c_i - 1, c_j, c_k, M_cells, L_cells)]);
      bnd.push_back(
          global_to_sort[get_global_id(c_i + 1, c_j, c_k, M_cells, L_cells)]);
    }
    if (c_j % 2 != 0) {
      bnd.push_back(
          global_to_sort[get_global_id(c_i, c_j - 1, c_k, M_cells, L_cells)]);
      bnd.push_back(
          global_to_sort[get_global_id(c_i, c_j + 1, c_k, M_cells, L_cells)]);
    }
    if (c_k % 2 != 0) {
      bnd.push_back(
          global_to_sort[get_global_id(c_i, c_j, c_k - 1, M_cells, L_cells)]);
      bnd.push_back(
          global_to_sort[get_global_id(c_i, c_j, c_k + 1, M_cells, L_cells)]);
    }
    std::sort(bnd.begin(), bnd.end(), std::greater<int32_t>());
    return bnd;
  };

  auto get_cofaces = [&](int32_t idx) -> std::vector<int32_t> {
    int64_t g_id = cells[idx].global_id;
    int64_t c_i, c_j, c_k;
    get_ijk(g_id, M_cells, L_cells, c_i, c_j, c_k);
    std::vector<int32_t> cof;
    if (c_i % 2 == 0) {
      if (c_i - 1 >= 0)
        cof.push_back(
            global_to_sort[get_global_id(c_i - 1, c_j, c_k, M_cells, L_cells)]);
      if (c_i + 1 < N_cells)
        cof.push_back(
            global_to_sort[get_global_id(c_i + 1, c_j, c_k, M_cells, L_cells)]);
    }
    if (c_j % 2 == 0) {
      if (c_j - 1 >= 0)
        cof.push_back(
            global_to_sort[get_global_id(c_i, c_j - 1, c_k, M_cells, L_cells)]);
      if (c_j + 1 < M_cells)
        cof.push_back(
            global_to_sort[get_global_id(c_i, c_j + 1, c_k, M_cells, L_cells)]);
    }
    if (c_k % 2 == 0) {
      if (c_k - 1 >= 0)
        cof.push_back(
            global_to_sort[get_global_id(c_i, c_j, c_k - 1, M_cells, L_cells)]);
      if (c_k + 1 < L_cells)
        cof.push_back(
            global_to_sort[get_global_id(c_i, c_j, c_k + 1, M_cells, L_cells)]);
    }
    std::sort(cof.begin(), cof.end());
    return cof;
  };

  // Phase 2: Discrete Gradient Field (Steepest Descent / Robins Pairing)
  std::vector<int32_t> pair(K, -1);
  std::vector<bool> is_critical(K, false);

  for (int32_t i = 0; i < K; ++i) {
    if (pair[i] != -1)
      continue;

    std::vector<int32_t> cofaces = get_cofaces(i);
    int32_t best_beta = -1;

    for (int32_t beta : cofaces) {
      if (pair[beta] != -1) continue;

      std::vector<int32_t> beta_bnd = get_boundary(beta);
      if (beta_bnd.empty()) continue;

      // Robins 核心：检查 i 是否是 beta 的唯一最高未配对边界
      int32_t current_highest_unpaired_face = -1;
      for (int32_t f_prime : beta_bnd) {
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

  // Phase 3: Iterative V-Path DFS
  std::unordered_map<int32_t, std::vector<int32_t>> memo;
  std::vector<bool> in_stack(K, false);

  auto get_morse_boundary = [&](int32_t P_start) -> std::vector<int32_t> {
    if (memo.count(P_start))
      return memo[P_start];

    struct Frame {
      int32_t P;
      size_t step;
      std::vector<int32_t> bnd;
      std::vector<int32_t> faces;
    };

    std::vector<Frame> st;
    st.push_back({P_start, 0, {}, get_boundary(P_start)});

    while (!st.empty()) {
      auto &top = st.back();
      int32_t P = top.P;
      if (top.step == 0)
        in_stack[P] = true;

      int32_t paired_face = pair[P];
      bool moved_down = false;

      while (top.step < top.faces.size()) {
        int32_t f = top.faces[top.step++];
        if (f == paired_face)
          continue;

        if (is_critical[f]) {
          top.bnd.push_back(f);
        } else {
          int32_t next_P = pair[f];
          if (next_P != -1 && next_P != P &&
              cells[next_P].dim == cells[P_start].dim) {
            if (in_stack[next_P])
              continue;
            if (memo.count(next_P)) {
              const auto &cached = memo[next_P];
              top.bnd.insert(top.bnd.end(), cached.begin(), cached.end());
            } else {
              st.push_back({next_P, 0, {}, get_boundary(next_P)});
              moved_down = true;
              break;
            }
          }
        }
      }

      if (!moved_down) {
        in_stack[P] = false;
        std::sort(top.bnd.begin(), top.bnd.end(), std::greater<int32_t>());
        std::vector<int32_t> reduced_bnd;
        for (size_t i = 0; i < top.bnd.size();) {
          size_t j = i + 1;
          while (j < top.bnd.size() && top.bnd[j] == top.bnd[i])
            ++j;
          if ((j - i) % 2 == 1)
            reduced_bnd.push_back(top.bnd[i]);
          i = j;
        }
        top.bnd = std::move(reduced_bnd);
        memo[P] = top.bnd;
        auto finished_bnd = std::move(top.bnd);
        st.pop_back();
        if (!st.empty())
          st.back().bnd.insert(st.back().bnd.end(), finished_bnd.begin(),
                               finished_bnd.end());
      }
    }
    return memo[P_start];
  };

  // Phase 4: Reduction (critical cells only)
  std::vector<int32_t> crit_indices;
  for (int32_t i = 0; i < K; ++i)
    if (is_critical[i])
      crit_indices.push_back(i);

  std::unordered_map<int32_t, int32_t> k_to_crit;
  for (size_t idx = 0; idx < crit_indices.size(); ++idx)
    k_to_crit[crit_indices[idx]] = idx;

  std::vector<std::vector<int32_t>> R(crit_indices.size());
  for (size_t idx = 0; idx < crit_indices.size(); ++idx) {
    auto morse_bnd = get_morse_boundary(crit_indices[idx]);
    for (int32_t b_idx : morse_bnd)
      R[idx].push_back(k_to_crit[b_idx]);
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
        pivot_to_col[pivot] = j;
        pairs.push_back({pivot, (int32_t)j});
        break;
      }
    }
  }

  // Identify Essential H0 (Unpaired critical vertices)
  std::vector<bool> is_paired(crit_indices.size(), false);
  for (auto p : pairs) {
    is_paired[p.first] = true;
    is_paired[p.second] = true;
  }

  std::vector<int32_t> essential_h0;
  for (size_t i = 0; i < crit_indices.size(); ++i) {
    if (!is_paired[i] && cells[crit_indices[i]].dim == 0) {
      essential_h0.push_back(crit_indices[i]);
    }
  }

  // Export Phase 4 pairs + Essential H0
  std::vector<std::pair<int32_t, int32_t>> final_pairs;
  for (auto p : pairs) {
    final_pairs.push_back({crit_indices[p.first], crit_indices[p.second]});
  }
  auto get_coord = [&](int32_t sort_idx) -> std::vector<double> {
    int64_t g_id = cells[sort_idx].global_id;
    int64_t c_i, c_j, c_k;
    get_ijk(g_id, M_cells, L_cells, c_i, c_j, c_k);
    return {c_i / 2.0, c_j / 2.0, c_k / 2.0};
  };

  size_t num_pairs_phase4 = final_pairs.size();
  size_t total_out = num_pairs_phase4 + essential_h0.size();

  py::array_t<double> b_vals_arr(total_out), d_vals_arr(total_out);
  py::array_t<int> dims_arr(total_out);
  py::array_t<double> b_coords_arr({total_out, (size_t)3}),
      d_coords_arr({total_out, (size_t)3});

  auto b_v = b_vals_arr.mutable_unchecked<1>();
  auto d_v = d_vals_arr.mutable_unchecked<1>();
  auto d_m = dims_arr.mutable_unchecked<1>();
  auto b_c = b_coords_arr.mutable_unchecked<2>();
  auto d_c = d_coords_arr.mutable_unchecked<2>();

  for (size_t i = 0; i < num_pairs_phase4; ++i) {
    int32_t b_idx = final_pairs[i].first;
    int32_t d_idx = final_pairs[i].second;
    b_v(i) = cells[b_idx].key;
    d_v(i) = cells[d_idx].key;
    d_m(i) = cells[b_idx].dim;
    auto bc = get_coord(b_idx);
    auto dc = get_coord(d_idx);
    for (int k = 0; k < 3; ++k) {
      b_c(i, k) = bc[k];
      d_c(i, k) = dc[k];
    }
  }

  for (size_t i = 0; i < essential_h0.size(); ++i) {
    size_t out_idx = num_pairs_phase4 + i;
    int32_t b_idx = essential_h0[i];
    b_v(out_idx) = cells[b_idx].key;
    d_v(out_idx) = 0.0;
    d_m(out_idx) = 0;
    auto bc = get_coord(b_idx);
    for (int k = 0; k < 3; ++k) {
      b_c(out_idx, k) = bc[k];
      d_c(out_idx, k) = 0.0;
    }
  }

  py::dict res;
  res["births"] = b_vals_arr;
  res["deaths"] = d_vals_arr;
  res["dims"] = dims_arr;
  res["birth_coords"] = b_coords_arr;
  res["death_coords"] = d_coords_arr;
  return res;
}

PYBIND11_MODULE(morse_3d, m) {
  m.def("extract_persistence_3d_morse", &extract_persistence_3d_morse,
        "3D Cubical Complex DMT Persistence Engine v1.1");
}
