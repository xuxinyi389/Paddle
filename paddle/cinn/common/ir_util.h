// Copyright (c) 2021 CINN Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once
#include <absl/container/flat_hash_map.h>

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "paddle/cinn/common/bfloat16.h"
#include "paddle/cinn/common/float16.h"
#include "paddle/cinn/common/integer_set.h"
#include "paddle/cinn/ir/ir.h"

namespace cinn {
namespace common {

Expr IndiceToAbsOffset(const std::vector<Expr> &shape,
                       const std::vector<Expr> &indices);
Expr IndiceToAbsOffset(const std::vector<int> &shape,
                       const std::vector<Expr> &indices);

Expr PrecedingAxisToAbsOffset(const std::vector<Expr> &shape,
                              int preceding_n_axis);

Expr CastIfNeeded(Expr body, Type type);

ir::IndexExpr MergeMulMod(SymbolicExprAnalyzer *analyzer,
                          const ir::IndexExpr &base);

//! Substitute vars to other expressions.
//! @param expr The expression to do modification.
//! @param var_map The map from variables to the target expressions.
void Substitute(Expr *expr, const std::map<const ir::_Var_ *, Expr> &var_map);

//! Get a stack of forloops(For and PolyFor nodes) to a Store node target to \p
//! tensor_name
std::vector<Expr *> GetForloopStackToStore(Expr *expr,
                                           const std::string &tensor_name);

// make const
// @{
inline Expr make_const(int32_t x) { return Expr(static_cast<int32_t>(x)); }
inline Expr make_const(int64_t x) { return Expr(static_cast<int64_t>(x)); }
inline Expr make_const(bfloat16 x) { return Expr(static_cast<bfloat16>(x)); }
inline Expr make_const(float16 x) { return Expr(static_cast<float16>(x)); }
inline Expr make_const(float x) { return Expr(static_cast<float>(x)); }
inline Expr make_const(double x) { return Expr(static_cast<double>(x)); }
inline Expr make_const(bool x) { return Expr(static_cast<bool>(x)); }
// @}

//! maker for some general consts.
// @{
template <typename T = int32_t>
inline Expr make_zero() {
  return make_const(static_cast<T>(0));
}
template <typename T = int32_t>
inline Expr make_one() {
  return make_const(static_cast<T>(1));
}
inline Expr make_bool(bool x) {
  return cinn::common::make_shared<ir::UIntImm>(Bool(), x);
}
inline Expr make_bool(bool x, int lanes) {
  return cinn::common::make_shared<ir::UIntImm>(Bool(lanes), x);
}
// @}

/**
 * \brief Check all the tensors are unique in an expression.
 */
void CheckTensorUniqueInExpr(Expr expr);

std::vector<std::string> GatherItersToTensorProducer(
    const std::string &target_tensor_name, Expr *expr);

bool is_zero(Expr v);

bool MathEqual(const Expr &a, const Expr &b);

//! helper function to get a ir::Select node.
Expr select(Expr cond, Expr true_value, Expr false_value);

//! helper function to get the And of all the conditions.
Expr and_all(const std::vector<Expr> &conds);

//! helper function to get the Or of all the conditions.
Expr or_any(const std::vector<Expr> &conds);

//! Cast the expression \p e to type \type.
Expr cast(Expr e, Type type);

Expr max(Expr a, Expr b);

Expr min(Expr a, Expr b);

template <typename T>
Expr make_const(Type t, T v) {
  if (t.is_vector()) {
    if (t.is_int()) {
      return ir::Broadcast::Make(
          make_shared<ir::IntImm>(t.ElementOf(), static_cast<int64_t>(v)),
          t.lanes());
    } else if (t.is_uint()) {
      return ir::Broadcast::Make(
          make_shared<ir::UIntImm>(t.ElementOf(), static_cast<uint64_t>(v)),
          t.lanes());
    } else if (t.is_float()) {
      return ir::Broadcast::Make(
          make_shared<ir::FloatImm>(t.ElementOf(), static_cast<double>(v)),
          t.lanes());
    } else if (t.is_bool()) {
      return ir::Broadcast::Make(
          make_shared<ir::UIntImm>(t.ElementOf(), static_cast<bool>(v)),
          t.lanes());
    } else {
      CINN_NOT_IMPLEMENTED
    }
  } else {
    if (t.is_int()) {
      return make_shared<ir::IntImm>(t, static_cast<int64_t>(v));
    } else if (t.is_uint()) {
      return make_shared<ir::UIntImm>(t, static_cast<uint64_t>(v));
    } else if (t.is_float()) {
      return make_shared<ir::FloatImm>(t, static_cast<double>(v));
    } else if (t.is_bool()) {
      return make_shared<ir::UIntImm>(t, static_cast<bool>(v));
    } else {
      CINN_NOT_IMPLEMENTED
    }
  }
  return Expr();
}

template <typename FuncOp>
Expr FoldExpr(FuncOp func_op, const std::vector<Expr> &values) {
  Expr init_value;
  for (const Expr &val : values) {
    if (!init_value.defined()) {
      init_value = val;
    } else {
      init_value = func_op(val, init_value);
    }
  }
  return init_value;
}

inline bool IsIterExpr(const Expr &a, const Expr &b) {
  return a.As<ir::IterSplit>() || a.As<ir::IterSum>() ||
         b.As<ir::IterSplit>() || b.As<ir::IterSum>();
}

inline bool IsOne(const Expr &expr) {
  if (expr.is_constant() && expr.get_constant() == 1) {
    return true;
  }
  return false;
}
inline bool IsZero(const Expr &expr) {
  if (expr.is_constant() && expr.get_constant() == 0) {
    return true;
  }
  return false;
}

/*!
 * \brief Apply func `fleaf` into each leaf node of `expr`.
 * which leaf node is the most outside node that has TNode type.
 * \param expr The expression to be applied.
 * \param fleaf The function to be applied.
 */
template <typename TNode, typename FLeaf>
inline void UnpackReduction(const ir::IndexExpr &expr, FLeaf fleaf) {
  if (const TNode *node = expr.As<TNode>()) {
    UnpackReduction<TNode, FLeaf>(node->a(), fleaf);
    UnpackReduction<TNode, FLeaf>(node->b(), fleaf);
  } else {
    fleaf(expr);
  }
}

/*!
 * \brief Flattern the expression into a vector of expressions splited by `Add`
 * or `Mul`.
 *
 * For example (Add):
 * 1. `S0 + S1` ==> {S0, S1}
 * 2. `S0 + S1 * S2` ==> {S0, S1 * S2}
 * 3. `S0 + S1 * (S2 + S3)` ==> {S0, S1 * (S2 + S3)}
 *
 * \param lhs The left hand side expression to be compared.
 * \param rhs The right hand side expression to be compared.
 * \return A boolean value indicating whether the priority of `lhs` is higher
 * than `rhs`.
 */
template <typename T>
inline std::vector<ir::IndexExpr> GetFlattenExprs(const ir::IndexExpr &expr) {
  std::vector<ir::IndexExpr> result;
  auto fcollect = [&](ir::IndexExpr val) { result.push_back(val); };
  UnpackReduction<T>(expr, fcollect);
  return result;
}

/*!
 * \brief Compare the priority of the two expressions. this func follows the
 * above rules:
 * 1. if lhs = var, rhs = const,    return true;
 * 2. if lhs = const, rhs = var,    return false;
 * 3. if lhs = var, rhs = var,      return lhs_var_name <= lhs_var_name;
 * 4. if lhs.length > rhs.length,   return true;
 * 5. if lhs.length == rhs.length,  return lhs_type <= rhs_type; (Add < Mul <
 * Div < Mod)
 * 6. if lhs.length < rhs.length    return false;
 *
 * For example:
 * 1. `ComparePriority(S0, 2)` return true;
 * 2. `ComparePriority(S0, S0)` return true;
 * 2. `ComparePriority(S0, S1)` return false;
 * 3. `ComparePriority(S0, S1 + 1)` return false;
 * 4. `ComparePriority(S0 % 2, S1 + 1)` return false;
 *
 * \param lhs The left hand side expression to be compared.
 * \param rhs The right hand side expression to be compared.
 * \return A boolean value indicating whether the priority of `lhs` is higher
 * than `rhs`.
 */
bool ComparePriority(const ir::IndexExpr &lhs, const ir::IndexExpr &rhs);

/*!
 * \brief Determines whether there are sub-parts in the `expr` that can be
 * simplified by `Add` operation with the input `symbol`. If true is returned,
 * the operation will be attempted on each subpart in outter
 * `SimplifySymbolicAdd` function.
 *
 * For example:
 * 1. `IsSumPartialBySymbol(5, S0)` return false;
 * 2. `IsSumPartialBySymbol(S0, S0)` return true;
 * 3. `IsSumPartialBySymbol(S0 + S1, S1)` return true;
 * 4. `IsSumPartialBySymbol(S0 * 5 + S1, S0)` return true;
 * 5. `IsSumPartialBySymbol(S0 / 3, S0)` return true;
 * 6. `IsSumPartialBySymbol(S0 / 3 + S1, S0)` return true;
 * 7. `IsSumPartialBySymbol(S0 % 3, S0)` return false;
 *
 * \param expr The expression to be checked.
 * \param symbol  The symbol to be checked.
 * \return True means there are sub-parts in the `expr` that can be simplified.
 */
bool IsSumPartialBySymbol(const ir::IndexExpr &expr,
                          const ir::IndexExpr &symbol);

/*!
 * \brief Determines whether there are sub-parts in the `expr` that can be
 * simplified by `Div` operation with the input `symbol`. If true is returned,
 * the operation will be attempted on each subpart in outter
 * `SimplifySymbolicDivide` function.
 *
 * For example:
 * 1. `IsDivisiblieBySymbol(5, S0, div)` return false;
 * 2. `IsDivisiblieBySymbol(S0, S0, div)` return true;
 * 3. `IsDivisiblieBySymbol(S0 + S1, S1, div)` return false;
 * 4. `IsDivisiblieBySymbol(S0 * 5 + S1 * S2, S0, div)` return true;
 * 5. `IsDivisiblieBySymbol(S0 / 3, S0, div)` return true;
 * 6. `IsDivisiblieBySymbol(S0 * 4 / 3, S0, div)` return true;
 * 7. `IsDivisiblieBySymbol(S0 % 3, S0, div)` return false;
 * 8. `IsDivisiblieBySymbol(S0 / 3, S0, mod)` return false;
 *
 * \param expr The expression to be checked.
 * \param symbol  The symbol to be checked.
 * \param ty ty is `Mod` or `Div`.
 * \return True means there are sub-parts in the `expr` that can be simplified.
 * \note this func dont deal the corner case, please use `ProveDivisible` for
 * exact result. e.g. `IsDivisiblieBySymbol(f % S0 - f, S0, div)` is false
 */
bool IsDivisiblieBySymbol(const ir::IndexExpr &expr,
                          const ir::IndexExpr &symbol,
                          const ir::IrNodeTy &ty);

/*!
 * \brief Determine whether `lhs` is divisible by `rhs`, regardless of whether
 * `rhs` is a constant or a symbol.
 * \param lhs lhs is dividend.
 * \param rhs rhs is divisor.
 * \return A boolean value indicating whether the `lhs` is divisible by `rhs`
 */
bool ProveDivisible(const ir::IndexExpr &lhs, const ir::IndexExpr &rhs);

/*!
 * \brief Judge whether `candidate` is a negated index expression.
 * \param lhs The expression to be checked.
 * \param rhs The positive part
 * \return A boolean value indicating whether `candidate` is negative.
 */
bool IsNegatedIndexExpr(const ir::IndexExpr &candidate,
                        ir::IndexExpr &expr);  // NOLINT

}  // namespace common
}  // namespace cinn
