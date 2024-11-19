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

#include "paddle/cinn/common/ir_util.h"

#include <algorithm>
#include <stack>
#include <unordered_set>

#include "paddle/cinn/common/cas.h"
#include "paddle/cinn/common/simplify_corner_case.h"
#include "paddle/cinn/ir/ir_mutator.h"
#include "paddle/cinn/ir/ir_printer.h"
#include "paddle/cinn/ir/op/ir_operators.h"
#include "paddle/cinn/ir/utils/ir_compare.h"
#include "paddle/common/enforce.h"
namespace cinn {
namespace common {

namespace {

// ramp + scalar or broadcast
Expr RampRelatedMul(ir::Ramp *ramp, Expr other) {
  PADDLE_ENFORCE_EQ(
      other.type().ElementOf(),
      Int(32),
      ::common::errors::InvalidArgument("The type of other should be int32."));
  PADDLE_ENFORCE_EQ(ramp->base.type(),
                    Int(32),
                    ::common::errors::InvalidArgument(
                        "The type of ramp->base should be int32."));
  PADDLE_ENFORCE_EQ(ramp->stride.type(),
                    Int(32),
                    ::common::errors::InvalidArgument(
                        "The type of ramp->stride should be int32."));
  auto *other_broadcast = other.As<ir::Broadcast>();
  if (other_broadcast) {
    PADDLE_ENFORCE_EQ(ramp->lanes,
                      other_broadcast->lanes,
                      ::common::errors::InvalidArgument(
                          "The lanes of ramp and other should be equal."));
    other = other_broadcast->value;
  }
  return ir::Ramp::Make(ramp->base * other, ramp->stride * other, ramp->lanes);
}

Expr RampRelatedMul(ir::Broadcast *broadcast, Expr other) {
  PADDLE_ENFORCE_EQ(
      other.type().lanes(),
      1,
      ::common::errors::InvalidArgument("The lanes of other should be 1."));
  return ir::Broadcast::Make(broadcast->value * other, broadcast->lanes);
}
// ramp * ramp
Expr RampRelatedMul(ir::Ramp *ramp, ir::Ramp *other) {
  CINN_NOT_IMPLEMENTED
  return Expr();
}
// ramp + scalar
Expr RampRelatedAdd(ir::Ramp *ramp, Expr other) {
  PADDLE_ENFORCE_EQ(
      other.type().ElementOf(),
      Int(32),
      ::common::errors::InvalidArgument("The type of other should be int32."));

  auto *other_broadcast = other.As<ir::Broadcast>();
  if (other_broadcast) {
    PADDLE_ENFORCE_EQ(ramp->lanes,
                      other_broadcast->lanes,
                      ::common::errors::InvalidArgument(
                          "The lanes of ramp and other should be equal."));
    other = other_broadcast->value;
  }
  return ir::Ramp::Make(ramp->base + other, ramp->stride, ramp->lanes);
}
Expr RampRelatedAdd(ir::Broadcast *broadcast, Expr other) {
  PADDLE_ENFORCE_EQ(
      other.type().lanes(),
      1,
      ::common::errors::InvalidArgument("The lanes of other should be 1."));
  return ir::Broadcast::Make(broadcast->value + other, broadcast->lanes);
}
// ramp + ramp
Expr RampRelatedAdd(ir::Ramp *ramp, ir::Ramp *other) {
  PADDLE_ENFORCE_NOT_NULL(
      ramp,
      ::common::errors::InvalidArgument("Ramp pointer should not be null."));
  PADDLE_ENFORCE_NOT_NULL(other,
                          ::common::errors::InvalidArgument(
                              "Other ramp pointer should not be null."));
  if (ramp->lanes == other->lanes) {
    Expr base_add = cinn::common::AutoSimplify(ramp->base + other->base);
    Expr stride_add = cinn::common::AutoSimplify(ramp->stride + other->stride);
    VLOG(2) << base_add;
    VLOG(2) << stride_add;
    return ir::Ramp::Make(base_add, stride_add, ramp->lanes);
  }
  CINN_NOT_IMPLEMENTED
  return Expr();
}

Expr RampRelatedAdd(Expr a, Expr b) {
  auto *a_ramp = a.As<ir::Ramp>();
  auto *b_ramp = b.As<ir::Ramp>();
  auto *a_broadcast = a.As<ir::Broadcast>();
  auto *b_broadcast = b.As<ir::Broadcast>();
  if (a_ramp && !b_ramp && (b->type().lanes() == 1 || b_broadcast)) {
    return RampRelatedAdd(a_ramp, b);
  } else if (!a_ramp && b_ramp && (a->type().lanes() == 1 || a_broadcast)) {
    return RampRelatedAdd(b_ramp, a);
  } else if (!a_ramp && !b_ramp && !a->type().is_vector() &&
             !b->type().is_vector()) {
    return a + b;
  } else if (a_ramp && b_ramp) {  // a_ramp && b_ramp
    return RampRelatedAdd(a_ramp, b_ramp);
  } else if (a_broadcast && !b_broadcast) {
    return RampRelatedAdd(a_broadcast, b);
  } else if (!a_broadcast && b_broadcast) {
    return RampRelatedAdd(b_broadcast, a);
  } else if (a_broadcast && b_broadcast) {
    PADDLE_ENFORCE_EQ(
        a_broadcast->lanes,
        b_broadcast->lanes,
        ::common::errors::InvalidArgument(
            "The lanes of a_broadcast and b_broadcast should be equal."));
    return ir::Broadcast::Make(a_broadcast->value + b_broadcast->value,
                               a_broadcast->lanes);
  } else {
    CINN_NOT_IMPLEMENTED
  }
}

Expr RampRelatedMul(Expr a, Expr b) {
  auto *a_ramp = a.As<ir::Ramp>();
  auto *b_ramp = b.As<ir::Ramp>();
  auto *a_broadcast = a.As<ir::Broadcast>();
  auto *b_broadcast = b.As<ir::Broadcast>();
  if (a_ramp && !b_ramp && (!b->type().is_vector() || b_broadcast)) {
    return RampRelatedMul(a_ramp, b);
  } else if (!a_ramp && b_ramp && (a->type().is_vector() || a_broadcast)) {
    return RampRelatedMul(b_ramp, a);
  } else if (!a_ramp && !b_ramp && !a->type().is_vector() &&
             !b->type().is_vector()) {
    return a * b;
  } else if (a_ramp && b_ramp) {  // a_ramp && b_ramp
    return RampRelatedMul(a_ramp, b_ramp);
  } else if (a_broadcast && !b_broadcast) {
    return RampRelatedMul(a_broadcast, b);
  } else if (!a_broadcast && b_broadcast) {
    return RampRelatedMul(b_broadcast, a);
  } else if (a_broadcast && b_broadcast) {
    PADDLE_ENFORCE_EQ(
        a_broadcast->lanes,
        b_broadcast->lanes,
        ::common::errors::InvalidArgument(
            "The lanes of a_broadcast and b_broadcast should be equal."));
    return ir::Broadcast::Make(a_broadcast->value * b_broadcast->value,
                               a_broadcast->lanes);
  } else {
    VLOG(3) << "a,b: " << a << " " << b;
    CINN_NOT_IMPLEMENTED
  }
}

}  // namespace

static void MergeMulModInsertElements(
    const std::vector<ir::IndexExpr> &eles,
    std::list<ir::IndexExpr> *mult_exprs,
    std::list<std::pair<ir::IndexExpr, ir::IndexExpr>> *mod_exprs,
    ir::IndexExpr *no_opt_sum,
    bool *has_mult,
    bool *has_mod) {
  *has_mult = false;
  *has_mod = false;
  for (const ir::IndexExpr ele : eles) {
    auto mod_ptr = ele.As<ir::Mod>();
    auto mult_ptr = ele.As<ir::Mul>();
    if (mod_ptr) {
      *has_mod = true;
      mod_exprs->emplace_back(
          std::make_pair(std::move(mod_ptr->a().as_index()),
                         std::move(mod_ptr->b().as_index())));
    } else if (mult_ptr) {
      *has_mult = true;
      mult_exprs->emplace_back(ele);
    } else {
      *no_opt_sum = no_opt_sum->get() ? *no_opt_sum + ele : ele;
    }
  }
}

static std::optional<ir::IndexExpr> MergeMulModInner(
    SymbolicExprAnalyzer *analyzer,
    const ir::IndexExpr &mult_expr,
    const ir::IndexExpr &mod_l_expr,
    const ir::IndexExpr &mod_r_expr) {
  const ir::Mul *mult_ptr = mult_expr.As<ir::Mul>();
  if (!mult_ptr) return std::nullopt;
  ir::IndexExpr mult_outer = mult_ptr->b();
  ir::IndexExpr inner = mult_ptr->a().as_index();

  while (true) {
    mult_ptr = inner.As<ir::Mul>();
    if (mult_ptr) {
      inner = mult_ptr->a().as_index();
      mult_outer = mult_ptr->b().as_index() * mult_outer.as_index();
    } else {
      break;
    }
  }

  ir::IndexExpr search_ptr = inner;
  ir::IndexExpr mult_inner;  // The inner multiplication factor
  ir::IndexExpr no_opt_sum;  // Sum of the exprs that cannot be optimized

  while (true) {
    auto inner_div_ptr = search_ptr.As<ir::Div>();
    auto inner_mult_ptr = search_ptr.As<ir::Mul>();
    auto inner_add_ptr = search_ptr.As<ir::Add>();
    if (!inner_div_ptr && !inner_mult_ptr && !inner_add_ptr) {
      return std::nullopt;
    } else if (inner_div_ptr) {
      ir::IndexExpr overall_mult =
          mult_inner.get() ? mult_inner * mult_outer : mult_outer;
      if (overall_mult == inner_div_ptr->b().as_index() &&
          overall_mult == mod_r_expr &&
          ProveDivisible(inner_div_ptr->a().as_index() - mod_l_expr,
                         mod_r_expr)) {
        // Found!
        return no_opt_sum.get()
                   ? no_opt_sum * mult_outer + inner_div_ptr->a().as_index()
                   : inner_div_ptr->a().as_index();
      } else {
        return std::nullopt;
      }
    } else if (inner_mult_ptr) {
      mult_inner = mult_inner.get()
                       ? inner_mult_ptr->b().as_index() * mult_inner
                       : inner_mult_ptr->b().as_index();
      search_ptr = inner_mult_ptr->a().as_index();
    } else if (inner_add_ptr) {
      if (mult_inner.get()) {
        return std::nullopt;
      }
      auto lhs = inner_add_ptr->a().as_index();
      auto rhs = inner_add_ptr->b().as_index();
      if (inner_add_ptr->b().as_index().is_constant()) {
        std::swap(lhs, rhs);
      } else if (inner_add_ptr->b().as_index().length() < mod_r_expr.length()) {
        std::swap(lhs, rhs);
      }
      no_opt_sum = no_opt_sum.get() ? no_opt_sum + lhs : lhs;
      search_ptr = rhs;
    } else {
      break;
    }
  }
  return std::nullopt;
}

ir::IndexExpr MergeMulMod(SymbolicExprAnalyzer *analyzer,
                          const ir::IndexExpr &base) {
  ir::IndexExpr simplified_base = base.as_index().Normalize();
  std::vector<ir::IndexExpr> eles = GetFlattenExprs<ir::Add>(simplified_base);
  std::list<ir::IndexExpr> mult_exprs;
  std::list<std::pair<ir::IndexExpr, ir::IndexExpr>> mod_exprs;
  ir::IndexExpr no_opt_sum;
  bool has_mult;
  bool has_mod;
  MergeMulModInsertElements(
      eles, &mult_exprs, &mod_exprs, &no_opt_sum, &has_mult, &has_mod);
  bool find_opt = false;
  std::list<std::pair<ir::IndexExpr, ir::IndexExpr>>::iterator search_mod_it =
      mod_exprs.begin();

  while (search_mod_it != mod_exprs.end()) {
    std::list<ir::IndexExpr>::iterator mult_it = mult_exprs.begin();
    bool inner_find_opt = false;
    while (mult_it != mult_exprs.end()) {
      auto ret = MergeMulModInner(
          analyzer, *mult_it, search_mod_it->first, search_mod_it->second);
      if (ret.has_value()) {
        inner_find_opt = true;
        auto temp_mod_it = search_mod_it;
        ++search_mod_it;
        mod_exprs.erase(temp_mod_it);
        mult_exprs.erase(mult_it);
        std::vector<ir::IndexExpr> ret_eles =
            GetFlattenExprs<ir::Add>(ret.value());
        MergeMulModInsertElements(ret_eles,
                                  &mult_exprs,
                                  &mod_exprs,
                                  &no_opt_sum,
                                  &has_mult,
                                  &has_mod);
        if (has_mult) {
          search_mod_it = mod_exprs.begin();
        } else if (has_mod && search_mod_it == mod_exprs.end()) {
          search_mod_it--;
        }
        break;
      } else {
        ++mult_it;
      }
    }
    find_opt = find_opt || inner_find_opt;
    if (!inner_find_opt) {
      ++search_mod_it;
    }
  }
  if (!find_opt) {
    return simplified_base;
  }
  for (std::list<ir::IndexExpr>::iterator it = mult_exprs.begin();
       it != mult_exprs.end();
       ++it) {
    no_opt_sum = no_opt_sum.get() ? no_opt_sum + *it : *it;
  }
  for (std::list<std::pair<ir::IndexExpr, ir::IndexExpr>>::iterator it =
           mod_exprs.begin();
       it != mod_exprs.end();
       ++it) {
    no_opt_sum = no_opt_sum.get() ? no_opt_sum + it->first % it->second
                                  : it->first % it->second;
  }
  return no_opt_sum;
}

Expr IndiceToAbsOffset(const std::vector<Expr> &shape,
                       const std::vector<Expr> &indices) {
  VLOG(3) << "Begin IndiceToAbsOffset";
  VLOG(3) << "shape is : " << utils::Join(shape, ",");
  VLOG(3) << "indices is : " << utils::Join(indices, ",");
  PADDLE_ENFORCE_LE(shape.size(),
                    indices.size(),
                    ::common::errors::InvalidArgument(
                        "The size of shape should be less than or "
                        "equal to the size of indices."));
  Expr res;
  ir::TryElevateInt32ToInt64(shape);
  common::cas_intervals_t var_intervals =
      common::CollectVarIntervalsOfExprs(indices);
  common::SymbolicExprAnalyzer analyzer{var_intervals};

  for (int32_t i = 0; i < shape.size(); i++) {
    PADDLE_ENFORCE_EQ(
        shape[i].type() == Int(64) || shape[i].type() == Int(32),
        true,
        ::common::errors::InvalidArgument(
            "The shape data type currently supports only int32 or int64, but "
            "the current data type of shape[{}] is {}",
            i,
            shape[i].type()));
    Expr indice_cast = indices[i];
    optim::SimplifyCast(&indice_cast);
    if (res.defined()) {
      res = RampRelatedAdd(RampRelatedMul(res, shape[i]), indice_cast);
      if (res.is_index()) {
        res = res.as_index().Normalize();
      }
    } else {
      res = indice_cast;
    }
    if (i > 0) {
      if (res.is_index()) {
        res = MergeMulMod(&analyzer, res.as_index()).as_index().Normalize();
      }
    }
  }

  return res;
}

Expr IndiceToAbsOffset(const std::vector<int> &shape,
                       const std::vector<Expr> &indices) {
  std::vector<Expr> shape_;
  for (int v : shape) shape_.push_back(Expr(v));
  return IndiceToAbsOffset(shape, indices);
}

Expr PrecedingAxisToAbsOffset(const std::vector<Expr> &shape,
                              int preceding_n_axis) {
  std::vector<Expr> indices;
  for (int i = 0; i < preceding_n_axis; i++) indices.push_back(shape[i]);
  return IndiceToAbsOffset(shape, indices);
}

namespace {

class SubstituteMutator : ir::IRMutator<ir::Expr *> {
 public:
  explicit SubstituteMutator(const std::map<const ir::_Var_ *, Expr> &var_map) {
    for (auto &item : var_map) {
      var_map_[item.first->name] = item.second;
    }
  }

  void operator()(ir::Expr *expr) { Visit(expr); }

 private:
  void Visit(Expr *expr) { ir::IRMutator<>::Visit(expr, expr); }

  void Visit(const ir::_Var_ *op, ir::Expr *expr) override {
    auto it = var_map_.find(op->name);
    if (it == var_map_.end()) return;
    *expr = it->second;
  }

  Expr *expr_{};
  std::map<std::string, Expr> var_map_;
};

}  // namespace

void Substitute(Expr *expr, const std::map<const ir::_Var_ *, Expr> &var_map) {
  SubstituteMutator mutator(var_map);
  mutator(expr);
}

bool is_zero(Expr v) {
  v = AutoSimplify(v);
  auto *int_n = v.As<ir::IntImm>();
  auto *float_n = v.As<ir::FloatImm>();

  if (int_n) return int_n->value == 0;
  if (float_n) return float_n->value == 0.f;
  return false;
}

Expr CastIfNeeded(Expr body, Type type) {
  if (body.type() == type) return body;
  return ir::Cast::Make(type, body);
}

bool MathEqual(const Expr &a, const Expr &b) {
  auto c = a - b;
  c = AutoSimplify(c);
  return is_zero(c);
}

Expr select(Expr cond, Expr true_value, Expr false_value) {
  return ir::Select::Make(cond, true_value, false_value);
}

Expr and_all(const std::vector<Expr> &conds) {
  PADDLE_ENFORCE_NE(conds.empty(),
                    true,
                    ::common::errors::InvalidArgument(
                        "The conditions vector should not be empty."));
  Expr res = conds.front();
  for (int i = 1; i < conds.size(); i++) {
    res = ir::And::Make(res, conds[i]);
  }
  return res;
}

Expr or_all(const std::vector<Expr> &conds) {
  PADDLE_ENFORCE_NE(conds.empty(),
                    true,
                    ::common::errors::InvalidArgument(
                        "The conditions vector should not be empty."));
  Expr res = conds.front();
  for (int i = 1; i < conds.size(); i++) {
    res = ir::Or::Make(res, conds[i]);
  }
  return res;
}

void CheckTensorUniqueInExpr(Expr expr) {
  auto tensor_uniq = ir::ir_utils::CollectIRNodes(
      expr, [](const Expr *x) { return x->as_tensor(); });
  absl::flat_hash_map<std::string, const ir::_Tensor_ *> tensor_names;
  for (auto &t : tensor_uniq) {
    auto *tp = t.as_tensor();
    if (!tensor_names.count(tp->name)) {
      tensor_names[tp->name] = tp;
    } else {
      PADDLE_ENFORCE_EQ(
          tensor_names[tp->name],
          tp,
          ::common::errors::InvalidArgument(
              "Found tensor not unique, The original express is %d .", expr));
    }
  }
}

Expr cast(Expr e, Type type) {
  if (e.is_constant()) {
    if (type.is_bool()) {
      return Expr(static_cast<bool>(e.get_constant()));
    } else if (type.is_int(8)) {
      return Expr(static_cast<int8_t>(e.get_constant()));
    } else if (type.is_int(16)) {
      return Expr(static_cast<int16_t>(e.get_constant()));
    } else if (type.is_int(32)) {
      return Expr(static_cast<int32_t>(e.get_constant()));
    } else if (type.is_int(64)) {
      return Expr(static_cast<int64_t>(e.get_constant()));
    } else if (type.is_uint(8)) {
      return Expr(static_cast<uint8_t>(e.get_constant()));
    } else if (type.is_uint(16)) {
      return Expr(static_cast<uint16_t>(e.get_constant()));
    } else if (type.is_uint(32)) {
      return Expr(static_cast<uint32_t>(e.get_constant()));
    } else if (type.is_uint(64)) {
      return Expr(static_cast<uint64_t>(e.get_constant()));
    } else if (type.is_float(32)) {
      return Expr(static_cast<float>(e.get_constant()));
    } else if (type.is_float(64)) {
      return Expr(static_cast<double>(e.get_constant()));
    } else if (type.is_bfloat16()) {
      return Expr(static_cast<cinn::common::bfloat16>(e.get_constant()));
    } else if (type.is_float16()) {
      return Expr(static_cast<cinn::common::float16>(e.get_constant()));
    } else {
      CINN_NOT_IMPLEMENTED
    }
  }

  return ir::Cast::Make(type, e);
}

std::vector<std::string> GatherItersToTensorProducer(
    const std::string &target_tensor_name, Expr *expr) {
  struct Visitor : public ir::IRMutator<> {
    std::vector<std::string> iters;
    const std::string &target_tensor_name;

    explicit Visitor(const std::string &target_tensor_name)
        : target_tensor_name(target_tensor_name) {}

    std::vector<std::string> operator()(Expr *expr) {
      ir::IRMutator<>::Visit(expr, expr);
      return iters;
    }

    void Visit(const ir::Store *op, Expr *expr) {
      if (op->tensor.as_tensor()->name == target_tensor_name) {
        PADDLE_ENFORCE_EQ(iters.empty(),
                          true,
                          ::common::errors::InvalidArgument(
                              "The iterators vector should be empty."));
        for (auto &e : for_stack) {
          auto *for_n = e->As<ir::For>();
          auto *polyfor_n = e->As<ir::PolyFor>();
          if (for_n) {
            iters.push_back(for_n->loop_var->name);
          } else {
            iters.push_back(polyfor_n->iterator->name);
          }
        }
      }
    }

    void Visit(const ir::For *op, Expr *expr) {
      for_stack.push_back(expr);
      ir::IRMutator<>::Visit(op, expr);
      for_stack.pop_back();
    }
    void Visit(const ir::PolyFor *op, Expr *expr) {
      for_stack.push_back(expr);
      ir::IRMutator<>::Visit(op, expr);
      for_stack.pop_back();
    }

    std::vector<Expr *> for_stack;
  };

  return Visitor(target_tensor_name)(expr);
}

std::vector<Expr *> GetForloopStackToStore(Expr *expr,
                                           const std::string &tensor_name) {
  VLOG(4) << "search store " << tensor_name << " in expr:\n";
  VLOG(4) << *expr;
  struct Mutator : public ir::IRMutator<> {
    std::vector<Expr *> forloop_stack;
    bool found{false};

    std::string tensor_name;

    explicit Mutator(const std::string &tensor_name)
        : tensor_name(tensor_name) {}

    std::vector<Expr *> operator()(Expr *expr) {
      ir::IRMutator<>::Visit(expr, expr);
      return forloop_stack;
    }

    void Visit(const ir::For *op, Expr *expr) {
      auto *node = expr->As<ir::For>();
      forloop_stack.push_back(expr);
      ir::IRMutator<>::Visit(&node->body, &node->body);
      if (!found) forloop_stack.pop_back();
    }

    void Visit(const ir::PolyFor *op, Expr *expr) {
      auto *node = expr->As<ir::PolyFor>();
      forloop_stack.push_back(expr);
      ir::IRMutator<>::Visit(&node->body, &node->body);
      if (!found) forloop_stack.pop_back();
    }

    void Visit(const ir::Store *op, Expr *expr) {
      found = op->tensor.as_tensor()->name == tensor_name;
    }
  };

  return Mutator(tensor_name)(expr);
}

Expr max(Expr a, Expr b) {
  PADDLE_ENFORCE_EQ(a.type(),
                    b.type(),
                    ::common::errors::InvalidArgument(
                        "The type of a and b should be equal."));
  return ir::Max::Make(a, b);
}

Expr min(Expr a, Expr b) {
  PADDLE_ENFORCE_EQ(a.type(),
                    b.type(),
                    ::common::errors::InvalidArgument(
                        "The type of a and b should be equal."));
  return ir::Min::Make(a, b);
}

bool ComparePriority(const ir::IndexExpr &lhs, const ir::IndexExpr &rhs) {
  if (lhs.node_type() == ir::IrNodeTy::IntImm &&
      rhs.node_type() != ir::IrNodeTy::IntImm)
    return false;
  if (rhs.node_type() == ir::IrNodeTy::IntImm &&
      lhs.node_type() != ir::IrNodeTy::IntImm)
    return true;
  if (auto lhsVar = lhs.As<ir::_Var_>())
    if (auto rhsVar = rhs.As<ir::_Var_>())
      return std::make_tuple(lhsVar->name.length(), lhsVar->name) <=
             std::make_tuple(rhsVar->name.length(), rhsVar->name);
  auto lhsLen = lhs.length();
  auto rhsLen = rhs.length();
  if (lhsLen < rhsLen) return false;
  // Add < Mul < Div < Mod.
  else if (lhsLen == rhsLen)
    return lhs.node_type() <= rhs.node_type();
  else
    return true;
}

bool IsSumPartialBySymbol(const ir::IndexExpr &expr,
                          const ir::IndexExpr &symbol) {
  // TODO(liujinnan): Check Ty
  switch (expr.node_type()) {
    case ir::IrNodeTy::IntImm: {
      return false;
    }
    case ir::IrNodeTy::_Var_:
      return expr == symbol;
    case ir::IrNodeTy::Add:
      return IsSumPartialBySymbol(expr->operand(0).as_index(), symbol) ||
             IsSumPartialBySymbol(expr->operand(1).as_index(), symbol);
    case ir::IrNodeTy::Mul: {
      if (expr->operand(1).is_constant() &&
          expr->operand(1).get_constant() == -1)
        return IsSumPartialBySymbol(expr->operand(0).as_index(), symbol);
      else
        return expr->operand(0).as_index() == symbol ||
               expr->operand(1).as_index() == symbol;
    }

    case ir::IrNodeTy::Div: {
      return IsSumPartialBySymbol(expr->operand(0).as_index(), symbol);
    }
    case ir::IrNodeTy::Mod:
      return false;
    default:
      PADDLE_THROW(::common::errors::InvalidArgument(
          "Unsupported type of expr in IsSumPartialBySymbol which is: %s",
          expr));
  }
}

bool IsDivisiblieBySymbol(const ir::IndexExpr &expr,
                          const ir::IndexExpr &symbol,
                          const ir::IrNodeTy &ty) {
  // TODO(liujinnan): Check Ty
  switch (expr.node_type()) {
    case ir::IrNodeTy::IntImm: {
      auto imm = expr.As<ir::IntImm>();
      return imm->value == 0;
    }
    case ir::IrNodeTy::_Var_:
      return expr == symbol;
    case ir::IrNodeTy::Add:
      return IsDivisiblieBySymbol(expr->operand(0).as_index(), symbol, ty) &&
             IsDivisiblieBySymbol(expr->operand(1).as_index(), symbol, ty);
    case ir::IrNodeTy::Mul:
      return IsDivisiblieBySymbol(expr->operand(0).as_index(), symbol, ty) ||
             IsDivisiblieBySymbol(expr->operand(1).as_index(), symbol, ty);
    case ir::IrNodeTy::Mod:
      // Because S0 % 3 + S0 % 5 is not divisiblie by S0, so we push
      // `expr.node_type()` into third parameter.
      return IsDivisiblieBySymbol(
                 expr->operand(0).as_index(), symbol, expr.node_type()) &&
             IsDivisiblieBySymbol(
                 expr->operand(1).as_index(), symbol, expr.node_type());
    case ir::IrNodeTy::Div: {
      if (ty != expr.node_type()) return false;
      return IsDivisiblieBySymbol(
          expr->operand(0).as_index(), symbol, expr.node_type());
    }
    default:
      PADDLE_THROW(::common::errors::InvalidArgument(
          "Unsupported type of expr in IsDivisiblieBySymbol which is: %s",
          expr));
  }
}

bool ProveDivisible(const ir::IndexExpr &lhs, const ir::IndexExpr &rhs) {
  if (IsZero(lhs % rhs)) return true;
  // remove AutoSimplify later.
  if (IsZero(AutoSimplify(lhs % rhs))) return true;
  return false;
}

bool IsNegatedIndexExpr(const ir::IndexExpr &candidate,
                        ir::IndexExpr &expr) {  // NOLINT
  if (auto mul = candidate.As<ir::Mul>()) {
    if (mul->b().is_constant() && mul->b().get_constant() == -1) {
      expr = mul->a();
      return true;
    }
  }
  return false;
}
}  // namespace common
}  // namespace cinn
