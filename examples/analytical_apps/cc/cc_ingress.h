
#ifndef ANALYTICAL_APPS_CC_CC_INGRESS_H_
#define ANALYTICAL_APPS_CC_CC_INGRESS_H_

#include "grape/app/ingress_app_base.h"
#include "grape/fragment/immutable_edgecut_fragment.h"

#include "z3++.h"
namespace grape {

template <typename FRAG_T, typename VALUE_T>
class CCIngress : public IterateKernel<FRAG_T, VALUE_T> {
 public:
  using fragment_t = FRAG_T;
  using vid_t = typename fragment_t::vid_t;
  using vertex_t = typename fragment_t::vertex_t;
  using value_t = VALUE_T;
  using adj_list_t = typename fragment_t::adj_list_t;
  value_t imax = std::numeric_limits<value_t>::max();
  value_t zero = 0;

  void init_c(const FRAG_T& frag, vertex_t v, value_t& delta,
              DenseVertexSet<vid_t>& modified) override {
    delta = frag.GetId(v);  // 获取顶点id
    modified.Insert(v);
  }

  void init_v(vertex_t v, value_t& value) override {
    value = imax;  // 初始化为最大值
  }

  bool aggregate(value_t& a, value_t b) override { return atomic_min(a, b); }

  z3::expr aggregate_z3(z3::expr a, z3::expr b) override { return ite(a < b, a, b); }

  bool update(value_t& a, value_t b) override { return aggregate(a, b); }

  bool accumulate(value_t& a, value_t b) override { return aggregate(a, b); }

  void priority(value_t& pri, const value_t& value, const value_t& delta) override {
    pri = std::abs(value - std::min(value, delta));
  }

  value_t generate(value_t v, value_t m, value_t w) override {
    return m * 0.85 * w;
  }

  z3::expr generate_z3(z3::solver& s, z3::expr x) override {
    z3::expr val = x;
    return val;
  }

  void g_function(const FRAG_T& frag, const vertex_t v, const value_t& value,
                  const value_t& delta, const adj_list_t& oes) override {
  }

  void g_function(const FRAG_T& frag, const vertex_t v, const value_t& value,
                  const value_t& delta, const adj_list_t& oes,
                  DenseVertexSet<vid_t>& modified) override {
    if (delta >= value)
      return;
    value_t outv = std::min(value, delta);

    for (auto e : oes) {
      auto dst = e.neighbor;

      if (this->accumulate_to(dst, outv)) {
        this->add_delta_dependency(frag.Vertex2Gid(v), dst);
        modified.Insert(dst);
      }
    }
  }

  value_t default_v() override { return imax; }  // 设置为delta中无作用的值

  value_t min_delta() override { return zero; }
};

}  // namespace grape

#endif  // ANALYTICAL_APPS_CC_CC_INGRESS_H_
