
#ifndef ANALYTICAL_APPS_SSSP_SSSP_INGRESS_H_
#define ANALYTICAL_APPS_SSSP_SSSP_INGRESS_H_

#include "flags.h"
#include "grape/app/ingress_app_base.h"
#include "grape/fragment/immutable_edgecut_fragment.h"

#include "z3++.h"

namespace grape {

template <typename FRAG_T, typename VALUE_T>
class SSSPIngress : public IterateKernel<FRAG_T, VALUE_T> {
 public:
  using fragment_t = FRAG_T;
  using oid_t = typename fragment_t::oid_t;
  using vid_t = typename fragment_t::vid_t;
  using vertex_t = typename fragment_t::vertex_t;
  using value_t = VALUE_T;
  using adj_list_t = typename fragment_t::adj_list_t;
  value_t imax = std::numeric_limits<value_t>::max();
  value_t zero = 0;

  void init_c(const FRAG_T& frag, vertex_t v, value_t& delta,
              DenseVertexSet<vid_t>& modified) override {
    vertex_t source;
    bool native_source = frag.GetInnerVertex(FLAGS_sssp_source, source);

    if (native_source && source == v) {  // 判断是否是源点
      delta = 0;
      modified.Insert(source);
    } else {
      delta = imax;
    }
  }

  void init_v(vertex_t v, value_t& value) override { value = imax; }

  bool aggregate(value_t& a, value_t b) override { return atomic_min(a, b); }

  z3::expr aggregate_z3(z3::expr a, z3::expr b) override { return ite(a < b, a, b); }

  bool update(value_t& a, value_t b) override { return aggregate(a, b); }

  bool accumulate(value_t& a, value_t b) override { return aggregate(a, b); }

  void priority(value_t& pri, const value_t& value,
                const value_t& delta) override {
    pri = value - std::min(value, delta);
  }

  value_t generate(value_t v, value_t m, value_t w) override{
    return v + w;
  }

  z3::expr generate_z3(z3::solver& s, z3::expr x) override {
    z3::expr dis = s.ctx().real_const("dis");
    z3::expr val = x + dis;
    return val;
  }

  void g_function(const FRAG_T& frag, const vertex_t v, const value_t& value,
                  const value_t& delta, const adj_list_t& oes,
                  DenseVertexSet<vid_t>& modified) override {
    if (delta != imax) {
      auto src_gid = frag.Vertex2Gid(v);

      if (FLAGS_cilk) {
        auto out_degree = oes.Size();

        auto it = oes.begin();
        granular_for(j, 0, out_degree, (out_degree > 1024), {
          auto& e = *(it + j);
          auto dst = e.neighbor;
          auto outv = generate(0, e.data,  delta);

          if (this->accumulate_to(dst, outv)) {
            this->add_delta_dependency(src_gid, dst);
            modified.Insert(dst);
          }
        })
      } else {
        for (auto e : oes) {
          auto dst = e.neighbor;
          auto outv = e.data + delta;

          if (this->accumulate_to(dst, outv)) {
            this->add_delta_dependency(src_gid, dst);
            modified.Insert(dst);
          }
        }
      }
    }
  }

  value_t default_v() override { return imax; }

  value_t min_delta() override { return zero; }
};

}  // namespace grape

#endif  // ANALYTICAL_APPS_SSSP_SSSP_INGRESS_H_
