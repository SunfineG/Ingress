
#ifndef ANALYTICAL_APPS_PHP_PHP_INGRESS_H_
#define ANALYTICAL_APPS_PHP_PHP_INGRESS_H_

#include "flags.h"
#include "grape/app/ingress_app_base.h"
#include "grape/fragment/immutable_edgecut_fragment.h"

#include "z3++.h"

namespace grape {

template <typename FRAG_T, typename VALUE_T>
class PHPIngress : public IterateKernel<FRAG_T, VALUE_T> {
 public:
  using fragment_t = FRAG_T;
  using vid_t = typename fragment_t::vid_t;
  using vertex_t = typename fragment_t::vertex_t;
  using value_t = VALUE_T;
  using adj_list_t = typename fragment_t::adj_list_t;
  vid_t source_gid;
  VertexArray<typename FRAG_T::edata_t, vid_t> weight_sum;

  void iterate_begin(const FRAG_T& frag) override {
    auto iv = frag.InnerVertices();
    auto source_id = FLAGS_php_source;

    weight_sum.Init(iv, 0);

    for (auto v : iv) {
      auto oes = frag.GetOutgoingAdjList(v);

      for (auto& e : oes) {
        auto dst = e.neighbor;
        auto weight = e.data;

        if (frag.GetId(dst) != source_id) {
          weight_sum[v] += weight;
        }
      }
    }
  }

  void init_c(vertex_t v, value_t& delta, const FRAG_T& frag) override {
    auto source_id = FLAGS_php_source;
    vertex_t source;
    auto native_source = frag.GetInnerVertex(source_id, source);

    CHECK(frag.Oid2Gid(source_id, source_gid));
    if (native_source && source == v) {  // 判断是否是源点
      delta = 1;
    } else {
      delta = 0;
    }
  }
  void init_c(const FRAG_T& frag, const vertex_t v, value_t& delta,
              DenseVertexSet<vid_t>& modified) override {}


  void init_v(const vertex_t v, value_t& value) override { value = 0.0f; }


  bool accumulate(value_t& a, value_t b) override {
    atomic_add(a, b);
    return true;
  }

  bool aggregate(value_t& a, value_t b) override {
    atomic_add(a, b);
    return true;
  }

  z3::expr aggregate_z3(z3::expr a, z3::expr b) override{
    return a + b;
  }

  bool update(value_t& a, value_t b) override {
    return aggregate(a, b);;
  }


  void priority(value_t& pri, const value_t& value,
                const value_t& delta) override {
    pri = delta;
  }

  value_t generate(value_t v, value_t m, value_t w){
    return m * 0.85 * w;
  }

  z3::expr generate_z3(z3::solver& s, z3::expr x){
    z3::expr d = s.ctx().real_val("0.85");
    z3::expr w = s.ctx().real_const("w");
    s.add(w > 0);
    z3::expr val = x * d * w;
    return val;
  }

  void g_function(const FRAG_T& frag, const vertex_t v, const value_t& value,
                  const value_t& delta, const adj_list_t& oes) override {
    if (delta != default_v()) {
      auto out_degree = oes.Size();
      auto it = oes.begin();

      granular_for(j, 0, out_degree, (out_degree > 1024), {
        auto& e = *(it + j);
        auto dst = e.neighbor;

        value_t outv = generate(e.data, 0, weight_sum[v]);
        if (frag.Vertex2Gid(dst) != source_gid) {
          this->accumulate_to(dst, outv);
        }
      })
    }
  }

  void g_function(const FRAG_T& frag, const vertex_t v,
                  const value_t& value, const value_t& delta,
                  const adj_list_t& oes,
                  DenseVertexSet<vid_t>& modified) override {

  }

  value_t default_v() override { return 0; }

  value_t min_delta() override { return 0; }
};

}  // namespace grape

#endif  // ANALYTICAL_APPS_PHP_PHP_INGRESS_H_
