
#ifndef EXAMPLES_ANALYTICAL_APPS_INGRESS_H_
#define EXAMPLES_ANALYTICAL_APPS_INGRESS_H_

#include <gflags/gflags.h>
#include <gflags/gflags_declare.h>
#include <glog/logging.h>
#include <grape/fragment/immutable_edgecut_fragment.h>
#include <grape/fragment/loader.h>
#include <grape/grape.h>
#include <grape/util.h>
#include <grape/worker/async_worker.h>
#include <grape/worker/ingress_sync_iter_worker.h>
#include <grape/worker/ingress_sync_traversal_worker.h>
#include <grape/worker/ingress_sync_worker.h>
#include <sys/stat.h>

#include <algorithm>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

#include "cc/cc_ingress.h"
#include "flags.h"
#include "gcn/gcn.h"
#include "pagerank/pagerank_ingress.h"
#include "php/php_ingress.h"
#include "sssp/sssp_ingress.h"
#include "timer.h"
#include "z3++.h"

namespace grape {

enum Engineer { MF, MP, MV, ME };

void Init() {
  if (FLAGS_vfile.empty() || FLAGS_efile.empty()) {
    LOG(FATAL) << "Please assign input vertex/edge files.";
  }

  if (access(FLAGS_out_prefix.c_str(), 0) != 0) {
    mkdir(FLAGS_out_prefix.c_str(), 0777);
  }

  InitMPIComm();
  CommSpec comm_spec;
  comm_spec.Init(MPI_COMM_WORLD);
  if (comm_spec.worker_id() == kCoordinatorRank) {
    VLOG(1) << "Workers of libgrape-lite initialized.";
  }
}

void Finalize() {
  FinalizeMPIComm();
  VLOG(1) << "Workers finalized.";
}

template <typename FRAG_T, typename APP_T>
void CreateAndQueryTypeOne(std::shared_ptr<APP_T>& app,
                           const CommSpec& comm_spec, const std::string efile,
                           const std::string& vfile,
                           const std::string& out_prefix,
                           const ParallelEngineSpec& spec) {
  timer_next("load graph");
  LoadGraphSpec graph_spec = DefaultLoadGraphSpec();
  graph_spec.set_directed(FLAGS_directed);
  graph_spec.set_rebalance(false, 0);
  SetSerialize(comm_spec, FLAGS_serialization_prefix, efile, vfile, graph_spec);

  auto fragment =
      LoadGraph<FRAG_T, SegmentedPartitioner<typename FRAG_T::oid_t>>(
          efile, vfile, comm_spec, graph_spec);
  timer_next("load application");
  if (comm_spec.worker_id() == grape::kCoordinatorRank) {
    LOG(INFO) << "第一类worker";
  }

  IngressSyncIterWorker<APP_T> worker(app, fragment);
  worker.Init(comm_spec, spec);
  timer_next("run algorithm");
  worker.Query();
  timer_next("print output");

  if (!out_prefix.empty()) {
    std::ofstream ostream;
    std::string output_path =
        grape::GetResultFilename(out_prefix, fragment->fid());
    ostream.open(output_path);
    worker.Output(ostream);
    ostream.close();
    VLOG(1) << "Worker-" << comm_spec.worker_id()
            << " finished: " << output_path;
  }
  worker.Finalize();
  timer_end();
  fragment.reset();
}

template <typename FRAG_T, typename APP_T>
void CreateAndQueryTypeTwo(const CommSpec& comm_spec, const std::string efile,
                           const std::string& vfile,
                           const std::string& out_prefix,
                           const ParallelEngineSpec& spec) {
  timer_next("load graph");
  LoadGraphSpec graph_spec = DefaultLoadGraphSpec();

  graph_spec.set_directed(FLAGS_directed);
  graph_spec.set_rebalance(false, 0);
  SetSerialize(comm_spec, FLAGS_serialization_prefix, efile, vfile, graph_spec);

  auto fragment =
      LoadGraph<FRAG_T, SegmentedPartitioner<typename FRAG_T::oid_t>>(
          efile, vfile, comm_spec, graph_spec);
  auto app = std::make_shared<APP_T>();
  timer_next("load application");
  if (comm_spec.worker_id() == grape::kCoordinatorRank) {
    LOG(INFO) << "第二类worker";
  }
  IngressSyncTraversalWorker<APP_T> worker(app, fragment);
  worker.Init(comm_spec, spec);
  timer_next("run algorithm");
  worker.Query();
  timer_next("print output");

  if (!out_prefix.empty()) {
    std::ofstream ostream;
    std::string output_path =
        grape::GetResultFilename(out_prefix, fragment->fid());
    ostream.open(output_path);
    worker.Output(ostream);
    ostream.close();
    VLOG(1) << "Worker-" << comm_spec.worker_id()
            << " finished: " << output_path;
  }
  worker.Finalize();
  fragment.reset();
  timer_end();
}

template <typename FRAG_T, typename APP_T, typename... Args>
void CreateAndQuery(const CommSpec& comm_spec, const std::string& efile,
                    const std::string& efile_update,
                    const std::string& efile_updated, const std::string& vfile,
                    const std::string& out_prefix,
                    const ParallelEngineSpec& spec, Args... args) {
  timer_next("load graph");
  LoadGraphSpec graph_spec = DefaultLoadGraphSpec();
  graph_spec.set_directed(FLAGS_directed);
  graph_spec.set_rebalance(false, 0);
  SetSerialize(comm_spec, FLAGS_serialization_prefix, efile, vfile, graph_spec);
  using oid_t = typename FRAG_T::oid_t;

  auto fragment = LoadGraph<FRAG_T, SegmentedPartitioner<oid_t>>(
      efile, vfile, comm_spec, graph_spec);
  auto app = std::make_shared<APP_T>();
  timer_next("load application");
  auto worker = APP_T::CreateWorker(app);
  worker->Init(comm_spec, spec);
  worker->SetFragment(fragment);
  worker->Query(std::forward<Args>(args)...);

  timer_next("Reloading graph");

  if (!efile_update.empty() && !efile_updated.empty()) {
    fragment.reset();

    graph_spec = DefaultLoadGraphSpec();
    graph_spec.set_directed(FLAGS_directed);
    graph_spec.set_rebalance(false, 0);
    fragment = LoadGraph<FRAG_T, SegmentedPartitioner<oid_t>>(
        efile_updated, vfile, comm_spec, graph_spec);
    std::vector<std::pair<oid_t, oid_t>> added_edges, deleted_edges;
    auto vm_ptr = fragment->vm_ptr();
    LoadChangeEFile<FRAG_T>(vm_ptr, efile_update, added_edges, deleted_edges);
    worker->SetFragment(fragment);
    worker->Inc(added_edges, deleted_edges);
  }

  timer_next("print output");
  if (!out_prefix.empty()) {
    std::ofstream ostream;
    std::string output_path =
        grape::GetResultFilename(out_prefix, fragment->fid());
    ostream.open(output_path);
    worker->Output(ostream);
    ostream.close();
    VLOG(1) << "Worker-" << comm_spec.worker_id()
            << " finished: " << output_path;
  }
  worker->Finalize();
  timer_end();
  fragment.reset();
}

// template<typename AppType>
bool check_gfg(z3::solver& s,
               const std::function<z3::expr(z3::expr, z3::expr)>& agg,
               const std::function<z3::expr(z3::solver&, z3::expr)>& gen
               //                   z3::expr (AppType::*f1)(z3::expr, z3::expr)
) {
  z3::context& c = s.ctx();
  z3::expr x1 = c.real_const("x1");
  z3::expr x2 = c.real_const("x2");
  z3::expr y1 = c.real_const("y1");
  z3::expr y2 = c.real_const("y2");

  z3::expr conjection =
      (forall(x1, x2, y1, y2,
              (agg(gen(s, agg(x1, y1)), gen(s, agg(x2, y2))) ==
               agg(agg(agg(gen(s, x1), gen(s, y1)), gen(s, x2)), gen(s, y2)))));

  s.add(!conjection);
  if (s.check() == z3::unsat) {
    return true;
  } else {
    return false;
  }
}

bool check_commutative(z3::solver& s,
                       const std::function<z3::expr(z3::expr, z3::expr)>& agg) {

  z3::context& c = s.ctx();
  z3::expr x1 = c.real_const("x1");
  z3::expr x2 = c.real_const("x2");

  z3::expr conjection =
      (forall(x1, x2, (agg(x1, x2) == agg(x2, x1))));

  s.add(!conjection);
  if (s.check() == z3::unsat) {
    return true;
  } else {
    return false;
  }
}

bool check_associative(z3::solver& s,
                     const std::function<z3::expr(z3::expr, z3::expr)>& agg) {
  z3::context& c = s.ctx();
  z3::expr x1 = c.real_const("x1");
  z3::expr x2 = c.real_const("x2");
  z3::expr x3 = c.real_const("x3");

  z3::expr conjection =
      (forall(x1, x2, x3, (agg(agg(x1, x2), x3) == agg(x1, agg(x2, x3)))));

  s.add(!conjection);
  if (s.check() == z3::unsat) {
    return true;
  } else {
    return false;
  }
}

bool check_agg_inverse(z3::solver& s,
                       const std::function<z3::expr(z3::expr, z3::expr)>& agg){

  z3::context &c = s.ctx();
  z3::expr x1 = c.real_const("x1");
  z3::expr x2 = c.real_const("x2");
  z3::expr x3 = c.real_const("x3");
  z3::sort I = c.real_sort();
  z3::func_decl f1 = function("f1", I, I);

  z3::expr conjection = forall(x1, x2, x3, agg(x1, x3) == agg(x1, agg(agg( x2, f1(x2)), x3)));
  std::cout << conjection << std::endl;
  s.add(conjection);
  if(s.check() == z3::sat){
    return true;
  }else{
    return false;
  }
}

bool check_singdep(z3::solver& s,
                   const std::function<z3::expr(z3::expr, z3::expr)>& agg){

  z3::context& c = s.ctx();
  z3::expr x = c.real_const("x");
  z3::expr y = c.real_const("y");

  z3::expr conjecture = (forall(x, y, agg(x, y) == x || agg(x, y) == y));
  std::cout << conjecture << std::endl;
  s.add(!conjecture);
  if(s.check() == z3::unsat){
    return true;
  }else{
    return false;
  }

}

template <typename APP_T>
Engineer choose_engine(z3::solver& s, std::shared_ptr<APP_T>& app) {

  bool is_gfg = check_gfg(s,
      [&app](z3::expr a, z3::expr b) -> z3::expr {
        return app->aggregate_z3(a, b);
      },
      [&app](z3::solver& s, z3::expr e) -> z3::expr {
        return app->generate_z3(s, e);
      });

  bool is_commutative =
      check_commutative(s, [&app](z3::expr a, z3::expr b) -> z3::expr {
        return app->aggregate_z3(a, b);
      });

  bool is_associative = check_associative(s, [&app](z3::expr a, z3::expr b) -> z3::expr {
    return app->aggregate_z3(a, b);
  });

  bool has_agg_inverse = check_agg_inverse(s, [&app](z3::expr a, z3::expr b) -> z3::expr {
    return app->aggregate_z3(a, b);
  });

  bool is_singdep = check_singdep(s, [&app](z3::expr a, z3::expr b) -> z3::expr {
    return app->aggregate_z3(a, b);
    });

  if (is_gfg && is_commutative && is_associative) {
    if(has_agg_inverse){
      return Engineer::MF;
    }else if(is_singdep){
      return Engineer::MP;
    }
  }

  if(has_agg_inverse){
    return Engineer::MV;
  }

  return Engineer::ME;
}

void RunIngress() {
  CommSpec comm_spec;
  comm_spec.Init(MPI_COMM_WORLD);

  bool is_coordinator = comm_spec.worker_id() == kCoordinatorRank;
  timer_start(is_coordinator);

  std::string name = FLAGS_application;
  std::string efile = FLAGS_efile;
  std::string vfile = FLAGS_vfile;
  std::string out_prefix = FLAGS_out_prefix;
  auto spec = DefaultParallelEngineSpec();

  z3::context c;
  z3::solver s(c);

  if (FLAGS_app_concurrency != -1) {
    spec.thread_num = FLAGS_app_concurrency;
    setWorkers(FLAGS_app_concurrency);
  }

  if (access(vfile.c_str(), 0) != 0) {
    LOG(ERROR) << "Can not access vfile, build oid set at runtime";
    vfile = "";
  }

  if (name == "pagerank") {
    using GraphType =
        grape::ImmutableEdgecutFragment<int32_t, uint32_t, grape::EmptyType,
                                        grape::EmptyType>;
    using AppType = grape::PageRankIngress<GraphType, float>;
    auto app = std::make_shared<AppType>();
    // select engine automatically

    CreateAndQueryTypeOne<GraphType, AppType>(app, comm_spec, efile, vfile,
                                              out_prefix, spec);
  } else if (name == "sssp") {
    using GraphType =
        grape::ImmutableEdgecutFragment<int32_t, uint32_t, grape::EmptyType,
                                        uint16_t, LoadStrategy::kOnlyOut>;
    using AppType = grape::SSSPIngress<GraphType, uint32_t>;

    auto app = std::make_shared<AppType>();
    // select engine automatically

    CreateAndQueryTypeTwo<GraphType, AppType>(comm_spec, efile, vfile,
                                              out_prefix, spec);
  } else if (name == "cc") {
    using GraphType =
        grape::ImmutableEdgecutFragment<int32_t, uint32_t, grape::EmptyType,
                                        grape::EmptyType,
                                        LoadStrategy::kOnlyOut>;
    using AppType = grape::CCIngress<GraphType, int32_t>;

    auto app = std::make_shared<AppType>();
    // select engine automatically

    CreateAndQueryTypeTwo<GraphType, AppType>(comm_spec, efile, vfile,
                                              out_prefix, spec);
  } else if (name == "php") {
    using GraphType = grape::ImmutableEdgecutFragment<int32_t, uint32_t,
                                                      grape::EmptyType, float>;
    using AppType = grape::PHPIngress<GraphType, float>;

    auto app = std::make_shared<AppType>();
    // select engine automatically

    CreateAndQueryTypeOne<GraphType, AppType>(app, comm_spec, efile, vfile,
                                              out_prefix, spec);
  } else if (name == "gcn") {
    using GraphType =
        grape::ImmutableEdgecutFragment<int32_t, uint32_t, grape::EmptyType,
                                        grape::EmptyType,
                                        LoadStrategy::kOnlyOut>;
    using AppType = grape::GCN<GraphType>;
    CreateAndQuery<GraphType, AppType>(comm_spec, efile, FLAGS_efile_update,
                                       FLAGS_efile_updated, vfile, out_prefix,
                                       spec, comm_spec, FLAGS_gcn_mr);
  } else {
    LOG(INFO) << "No this application: " << name;
  }
}
}  // namespace grape

#endif  // EXAMPLES_ANALYTICAL_APPS_INGRESS_H_
